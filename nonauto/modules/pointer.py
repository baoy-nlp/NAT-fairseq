import torch
import torch.nn as nn
from torch.nn import Parameter

from nonauto.modules.attention import FFNAttention


class PointerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, repeat=False):
        super(PointerDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.repeat = repeat

        self.input0 = Parameter(torch.Tensor(input_dim).float(), requires_grad=True)
        self.input_to_hidden = nn.Linear(input_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)

        self.att = FFNAttention(hidden_dim, hidden_dim)
        nn.init.uniform_(self.input0, -1, 1)

    def forward_step(self, x, hidden_states, ctx, mask):
        h, c = hidden_states
        gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
        inputs, forget, cell, out = gates.chunk(4, 1)
        inputs = inputs.sigmoid()
        forget = forget.sigmoid()
        cell = cell.tanh()
        out = out.sigmoid()
        c_t = (forget * c) + (inputs * cell)
        h_t = out * c_t.tanh()

        # Attention section
        attn, attn_weight = self.att(h_t, ctx, mask.eq(0))
        attn = (self.hidden_out(torch.cat((attn, h_t), 1))).tanh()
        return attn, c_t, attn_weight

    def forward(self, inputs, hidden0, candidate, candidate_mask=None, force_index=None, force_mask=None):
        # TODO: REMOVE HIDDEN 0
        batch_size, input_length, _ = inputs.size()
        if candidate_mask is not None:
            mask = candidate_mask.float()
        else:
            mask = torch.ones(batch_size, input_length).float()
        nat_index = torch.arange(input_length).unsqueeze(0).expand(batch_size, -1).long()
        if inputs.is_cuda:
            mask = mask.cuda(inputs.get_device())
            nat_index = nat_index.cuda(inputs.get_device())
        max_len = mask.sum(dim=-1).view(-1, 1)
        self.att.init_inf(mask.size())

        outputs, pointers = [], []
        input0 = self.input0.unsqueeze(0).expand(batch_size, -1)
        hidden0 = hidden0
        for cur_len in range(input_length):
            h_t, c_t, scores = self.forward_step(input0, hidden0, candidate, mask)
            hidden0 = (h_t, c_t)
            masked_scores = scores.softmax(dim=-1) * mask
            max_probs, select_index = masked_scores.max(1)

            is_out_index = max_len.gt(cur_len).long()
            select_index = (select_index.unsqueeze(1) * is_out_index + (1 - is_out_index) * cur_len)
            if force_index is not None:
                select_index = force_index[:, cur_len].contiguous().unsqueeze(1)
            pointers.append(select_index)
            one_hot_pointers = nat_index.eq(select_index.expand(-1, scores.size(1))).float()
            if not self.repeat:
                mask = mask * (1 - one_hot_pointers)
            input0_select = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.input_dim).byte()
            input0 = inputs[input0_select.data].view(batch_size, self.input_dim)
            outputs.append(scores.unsqueeze(0))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)
        return (outputs, pointers), hidden0

    def beam_search(self, inputs, hidden0, candidate, candidate_mask=None, beam_size=2):
        """
        Args:
            inputs: [batch_size,input_length,hidden]
            hidden0: [batch_size,hidden]
            candidate: [batch_size,input_length, hidden]
            candidate_mask: [batch_size,input_length]
            beam_size:

        Returns:

        """
        W = beam_size
        B, T, _ = inputs.size()
        if candidate_mask is not None:
            mask = candidate_mask.float()
        else:
            mask = torch.ones(B, T).float()
        mask = mask[:, None, :].expand(B, W, T)  # .contiguous().view(B * W, T)
        alive = torch.arange(T).unsqueeze(0).expand(B, -1).long()
        if inputs.is_cuda:
            mask = mask.cuda(inputs.get_device())
            alive = alive.cuda(inputs.get_device())
        alive = alive[:, None, :].expand(B, W, T)  # .contiguous().view(B * W, T)
        self.att.init_inf(mask.contiguous().view(B * W, T).size())
        inputs = inputs.unsqueeze(1).expand(-1, beam_size, -1, -1).contiguous().view(B * W, T, -1)
        input0 = self.input0.unsqueeze(1).expand(-1, beam_size, -1).contiguous().view(B * beam_size, -1)
        h_t, c_t = hidden0
        h_t = h_t.unsqueeze(1).expand(-1, beam_size, -1).contiguous().view(B * beam_size, -1)
        c_t = c_t.unsqueeze(1).expand(-1, beam_size, -1).contiguous().view(B * beam_size, -1)
        hidden0 = (h_t, c_t)

        candidate = candidate.unsqueeze(1).expand(B, beam_size, T, -1).contiguous().view(B * beam_size, T, -1)
        logps = inputs.data.new(B, beam_size).float().fill_(0)
        outs = inputs.data.new(B, W, T + 1).long().fill_(-1)
        max_len = mask.sum(dim=-1).view(-1, 1)

        for cur_len in range(T):
            use_mask = mask.contiguous().view(B * W, T)
            h_t, c_t, scores = self.forward_step(input0, hidden0, candidate, use_mask)
            masked_outs = (scores - (1 - use_mask.float()) * 1e10).log_softmax(dim=-1)
            topk2_logps, topk2_inds = masked_outs.contiguous().view(B, W, -1).topk(W, dim=-1)
            topk2_logps = topk2_logps + logps[:, :, None]
            if cur_len == 0:
                logps, topk_inds = topk2_logps[:, 0].topk(W, dim=-1)
            else:
                logps, topk_inds = topk2_logps.view(B, W * W).topk(W, dim=-1)
            topk_beam_inds = topk_inds.div(W)
            topk_token_inds = topk2_inds.view(B, W * W).gather(1, topk_inds)  # B, W
            is_allow = max_len.gt(cur_len).long().view(*topk_token_inds.size())
            topk_token_inds = (topk_token_inds * is_allow + (1 - is_allow) * cur_len)
            outs = outs.gather(1, topk_beam_inds[:, :, None].expand_as(outs))
            outs[:, :, cur_len + 1] = topk_token_inds.squeeze(1)

            mask = mask.gather(1, topk_beam_inds[:, :, None].expand_as(mask))
            alive = alive.gather(1, topk_beam_inds[:, :, None].expand_as(alive))
            use_alive = alive.contiguous().view(B * W, T)
            one_hot_pointers = (use_alive == topk_token_inds.contiguous().view(-1, 1).expand(-1, T)).float()
            mask = mask * (1 - one_hot_pointers.view(B, W, T))
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.input_dim).byte()
            input0 = inputs[embedding_mask.data].view(B * W, self.input_dim)
            hidden0 = (h_t, c_t)

        return outs[:, 0, 1:]


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, bidirectional):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim // 2 if bidirectional else hidden_dim
        self.n_layers = num_layers * 2 if bidirectional else num_layers
        self.bidirectional = bidirectional
        self.pointer_enc = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=self.bidirectional,
        )

        self.h0 = Parameter(torch.zeros(1), requires_grad=True)
        self.c0 = Parameter(torch.zeros(1), requires_grad=True)
        nn.init.uniform_(self.h0, -1, 1)
        nn.init.uniform_(self.c0, -1, 1)

    def forward(self, inputs, mask_inputs=None):
        # TODO: ADD MASK INPUTS PROCESS HERE
        # hidden = self.init_hidden(inputs)
        hidden = None
        inputs = inputs.contiguous().permute(1, 0, 2)  # to [seq_len, bsz, hidden_dim]
        outputs, hidden = self.pointer_enc(inputs, hidden)
        return outputs.contiguous().permute(1, 0, 2), self.return_hidden(hidden)

    def init_hidden(self, inputs):
        """
        Initiate hidden units

        :param Tensor inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = inputs.size(0)
        h0 = self.h0.contiguous().view(1, 1, 1).expand(self.n_layers, batch_size, self.hidden_dim)
        c0 = self.c0.contiguous().view(1, 1, 1).expand(self.n_layers, batch_size, self.hidden_dim)
        return h0, c0

    def return_hidden(self, hidden):
        if self.bidirectional:
            hidden0 = (hidden[0][-2:].mean(dim=0), hidden[1][-2:].mean(dim=0))
        else:
            hidden0 = (hidden[0][-1], hidden[1][-1])
        return hidden0


class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, bidirectional=False, repeat=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout, bidirectional)
        self.decoder = PointerDecoder(input_dim, hidden_dim, repeat)

    def forward(self, inputs, mask_inputs=None, inputs_index=None, beam_size=1):
        batch_size, input_size, _ = inputs.size()
        encoder_outputs, hidden0 = self.encoder(inputs)
        # TODO: FIX BIDIRECTIONAL Module

        if inputs_index is not None:
            rank_id = inputs_index.long().sort()[1]
        else:
            rank_id = None

        if beam_size > 1:
            pointers = self.decoder.beam_search(
                inputs,
                hidden0=hidden0,
                candidate=encoder_outputs,
                candidate_mask=mask_inputs,
                beam_size=beam_size
            )
            return pointers

        else:
            (outputs, pointers), decoder_hidden = self.decoder(
                inputs,
                hidden0=hidden0,
                candidate=encoder_outputs,
                candidate_mask=mask_inputs,
                force_index=rank_id,
            )
            return outputs, pointers
