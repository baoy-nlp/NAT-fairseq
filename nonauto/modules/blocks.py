import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
# from fairseq.modules import LayerNorm
from nonauto.model_utils import (
    Linear,
    positional_encodings_like,
    INF
)
from nonauto.modules.attention import MultiheadAttention, get_attention_cls


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, act_drop=0.0):
        super().__init__()
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)
        self.act_dropout = nn.Dropout(act_drop)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        h = self.act_dropout(h)
        return self.linear2(h)


class NonresidualBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, residual, prev):
        return prev


class ResidualBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, residual, prev):
        return residual + prev


class HighwayBlock(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.gate = FeedForward(d_model, d_hidden)

    def forward(self, residual, prev):
        g = self.gate(residual).sigmoid()
        return residual * g + prev * (1 - g)


RESIDUAL_CLS_DICT = {
    'residual': ResidualBlock,
    'nonresidual': NonresidualBlock,
    'highway': HighwayBlock
}


def get_residual_block(residual_type, d_model, d_hidden):
    return RESIDUAL_CLS_DICT[residual_type](d_model=d_model, d_hidden=d_hidden)


class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, activation_fn='relu', activation_dropout=0.0,
                 normalize_before=False, export=False, residual_type='residual'):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.activation_fn = utils.get_activation_fn(activation=activation_fn)
        self.activation_dropout = activation_dropout if activation_dropout > 0 else dropout
        self.normalize_before = normalize_before
        self.fc1 = Linear(self.input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, self.input_dim)
        self.ffn_layer_norm = LayerNorm(self.input_dim, export=export)
        self.residual_block = get_residual_block(residual_type, input_dim, hidden_dim)

    def forward(self, x):

        residual = x

        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.residual_block(residual, x)

        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class AttentionBlock(nn.Module):
    def __init__(self, attn, normalize_before):
        super().__init__()
        self.attn = attn
        self.normalize_before = normalize_before

    def forward(self, **kwargs):
        raise NotImplementedError

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def set_input_buffer(self, *inputs):
        self.attn.set_input_buffer(inputs)

    def get_input_buffer(self, *inputs):
        return self.attn.get_input_buffer(inputs)


class SelfAttentionBlock(AttentionBlock):
    def __init__(self, input_dim, num_heads, dropout=0.0, normalize_before=False, add_bias_kv=False,
                 add_zero_attn=False, attn_type='multi_attn', mask_future=True, mask_self=False, export=False,
                 residual_type='residual'):
        attention_cls = get_attention_cls(attn_type)
        self_attn = attention_cls(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn
        )
        super(SelfAttentionBlock, self).__init__(attn=self_attn, normalize_before=normalize_before)
        self.self_attn_layer_norm = LayerNorm(input_dim, export=export)
        self.dropout = dropout
        self.mask_future = mask_future
        self.mask_self = mask_self
        self.residual_block = get_residual_block(residual_type, input_dim, input_dim * 2)

    def forward(self, x, padding_mask, attn_mask=None, ret_attn=False, **kwargs):
        if attn_mask is None:
            attn_mask = self.build_masks(x)
        if 'value' in kwargs:
            value = kwargs['value']
            kwargs.pop('value')
            residual = value
        else:
            value = x
            residual = x

        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, attn = self.attn(query=x, key=x, value=value, key_padding_mask=padding_mask, attn_mask=attn_mask,
                            **kwargs)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = residual + x
        x = self.residual_block(residual, x)
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
        if ret_attn:
            return x, attn
        else:
            return x

    def build_masks(self, key):
        sequence_length = key.size(0)
        self_masks = key.data.new(sequence_length, sequence_length).fill_(0).float()
        if self.mask_future:
            self_masks.fill_(-INF).triu(1)
        if self.mask_self:
            diag_masks = torch.eye(sequence_length)
            if key.is_cuda:
                diag_masks = diag_masks.cuda(key.get_device())
            self_masks = self_masks - INF * diag_masks
        return self_masks


class InterAttentionBlock(AttentionBlock):
    def __init__(self, input_dim, num_heads, dropout=0.0, normalize_before=False, export=False,
                 residual_type='residual'):
        inter_attn = MultiheadAttention(
            input_dim, num_heads,
            dropout=dropout,
        )
        super(InterAttentionBlock, self).__init__(attn=inter_attn, normalize_before=normalize_before)
        self.input_dim = input_dim
        self.dropout = dropout
        self.inter_attn_layer_norm = LayerNorm(self.input_dim, export=export)
        self.normalize_before = normalize_before
        self.residual_block = get_residual_block(residual_type, input_dim, input_dim * 2)

    def forward(self, query, key, value, padding_mask=None, ret_attn=False, **kwargs):
        residual = query
        query = self.maybe_layer_norm(self.inter_attn_layer_norm, query, before=True)
        query, attn = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=padding_mask,
            **kwargs
        )
        query = F.dropout(query, p=self.dropout, training=self.training)
        # query = residual + query
        query = self.residual_block(residual, query)

        query = self.maybe_layer_norm(self.inter_attn_layer_norm, query, after=True)
        if ret_attn:
            return query, attn
        else:
            return query


class ConditionRNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, num_heads=8, num_layers=2, bidirectional=True,
                 positional=False, mask_future=False, mask_self=True, export=False, **kwargs):
        super().__init__()
        self.self_encoder = nn.LSTM(
            num_layers=num_layers,
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        if bidirectional:
            self.map_to_model = nn.Sequential(
                nn.Dropout(dropout),
                Linear(hidden_dim * 2, input_dim, bias=True)
            )
        else:
            self.map_to_model = None

        if positional:
            self.pos_slf_attn = SelfAttentionBlock(
                input_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                mask_future=mask_future,
                mask_self=mask_self,
                export=export
            )
        else:
            self.pos_slf_attn = None

        self.cond_attn = InterAttentionBlock(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            export=export
        )
        self.feedforward = FeedForwardBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation_dropout=dropout,
            export=export
        )

    def encode(self, input_states, mask_input=None):
        """
        :param input_states: seq_len, batch_size, depth
        :param mask_input: batch_size, seq_len
        :return:
        """
        input_states = input_states.permute(1, 0, 2)
        self.self_encoder.flatten_parameters()
        total_length = input_states.size(1)
        if mask_input is not None:
            length_tensor = mask_input.sum(dim=-1).long()
            sorted_length, sorted_ids = (-length_tensor).sort(dim=-1)
            inputs = input_states[sorted_ids, :, :]
            # print(inputs.size())
            pack = nn.utils.rnn.pack_padded_sequence(inputs, (-sorted_length).tolist(), batch_first=True)
            out, _ = self.self_encoder.forward(pack)
            # out = self.test_lstm(pack)
            unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=total_length)
            # print(unpacked.size())
            recover_ids = sorted_ids.sort(dim=-1)[1]
            output = unpacked[recover_ids, :, :]
        else:
            output = self.self_encoder.forward(input_states)
        if self.map_to_model is not None:
            output = self.map_to_model(output)
        output = output.permute(1, 0, 2)
        return output

    def forward(self, input_states, ctx_states, mask_input, mask_ctx, **kwargs):
        """ return shape 'seq_len, batch_size, depth '"""
        output = self.encode(input_states, mask_input)
        self_padding_mask = (1 - mask_input).byte()
        if self.pos_slf_attn is not None:
            pos_encoding = positional_encodings_like(output)
            output = self.pos_slf_attn(x=pos_encoding, value=output, padding_mask=self_padding_mask)
        key_padding_mask = (1 - mask_ctx).byte()
        output = self.cond_attn(output, ctx_states, ctx_states, key_padding_mask)
        output = self.feedforward(output)
        return output


class ConditionATTLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, num_heads=8, position_init=True, positional=False,
                 mask_future=False, mask_self=True, export=False, **kwargs):
        super(ConditionATTLayer, self).__init__()
        self.pos_init = position_init
        self.self_encoder = SelfAttentionBlock(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            mask_future=mask_future,
            mask_self=mask_self,
            export=export
        )
        if positional:
            self.pos_slf_attn = SelfAttentionBlock(
                input_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                mask_future=mask_future,
                mask_self=mask_self,
                export=export
            )
        else:
            self.pos_slf_attn = None

        self.cond_attn = InterAttentionBlock(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            export=export,
        )

        self.feedforward = FeedForwardBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation_dropout=dropout,
            export=export,
        )

    def forward(self, input_states, ctx_states, mask_input, mask_ctx, **kwargs):
        """ return shape 'seq_len, batch_size, depth '"""
        if self.pos_init:
            pos_encoding = positional_encodings_like(input_states)
            output = input_states + pos_encoding
        else:
            pos_encoding = None
            output = input_states
        self_padding_mask = (1 - mask_input).byte()
        output = self.self_encoder(x=output, padding_mask=self_padding_mask)
        if self.pos_slf_attn is not None:
            if pos_encoding is None:
                pos_encoding = positional_encodings_like(output)
            output = self.pos_slf_attn(x=pos_encoding, value=output, padding_mask=self_padding_mask)
        inter_padding_mask = (1 - mask_ctx).byte()
        output = self.cond_attn(output, ctx_states, ctx_states, padding_mask=inter_padding_mask, static_kv=True)
        output = self.feedforward(output)
        return output


def get_sublayer_cls(cls_type_name):
    query_dict = {
        'cond-rnn': ConditionRNNLayer,
        'cond-attn': ConditionATTLayer,
    }
    if cls_type_name in query_dict:
        return query_dict[cls_type_name]
    else:
        raise RuntimeError("{} is not be implements".format(cls_type_name))
