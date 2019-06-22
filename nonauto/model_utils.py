import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

INF = 1e10
TINY = 1e-9


class SharedModule(object):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **unused):
        return self.module.forward(*args, **unused)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    # nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    # nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class OutLinear(nn.Linear):
    def __init__(self, d_in, d_out, bias=True, out_norm=False, _weight=None):
        super(OutLinear, self).__init__(d_in, d_out, bias)
        self.out_norm = out_norm
        if _weight is None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            init.uniform_(self.weight, -stdv, stdv)
        else:
            self.weight = _weight
        if bias:
            self.bias.data.zero_()

    def forward(self, x):
        size = x.size()
        if self.out_norm:
            weight = self.weight / (1e-6 + torch.sqrt((self.weight ** 2).sum(0, keepdim=True)))
            x_ = x / (1e-6 + torch.sqrt((x ** 2).sum(-1, keepdim=True)))
            logit_ = torch.mm(x_.contiguous().view(-1, size[-1]), weight.t()).view(*size[:-1], -1)
            if self.bias:
                logit_ = logit_ + self.bias
            return logit_
        return super().forward(x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)


def positional_encodings_like(x, t=None):  # hope to be differentiable
    if t is None:
        positions = torch.arange(0, x.size(-2))  # .expand(*x.size()[:2])
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
        positions = Variable(positions.float())
    else:
        positions = t

    # channels
    channels = torch.arange(0, x.size(-1), 2) / x.size(-1)  # 0 2 4 6 ... (256)
    channels = channels.float()
    if x.is_cuda:
        channels = channels.cuda(x.get_device())
    channels = 1 / (10000 ** Variable(channels))

    # get the positional encoding: batch x target_len
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)  # batch x target_len x 256
    encodings = torch.cat([torch.sin(encodings).unsqueeze(-1), torch.cos(encodings).unsqueeze(-1)], -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)  # batch x target_len x 512

    if encodings.ndimension() == 2:
        encodings = encodings.unsqueeze(0).expand_as(x)

    return encodings


def softmax(x, T=1):
    return F.softmax(x / T, dim=-1)


def log_softmax(x):
    if x.dim() == 3:
        return F.log_softmax(x.transpose(0, 2)).transpose(0, 2)
    return F.log_softmax(x)


def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-1)).squeeze(-1)


def apply_mask(inputs, mask, p=1):
    _mask = Variable(mask.type(inputs.type()))
    outputs = inputs * _mask + (torch.mul(_mask, -1) + 1) * p
    return outputs


def sequence_mask(sequence_len, max_len=None):
    if max_len is None:
        max_len = int(sequence_len.max().item())

    batch_size = sequence_len.size(0)
    init_index = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)
    if sequence_len.is_cuda:
        init_index = init_index.cuda(sequence_len.get_device())
    final_mask = init_index.lt(sequence_len.view(batch_size, -1)).long()
    return final_mask


def project_mask(inputs1, inputs2):
    return (inputs1[:, :, None] @ inputs2[:, None, :]).long()


def spread_mask(inputs1, inputs2):
    return (inputs1[:, :, None] + inputs2[:, None, :]).gt(0).long()


def self_project_mask(inputs):
    """
    inputs = ([
            [1., 0., 0.],
            [1., 1., 0.]
        ])
    return:
        ([
            [
                [1., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]
            ],

            [
                [1., 1., 0.],
                [1., 1., 0.],
                [0., 0., 0.]
            ]
        ])
    """
    return project_mask(inputs, inputs)


def self_spread_mask(inputs):
    """
    inputs = ([
            [1., 0., 0.],
            [1., 1., 0.]
        ])
    return:
        ([
            [
                [1, 1, 1],
                [1, 0, 0],
                [1, 0, 0]
            ],

            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 0]
            ]
        ])

    """
    return spread_mask(inputs, inputs)


def diagonal_mask(inputs):
    batch_size = inputs.size(0)
    seq_len = inputs.size(1)
    mask = torch.eye(seq_len).long()
    if inputs.is_cuda:
        mask = mask.cuda(inputs.get_device())
    return (1 - mask).unsqueeze(0).expand(batch_size, *mask.size())


def mask_score(score, candidate=None, dim=-1):
    if candidate is not None:
        score = score - INF * (1 - candidate.float())
    probs = score.softmax(dim)
    return probs


def line_select(probs, dim=-1, ret_onehot=True, ret_score=False):
    select_score, select_index = probs.max(dim)

    if ret_onehot:
        select_index = index_to_onehot(probs, select_index)
    if ret_score:
        return select_score, select_index
    else:
        return select_index


def max_select(score, candidate=None, dim=-1, ret_onehot=True, ret_score=False):
    if candidate is not None:
        score = score - INF * (1 - candidate.float())
    probs = score.softmax(dim)
    select_score, select_index = probs.max(dim)

    if ret_onehot:
        select_index = index_to_onehot(probs, select_index)
    if ret_score:
        return select_score, select_index
    else:
        return select_index


def index_to_onehot(score, index):
    """
    :param score: batch, seq, seq
    :param index: batch, seq
    :return: batch, seq, seq
    """
    shape = score.size()
    score = score.contiguous().view(-1, score.size(-1))
    one_hot = score.new_zeros(*score.size()).scatter_(-1, index.contiguous().view(-1, 1), 1.0)
    return one_hot.contiguous().view(*shape)


def onehot_to_index(onehot, start_index=0):
    shape = onehot.size()
    num_length = shape[-1]
    tgt_sequence = torch.arange(start_index, start_index + num_length).view(-1, 1).float()
    if onehot.is_cuda:
        tgt_sequence = tgt_sequence.cuda(onehot.get_device())
    onehot = onehot.contiguous().view(-1, num_length)
    tgt_index = matmul(onehot, tgt_sequence)
    tgt_index = tgt_index.contiguous().view(*shape[:-1])
    if tgt_index.dim() > 2:
        tgt_index = tgt_index.squeeze(-1)
    return tgt_index


def non_auto_max_connect_search(logits, mask_rows, mask_cows, share_prob=False):
    """
        Non-autoregressive determine index

    :param logits: batch, seq_len , seq_len
    :param mask_rows: batch, mask_rows
    :param mask_cows: batch, mask_cows
    :param share_prob
    :return:
    """
    matched_one_hot = 0
    search_iter = 0
    row_candidate = mask_rows.float()
    cow_candidate = mask_cows.float()
    condition = row_candidate * cow_candidate
    condition_item = condition.sum().item()
    while condition_item > 0:
        # determine the candidate elements using the row_candidate and cow_candidate
        candidate = project_mask(row_candidate, cow_candidate).float()

        if share_prob:
            probs = mask_score(score=logits, candidate=candidate)
            # select the largest element by row
            row_selected = line_select(probs)
            # select the largest element by column
            cow_selected_t = line_select(probs.transpose(1, 2))
        else:
            # select the largest element by row
            row_selected = max_select(score=logits, candidate=candidate)
            # select the largest element by column
            cow_selected_t = max_select(score=logits.transpose(1, 2), candidate=candidate.transpose(1, 2))
        cow_selected = cow_selected_t.transpose(1, 2)
        # combine the row selection and column selection
        sub_one_hot = (row_selected * cow_selected) * candidate
        # process the result
        matched_one_hot = matched_one_hot + sub_one_hot

        # update for next iter
        search_iter += 1
        row_candidate = row_candidate * (sub_one_hot.sum(dim=2).lt(1).float())
        cow_candidate = cow_candidate * (sub_one_hot.sum(dim=1).lt(1).float())
        condition = row_candidate * cow_candidate
        condition_item = condition.sum().item()
        del row_selected, cow_selected_t, cow_selected, candidate

    # compute the selected with shape 'batch, seq, seq'
    matched_index = onehot_to_index(matched_one_hot).long()  # [0, seq_len)
    default_index = torch.arange(0, mask_cows.size(1))  # [0, seq_len)
    if matched_index.is_cuda:
        default_index = default_index.cuda(matched_index.get_device())
    default_index = default_index.contiguous().expand(*matched_index.size())
    matched_index = matched_index * mask_cows.long() + default_index * (1 - mask_cows.long())

    return {
        'iter': search_iter,
        'index': matched_index,
        'one_hot': matched_one_hot,
    }


def hitting(logits, gumbel=True):
    y_soft = logits.contiguous().view(-1, logits.size(-1))
    shape = y_soft.size()
    _, k = y_soft.max(-1)
    y_hard = y_soft.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
    if gumbel:
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_hard
    return y.contiguous().view(*logits.size())  # [batch_size, seq_len, vocab_size]


def previous_max_connect_search(match_logits, mask_rows=None, mask_cows=None, **kwargs):
    """
    recursive mapping each state to a position

    Args:
        match_logits: batch_size, dec_len, tgt_len
        mask_rows: batch_size, dec_len, tgt_len
        mask_cows: batch_size, dec_len, tgt_len
    Returns:

    """

    # prob_mask = mask_dec.float().unsqueeze(-1) * mask_tgt.float().unsqueeze(-2)
    # probs = get_prob(match_logits, prob_mask, gumbel)

    be_selected = (1 - mask_rows.float()).unsqueeze(-1) * (1 - mask_cows.float()).unsqueeze(-2)
    select_iters = 0
    # _sub_row_hits = hitting(probs, gumbel)  # [batch_size, seq_lenm, tgt_len]
    # _sub_cow_hits = hitting(probs.transpose(-1, -2), gumbel).transpose(-1, -2)  # [batch_size, seq_len, tgt_len]
    # final_hits += _sub_row_hits * _sub_cow_hits  # [batch_size, seq_len, tgt_len] 1/0
    row_candidate = be_selected.sum(dim=2).lt(1).float()  # [batch_size, seq_len]
    cow_candidate = be_selected.sum(dim=1).lt(1).float()  # [batch_size, tgt_len]
    condition_row = row_candidate.sum().item()
    condition_cow = cow_candidate.sum().item()
    while condition_row > 0 and condition_cow > 0:
        select_iters += 1
        _state_masks = row_candidate[:, :, None] @ cow_candidate[:, None, :]  # indicat could select
        # _probs = probs * _state_masks - (1 - _state_masks) * INF
        _probs = mask_score(match_logits, _state_masks) * _state_masks
        _sub_row_hits = hitting(_probs, gumbel=False)
        _sub_cow_hits = hitting(_probs.transpose(-1, -2), gumbel=False).transpose(-1, -2)
        _state_hits = _sub_row_hits * _sub_cow_hits
        be_selected += _state_hits * _state_masks

        row_candidate = be_selected.sum(dim=2).lt(1).float()  # [batch_size, seq_len]
        cow_candidate = be_selected.sum(dim=1).lt(1).float()  # [batch_size, tgt_len]
        condition_row = row_candidate.sum().item()
        condition_cow = cow_candidate.sum().item()
        del _probs, _state_masks, _sub_cow_hits, _sub_row_hits, _state_hits

    index = onehot_to_index(be_selected).view(match_logits.size(0), -1)
    index = index.long()
    raw_index = torch.arange(0, mask_cows.size(1))
    raw_index = raw_index.contiguous().expand(*mask_cows.size())
    if index.is_cuda:
        raw_index = raw_index.cuda(index.get_device())
    index = index * mask_rows.long() + raw_index * (1 - mask_rows.long())

    return {
        'iter': select_iters,
        'index': index,
        'one_hot': be_selected,
    }


def rearrange_tensor_with_index(states, index=None):
    """

    :param states: batch_size, seq_len,
    :param index:
    :return:
    """
    if index is None:
        return states
    order_selection = index.sort(dim=-1)[1]

    return rearrange_tensor_with_reorder(states, order_selection)


def rearrange_tensor_with_reorder(states, order_selection=None):
    """
    :param states: batch_size, seq_len,
    :param order_selection:
    :return:
    """
    if order_selection is None:
        return states
    return states.gather(1, order_selection.unsqueeze(-1).expand(*states.size()))
