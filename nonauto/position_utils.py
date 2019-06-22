import numpy as np
import torch


# TODO: need check the code

def consistent_direction_predict(logits, use_prob=False):
    """
    """
    if use_prob:
        logits = logits.softmax(dim=-1)
    sym_logits = logits.transpose(1, 2)
    return logits + sym_logits


def consistent_analysis(relative_mat, accept_threshold=0.5):
    negative_mask = relative_mat.lt(0).long()
    positive_mask = relative_mat.gt(0).long()
    negative_sum = negative_mask.sum(dim=-1)
    positive_sum = positive_mask.sum(dim=-1)
    consistent_sum = ((negative_sum.unsqueeze(-1) + positive_sum.unsqueeze(-2)) * positive_mask + (
            positive_sum.unsqueeze(-1) + negative_sum.unsqueeze(-2)) * negative_mask).float() + 0.01
    # consistent_sum2 = (negative_sum.unsqueeze(-1) * positive_mask + positive_sum.unsqueeze(
    #     -1) * negative_mask + negative_sum.unsqueeze(-2) * positive_mask + positive_sum.unsqueeze(
    #     -2) * negative_mask).float() + 0.01
    positive_match = positive_mask.unsqueeze(-2).eq(positive_mask.unsqueeze(-3)).long()
    positive_match *= positive_mask.unsqueeze(dim=-1)
    positive_match_sum = positive_match.sum(dim=-1)
    negative_match = negative_mask.unsqueeze(-2).eq(negative_mask.unsqueeze(-3)).long()
    negative_match *= negative_mask.unsqueeze(dim=-1)
    negative_match_sum = negative_match.sum(dim=-1)
    # consistent_match1 = ((negative_match_sum.unsqueeze(-1) + positive_match_sum.unsqueeze(-2)) * positive_mask + (
    #         positive_match_sum.unsqueeze(-1) + negative_match_sum.unsqueeze(-2)) * negative_mask).float() + 0.01
    consistent_match = (negative_match_sum + positive_match_sum - negative_mask - positive_mask).float() + 0.01
    # consistent_match3 = ((negative_match_sum + positive_match_sum) * positive_mask + (
    #         positive_match_sum + negative_match_sum) * negative_mask).float() + 0.01
    accuracy = (consistent_match.float() + 0.01) / (consistent_sum.float() + 0.01)
    # accuracy = None
    return accuracy, (accuracy < accept_threshold).long()


def consistent_sort_to_index(relative_mat, accept_threshold=0.5, consistent_num=1, index_type='straight'):
    accept_rate, reject_mask = consistent_analysis(relative_mat, accept_threshold)
    reject_number = reject_mask.eq(1).long().sum().item()
    accept_mat = relative_mat
    # print('reject:', reject_number)
    iter_ = 0
    while reject_number > 0 and iter_ < consistent_num:
        iter_ += 1
        accept_mask = -reject_mask.long() + reject_mask.eq(0).long()
        accept_mat = accept_mat * accept_mask
        _, reject_mask = consistent_analysis(accept_mat, accept_threshold)
        current_num = reject_mask.eq(1).long().sum().item()
        reduce_num = reject_number - current_num
        print('\treduce:', reduce_num, " to: ", current_num)
        reject_number = current_num
    if index_type == 'quick':
        return quick_sort_to_index(relative_pos=accept_mat)[0], iter_
    elif index_type == 'straight':
        return straight_to_index(accept_mat), iter_
    else:
        return accept_mat


def straight_to_index(relative_mat):
    abs_index = relative_mat.gt(0).long().sum(dim=-1)
    sort_index = abs_index.sort(dim=-1)[1]
    return sort_index.sort(dim=-1)[1]


def _quick_sort(rel_mat, init_index=None, index_set=[]):
    """
    quick sort for position convert

    Args:
        rel_mat: [seq_len, seq_len]
        init_index:
        index_set:
    Returns:
        re-index of origin inputs

    Examples:
        rel_pos = [[0, 2, 1, -1], [-2, 0, -1, -3], [-1, 1, 0, -2], [1, 3, 2, 0]]
        print(quick_sort_with_rel_mat(rel_pos, index_set=[0, 1, 2, 3]))
        1 2 0 3

    """
    if len(index_set) == 0:
        return index_set
    if init_index is None:
        init_index = index_set[0]
        index_set = index_set[1:]
    elif init_index in index_set:
        index_set.remove(init_index)
    left_set = []
    right_set = []
    for index in index_set:
        if rel_mat[init_index][index] > 0:
            left_set.append(index)
        else:
            right_set.append(index)
    left_set = _quick_sort(rel_mat, index_set=left_set)
    right_set = _quick_sort(rel_mat, index_set=right_set)
    ret_index = left_set + [init_index] + right_set
    return ret_index


def quick_sort_to_index(relative_pos, init_index=None, mask_tgt=None):
    index_list, reindex_list = [], []
    if relative_pos.dim() < 3:
        relative_pos = relative_pos.unsqueeze(dim=0)
    batch_size, seq_len, _ = relative_pos.size()
    index_length_list = [seq_len] * batch_size

    if mask_tgt is not None:
        index_length_list = mask_tgt.sum(dim=-1).long().cpu().data.numpy()
    if init_index is not None:
        init_index = init_index.long().cpu().data.numpy()
    else:
        init_index = [None] * batch_size

    for b_idx in range(batch_size):
        _rel_mat = relative_pos[b_idx].cpu().data.numpy()
        _init_index = init_index[b_idx]
        _index_set = list(range(index_length_list[b_idx]))
        _reindex = np.array(_quick_sort(rel_mat=_rel_mat, init_index=_init_index, index_set=_index_set))
        _index = np.argsort(_reindex)
        index_list.append(_index)
        reindex_list.append(_reindex)

    if mask_tgt is not None:
        return index_list, reindex_list
    else:
        index = torch.from_numpy(np.array(index_list))
        reindex = torch.from_numpy(np.array(reindex_list))
        if relative_pos.is_cuda:
            index = index.cuda(relative_pos.get_device())
            reindex = reindex.cuda(relative_pos.get_device())
        return index, reindex


def relative_to_absolute(relative_pos, init_index=None, mask_tgt=None, accept_threshold=0.5, consistent_num=-1,
                         convert_type='quick'):
    """
    Examples:
        rel_pos = torch.Tensor([[[0, 2, 1, -1], [-2, 0, -1, -3], [-1, 1, 0, -2], [1, 3, 2, 0]],
                        [[0, -1, 1, 2], [-1, 0, 2, 3], [-1, -2, 0, 1], [-2, -3, -1, 0]]])
        i_lst, r_lst = relative_to_absolute(rel_pos)
    """
    if consistent_num != -1:
        relative_pos = relative_pos.clamp(-1, 1)
        relative_pos = consistent_sort_to_index(relative_pos, accept_threshold, consistent_num, index_type='prep')
    if convert_type == 'quick':
        return quick_sort_to_index(relative_pos, init_index, mask_tgt)
    else:
        index = straight_to_index(relative_pos)
        reindex = index.long().sort(dim=-1)[1]
        return index, reindex


class Position(object):
    def __init__(self, rel_pos=None, abs_pos=None, cur_max_len=1, pos_consistent_num=3, pos_index_type='default'):
        self.rel_pos = rel_pos
        self.cur_max_len = cur_max_len
        self.abs_pos = abs_pos
        self.pos_consistent_num = pos_consistent_num
        self.pos_index_type = pos_index_type

    def relative_pos(self, max_rel_len=-1, in_place=True):
        if self.rel_pos is not None and self.cur_max_len >= max_rel_len:
            return self.rel_pos
        elif self.absolute_pos is not None:
            rel_pos = (self.abs_pos.unsqueeze(dim=-1) - self.abs_pos.unsqueeze(-2))
            if max_rel_len > 0:
                # TODO: NEED CHECK
                rel_pos = rel_pos.clamp(-max_rel_len, max_rel_len) + max_rel_len  # [0,]
            if in_place:
                self.rel_pos = rel_pos
            return rel_pos
        else:
            raise RuntimeError('no pos has set')

    @property
    def absolute_pos(self):
        if self.abs_pos is not None:
            return self.abs_pos
        elif self.rel_pos is not None:
            self.abs_pos, _ = relative_to_absolute(
                relative_pos=self.rel_pos,
                consistent_num=self.pos_consistent_num,
                convert_type=self.pos_index_type
            )
            return self.abs_pos
        else:
            raise RuntimeError('no pos has set')

    @property
    def re_index(self):
        return self.absolute_pos.sort(dim=-1)[1]

    def relative_oracle(self, pos_class_num):
        relative_pos = self.relative_pos(max_rel_len=-1, in_place=False)
        max_rel_len = pos_class_num // 2
        gap = 1 - (pos_class_num % 2)
        return relative_pos.clamp(-max_rel_len, max_rel_len - gap) + max_rel_len
