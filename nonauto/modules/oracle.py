import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nonauto.model_utils import (
    Linear, previous_max_connect_search, positional_encodings_like
)
from nonauto.modules.blocks import (
    FeedForwardBlock
)
from nonauto.position_utils import Position


class PositionHeuristicSearcher(nn.Module):
    """
    Position Aware: position type
        0. no position info be considered while search
        1. just add position embedding to candidate state
        2. predict expectation position embedding for each query states
    """

    def __init__(self, embed_dim, dropout, search_type=0, decompose=False, normalize=True, out=None):
        super().__init__()
        if decompose and search_type != 0:
            self.decomposer = nn.Sequential(
                FeedForwardBlock(embed_dim, 4 * embed_dim, dropout),
                nn.Dropout(dropout),
                Linear(embed_dim, 2 * embed_dim)
            )
        else:
            self.decomposer = None
        self.normalize = normalize
        self.position_type = search_type
        self.embed_dim = embed_dim
        self.out = out
        self.embed_scale = math.sqrt(embed_dim)

    def forward(self, dec, tgt, mask_dec, mask_tgt, use_embed=True):
        if not use_embed:
            tgt = F.embedding(tgt, self.out.weight * self.embed_scale)
        query_content, query_pos = self.wrap_decompose(dec)
        query_pos, candidate_pos = self.wrap_position(query_pos, tgt)
        inputs = query_content + query_pos
        targets = tgt + candidate_pos

        if self.normalize:
            # logits = F.cosine_similarity(inputs[:, None, :, :], targets[:, :, None, :], dim=-1)
            inputs = inputs / inputs.norm(dim=-1)[:, :, None]
            targets = targets / targets.norm(dim=-1)[:, :, None]
            logits = torch.matmul(inputs, targets.contiguous().transpose(1, 2))
        else:
            logits = torch.matmul(inputs, targets.contiguous().transpose(1, 2))

        match_ret = previous_max_connect_search(
            # match_ret = non_auto_max_connect_search(
            logits,
            mask_dec,
            mask_tgt,
            share_prob=True
        )

        return {
            'os_dec': dec,
            'os_dec_mask': mask_dec,
            'os_tgt': tgt,
            'os_tgt_mask': mask_tgt,
            'os_logits': logits,
            'os_iter': match_ret['iter'],
            'os_pos': Position(abs_pos=match_ret['index']),
            'os_one_hot': match_ret['one_hot']
        }

    def wrap_decompose(self, query):
        """ decompose the origin output for content and position """
        if self.decomposer is None:
            query_pos = query
        else:
            decomposes = self.decomposer.forward(query)
            query, query_pos = decomposes.chunk(2, dim=-1)

        return query, query_pos

    def wrap_position(self, pos_query, candidate):
        """
             return the position for matching
        :param pos_query: batch, query_len, state_dim
        :param candidate: batch, target_len, state_dim
        :return:
        """
        # batch_size, target_len, state_dim
        if self.position_type == 0:
            return 0, 0
        position_candidate = positional_encodings_like(candidate)
        if self.position_type == 1:
            return 0, position_candidate
        if self.normalize:
            # batch, query_len, target_len
            position_logits = F.cosine_similarity(pos_query[:, None, :, :], position_candidate[:, :, None, :], dim=-1)
        else:
            # batch, query_len, target_len
            position_logits = torch.matmul(pos_query, position_candidate.contiguous().transpose(1, 2))

        # batch_size, query_len, state_dim
        position_query = torch.matmul(position_logits, position_candidate)
        return position_query, position_candidate
