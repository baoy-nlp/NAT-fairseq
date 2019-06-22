import torch
import torch.nn as nn

from nonauto.model_utils import matmul

# import torch.nn.Embedding as Embedding

INF = 1e10
TINY = 1e-9


def process_relative_position(distance_mat, direction, max_rel_len, shift_to_zero=True):
    if max_rel_len is None:
        distance_mat_clipped = distance_mat
    else:
        distance_mat_clipped = distance_mat.clamp(-max_rel_len, max_rel_len)
    if direction:
        if shift_to_zero and max_rel_len is not None:
            final_mat = distance_mat_clipped + max_rel_len
        else:
            final_mat = distance_mat_clipped
    else:
        final_mat = distance_mat_clipped.abs()
    return final_mat


def get_relative_position_matrix(length, max_relative_position, direction, offset=True):
    """ Generate matrix of relative positions between inputs ([..., length])."""
    index_vec = torch.arange(length).long()
    distance_mat = index_vec[:, None] - index_vec[None, :]
    return process_relative_position(
        distance_mat,
        direction,
        max_relative_position,
        offset
    )


class RelativePositionEmbeddings(nn.Module):
    def __init__(self, max_relative_position, embedding_dim, dropout=0.0, direction=True, **params):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.embedding_dim = embedding_dim
        self.direction = direction
        if self.direction:
            vocab_size = max_relative_position * 2 + 1
            self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=None)
        else:
            vocab_size = max_relative_position + 1
            self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=None)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        """ Generate tensor of size [length, length, depth] """
        if type(inputs) == int:
            rel_pos_mat = get_relative_position_matrix(inputs, self.max_relative_position, self.direction)
        elif type(inputs) == torch.Tensor:
            if isinstance(inputs, torch.FloatTensor) or isinstance(inputs, torch.cuda.FloatTensor):
                rel_embed = matmul(inputs, self.embeddings.weight)
                # _,embed_length * [embed_lenth, embed_dims] -> _ , embed_dims
                rel_embed = self.dropout(rel_embed)
                return rel_embed
            else:
                rel_pos_mat = process_relative_position(inputs, self.direction, self.max_relative_position)
        else:
            raise RuntimeError('length type is error with {}'.format(type(inputs)))
        if torch.cuda.is_available():
            rel_pos_mat = rel_pos_mat.cuda(self.embeddings.weight.get_device())
        rel_embed = self.embeddings(rel_pos_mat)
        rel_embed = self.dropout(rel_embed)
        return rel_embed
