import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from fairseq import utils
from nonauto.model_utils import OutLinear
from nonauto.model_utils import matmul


class FFNAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=False):
        """
        Initiate Attention

        :param int input_dim: Input's dimension
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(FFNAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        # self.out_V = Parameter(torch.Tensor(nhid).float(), requires_grad=True)
        self.out_proj = nn.Linear(hidden_dim, 1, bias=bias)
        self._inf = Parameter(torch.Tensor([-1e18]), requires_grad=False)
        self.inf = None

        # Initialize vector V
        nn.init.uniform_(self.out_proj.weight, -1, 1)

    def forward(self, query, key, mask=None):
        """
        feed-forward attention: score = W * [U * `query` + V * `key`], `query` and `key` 's dimension both is `batch,
         seq, input`.
        """
        query = self.q_proj(query).unsqueeze(2).expand(-1, -1, key.size(1))  # (batch, hidden, seq_len)
        key = key.permute(0, 2, 1)  # (batch, hidden, seq_len)
        key = self.k_proj(key)  # (batch, hidden, seq_len)

        # out_mat = self.out_V.unsqueeze(0).expand(key.size(0), -1).unsqueeze(1)  # (batch, 1, hidden)
        # attn_weight = torch.bmm(out_mat, (query + key).tanh()).squeeze(1)  # (batch, seq_len)
        attn_weight = self.out_proj((query + key).permute(0, 2, 1)).squeeze(-1)  # (batch, seq_len)
        if mask is not None and len(attn_weight[mask]) > 0:
            attn_weight[mask] = self.inf[mask]

        attn_prob = attn_weight.softmax(dim=-1)
        attn = torch.bmm(key, attn_prob.unsqueeze(2)).squeeze(2)
        return attn, attn_weight

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True, add_bias_kv=False,
                 add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        # if self.qkv_same_dim:
        #     self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        # else:
        #     self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
        #     self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
        #     self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        #
        # if bias:
        #     self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        # else:
        #     self.register_parameter('in_proj_bias', None)
        # if self.qkv_same_dim:
        #     self.w_in = OutLinear(embed_dim, 3 * embed_dim, bias=bias)
        # else:
        self.wq = OutLinear(embed_dim, embed_dim, bias=bias)
        self.wk = OutLinear(self.kdim, embed_dim, bias=bias)
        self.wv = OutLinear(self.vdim, embed_dim, bias=bias)
        self.out_proj = OutLinear(embed_dim, embed_dim, bias=bias)
        # self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # if add_bias_kv:
        #     self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
        #     self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        # else:
        #     self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        # if self.qkv_same_dim:
        #     nn.init.xavier_uniform_(self.in_proj_weight)
        # else:
        #     nn.init.xavier_uniform_(self.k_proj_weight)
        #     nn.init.xavier_uniform_(self.v_proj_weight)
        #     nn.init.xavier_uniform_(self.q_proj_weight)
        #
        # nn.init.xavier_uniform_(self.out_proj.weight)
        # if self.in_proj_bias is not None:
        #     nn.init.constant_(self.in_proj_bias, 0.)
        #     nn.init.constant_(self.out_proj.bias, 0.)
        # if self.bias_k is not None:
        #     nn.init.xavier_normal_(self.bias_k)
        # if self.bias_v is not None:
        #     nn.init.xavier_normal_(self.bias_v)
        raise NotImplementedError

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True,
                static_kv=False, attn_mask=None):

        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # if qkv_same:
        #     # self-attention
        #     q, k, v = self.in_proj_qkv(query)
        # elif kv_same:
        #     # encoder-decoder attention
        #     q = self.in_proj_q(query)
        #     if key is None:
        #         assert value is None
        #         k = v = None
        #     else:
        #         k = self.in_proj_k(key)
        #         v = self.in_proj_v(key)
        # else:
        # q = self.in_proj_q(query)
        # k = self.in_proj_k(key)
        # v = self.in_proj_v(value)
        q = self.in_proj_q(query)
        k = self.in_proj_k(key)
        v = self.in_proj_v(value)

        q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        # if self.qkv_same_dim:
        #     return self._in_proj(query, end=self.embed_dim)
        # else:
        #     bias = self.in_proj_bias
        #     if bias is not None:
        #         bias = bias[:self.embed_dim]
        #     return F.linear(query, self.q_proj_weight, bias)
        return self.wq(query)

    def in_proj_k(self, key):
        # if self.qkv_same_dim:
        #     return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        # else:
        #     weight = self.k_proj_weight
        #     bias = self.in_proj_bias
        #     if bias is not None:
        #         bias = bias[self.embed_dim:2 * self.embed_dim]
        #     return F.linear(key, weight, bias)
        return self.wk(key)

    def in_proj_v(self, value):
        # if self.qkv_same_dim:
        #     return self._in_proj(value, start=2 * self.embed_dim)
        # else:
        #     weight = self.v_proj_weight
        #     bias = self.in_proj_bias
        #     if bias is not None:
        #         bias = bias[2 * self.embed_dim:]
        #     return F.linear(value, weight, bias)
        return self.wv(value)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self.get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self.set_input_buffer(incremental_state, input_buffer)

    def get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )


class ShawAttention(MultiheadAttention):
    """
    Used in self-attn
    """

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True,
                static_kv=False, attn_mask=None, pos_key=None, pos_val=None):
        """
            query,key,value: sequence_len, batch_size, depth
            attn_mask: including `-INF`，
            key_padding_mask: 1 if you do not attend to.
        """

        def shaw_rel_attention(query_, key_):
            """

            Args:
                query_: batch_size, heads, length, depth
                key_: batch_size, heads, length, depth
            Returns:
                weight: batch_size, heads, length,length

            """
            bsize, heads, length, depth = query_.size()

            q_dot_k = matmul(query_, key_.contiguous().transpose(-1, -2))  # batch, heads, length, length

            query_for_pos = query_.contiguous().permute(2, 0, 1, 3).view(length, bsize * heads, depth)
            pos_for_att = pos_key.contiguous().transpose(-2, -1)  # length, depth, length

            q_dot_p = matmul(query_for_pos, pos_for_att)  # length, batch*heads, length
            q_dot_p = q_dot_p.contiguous().permute(1, 0, 2).view(bsize, heads, length, length)

            return q_dot_k + q_dot_p

        def shaw_rel_combine(weight, value_):
            """
            Args:
                weight: batch_size, heads, length, length
                value_: batch_size, heads, length, depth
            Returns:
                attn: batch_size,heads,length,depth

            """
            bsize, heads, length, depth = value_.size()
            w_dot_v = matmul(weight, value_)  # batch, head, length, depth

            w_for_comb = weight.contiguous().permute(2, 0, 1, 3).view(length, bsize * heads, length)
            w_dot_p = matmul(w_for_comb, pos_val)  # length,batch*heads, depth
            w_dot_p = w_dot_p.contiguous().permute(1, 0, 2).view(bsize, heads, length, depth)

            return w_dot_v + w_dot_p

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        # if qkv_same:
        #     # self-attention
        #     q, k, v = self.in_proj_qkv(query)
        # elif kv_same:
        #     # encoder-decoder attention
        #     q = self.in_proj_q(query)
        #     if key is None:
        #         assert value is None
        #         k = v = None
        #     else:
        #         k = self.in_proj_k(key)
        #         v = self.in_proj_v(key)
        # else:
        #     q = self.in_proj_q(query)
        #     k = self.in_proj_k(key)
        #     v = self.in_proj_v(value)
        q = self.in_proj_q(query)
        k = self.in_proj_k(key)
        v = self.in_proj_v(value)

        q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        # attn_weights = torch.bmm(q, k.transpose(1, 2)) for absolute attention
        attn_weights = shaw_rel_attention(
            q.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
            k.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
        ).contiguous().view(bsz * self.num_heads, tgt_len, src_len)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn = torch.bmm(attn_weights, v) for absolute attention
        attn = shaw_rel_combine(
            weight=attn_weights.contiguous().view(bsz, self.num_heads, tgt_len, src_len),
            value_=v.contiguous().view(bsz, self.num_heads, -1, self.head_dim),
        ).contiguous().view(bsz * self.num_heads, -1, self.head_dim)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights


class BatchAttention(MultiheadAttention):
    pass


def get_attention_cls(cls_type_name):
    query_dict = {
        'multi_attn': MultiheadAttention,
        'shaw_attn': ShawAttention,
        'ffn_attn': FFNAttention
    }
    if cls_type_name in query_dict:
        return query_dict[cls_type_name]
    else:
        raise RuntimeError("{} is not be implements".format(cls_type_name))
