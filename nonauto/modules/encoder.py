import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
)
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from nonauto.model_utils import positional_encodings_like
from nonauto.modules.blocks import (
    SelfAttentionBlock,
    FeedForwardBlock
)


class NATEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        embed_dim = args.encoder_embed_dim
        normalize_before = args.encoder_normalize_before
        head_nums = args.encoder_attention_heads
        dropout = args.dropout

        self.self_attn_block = SelfAttentionBlock(
            input_dim=embed_dim,
            num_heads=head_nums,
            dropout=args.attention_dropout,
            normalize_before=normalize_before,
        )

        activation_dropout = getattr(args, 'activation_dropout', 0)
        if activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout = getattr(args, 'relu_dropout', 0)

        self.ffn_block = FeedForwardBlock(
            input_dim=embed_dim,
            hidden_dim=args.encoder_ffn_embed_dim,
            dropout=dropout,
            activation_fn=getattr(args, 'activation_fn', 'relu'),
            activation_dropout=activation_dropout,
            normalize_before=normalize_before,
            residual_type=args.residual_type
        )

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        x = self.self_attn_block(
            x=x,
            padding_mask=encoder_padding_mask
        )
        x = self.ffn_block(x)
        return x

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class NATEncoder(FairseqEncoder):
    """
    Non-autoregressive Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`NATEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.share_embed = args.share_all_embeddings
        embed_dim = args.encoder_embed_dim
        self.padding_idx = dictionary.pad()
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            NATEncoderLayer(args)
            for _ in range(args.encoder_layers)
        ])
        self.dropout = nn.Dropout(args.dropout)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        if self.share_embed:
            x = F.embedding(src_tokens, self.embed_tokens.weight * self.embed_scale)
        else:
            x = self.embed_tokens(src_tokens) * self.embed_scale
        # x = self.embed_scale * self.embed_tokens(src_tokens)
        # if self.embed_positions is not None:
        #     x += self.embed_positions(src_tokens)
        x += positional_encodings_like(x)
        encoder_history = [x]
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dropout(x)
        # B x T x C -> T x B x C

        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            encoder_history.append(x.transpose(0, 1))

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_history': encoder_history,  # List<B X T X C>
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, f"{name}.layers.{i}")

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict
