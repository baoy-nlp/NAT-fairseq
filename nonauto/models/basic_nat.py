import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture
)
from nonauto.model_utils import (
    Embedding,
    OutLinear,
)
from nonauto.modules.layers import (
    NATEncoder,
    BasicNATDecoder,
)
from nonauto.modules.length import LengthPredictorBridge


@register_model('basic_nat')
class BasicNAT(FairseqEncoderDecoderModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn', choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--decoder-final-norm', default=False, action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--residual-type', type=str, metavar='STR',
                            help='residual type in ffn layer')

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # +++++++++++++++++++++++++++++++++++++++++++ APPEND for BASIC NAT +++++++++++++++++++++++++++++++++++++++++++#
        LengthPredictorBridge.add_args(parser)

        parser.add_argument('--decoder-pos-attn', action='store_true',
                            help='if set, use the position self attention mentioned in Gu et al.')
        parser.add_argument('--attn-use-future', action='store_true')
        parser.add_argument('--attn-use-self', action='store_true')
        parser.add_argument('--use-enc-last', action='store_true')
        parser.add_argument('--out-norm', action='store_true')

    @classmethod
    def build_embed(cls, args, src_dict, tgt_dict):
        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)  # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError('--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            # encoder_embed = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            encoder_embed = OutLinear(args.encoder_embed_dim, len(src_dict), bias=False, out_norm=args.out_norm)
            decoder_embed = encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            # decoder_embed = build_embedding(tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)
            decoder_embed = OutLinear(args.decoder_embed_dim, len(tgt_dict), bias=False, out_norm=args.out_norm)

        return encoder_embed, decoder_embed

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        # base_architecture(args)
        basic_nat_small(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        encoder_embed, decoder_embed = cls.build_embed(args, src_dict, tgt_dict)

        encoder = NATEncoder(args, src_dict, encoder_embed)
        decoder = BasicNATDecoder(args, tgt_dict, decoder_embed)
        return BasicNAT(encoder=encoder, decoder=decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return features

    def get_normalized_probs(self, net_output, log_probs, sample=None, adaptive_softmax=True):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError


def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.decoder_pos_attn = getattr(args, 'decoder_pos_attn', True)
    args.attn_use_future = getattr(args, 'attn_use_future', True)
    args.attn_use_self = getattr(args, 'attn_use_self', False)
    args.use_enc_last = getattr(args, 'use_enc_last', False)
    args.out_norm = getattr(args, 'out_norm', False)

    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.dropout = getattr(args, 'dropout', 0.1)
    # bridge parameter
    args.bridge_train_with_predict = getattr(args, 'bridge_train_with_predict', False)
    args.bridge_valid_with_predict = getattr(args, 'bridge_valid_with_predict', False)
    args.bridge_len_opt = getattr(args, 'bridge_len_opt', 'predict')
    args.bridge_len_ratio = getattr(args, 'bridge_len_ratio', 1.0)
    args.bridge_max_offset = getattr(args, 'bridge_max_offset', 20)
    args.bridge_input_select = getattr(args, 'bridge_input_select', 'embed')
    args.bridge_input_how = getattr(args, 'bridge_input_how', 'copy')
    args.bridge_dropout = getattr(args, 'bridge_dropout', 0.1)
    args.residual_type = getattr(args, 'residual_type', 'residual')
    args.left_pad_source = getattr(args, 'left_pad_source', False)


@register_model_architecture('basic_nat', 'small')
def basic_nat_small(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 278)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 507)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 2)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 2)
    args.encoder_layers = getattr(args, 'encoder_layers', 5)
    args.decoder_layers = getattr(args, 'decoder_layers', 5)
    args.dropout = getattr(args, 'dropout', 0.079)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    base_architecture(args)


@register_model_architecture('basic_nat', 'big')
def basic_nat_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.dropout = getattr(args, 'dropout', 0.1)
    basic_nat_small(args)


@register_model_architecture('basic_nat', 'bigger')
def basic_nat_bigger(args):
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    basic_nat_big(args)


@register_model_architecture('basic_nat', 'biggest')
def basic_nat_biggest(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    basic_nat_bigger(args)
