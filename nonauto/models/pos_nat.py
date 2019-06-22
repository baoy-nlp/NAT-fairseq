from fairseq.models import (
    register_model,
    register_model_architecture
)
from nonauto.models.basic_nat import (
    BasicNAT,
    base_architecture,
)
from nonauto.modules.layers import NATEncoder
from nonauto.modules.layers import PosNATDecoder
from nonauto.modules.position import PositionPredictor


@register_model('pos_nat')
class PosNAT(BasicNAT):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        BasicNAT.add_args(parser)  # including: encoder,decoder,bridge
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # +++++++++++++++++++++++++++++++++++++++++++ APPEND for Pos NAT +++++++++++++++++++++++++++++++++++++++++++#
        PositionPredictor.add_args(parser)
        parser.add_argument('--max-rel-len', type=int, metavar='N')
        parser.add_argument('--relative-attn-type', type=str, metavar='STR', choices=['shaw', 'batch'])
        parser.add_argument('--share-rel-embed', action='store_true')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        # base_architecture(args)
        pos_base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        encoder_embed, decoder_embed = cls.build_embed(args, src_dict, tgt_dict)

        encoder = NATEncoder(args, src_dict, encoder_embed)
        decoder = PosNATDecoder(args, tgt_dict, decoder_embed)
        return PosNAT(encoder=encoder, decoder=decoder)


def pos_base_architecture(args):
    base_architecture(args)

    # position-decoder args
    args.max_rel_len = getattr(args, 'max_rel_len', 4)
    args.share_rel_embed = getattr(args, 'share_rel_embed', False)
    args.relative_attn_type = getattr(args, 'relative_attn_type', 'shaw')

    # position-predictor args
    args.pos_dropout = getattr(args, 'pos_dropout', args.dropout)
    args.pos_class_num = getattr(args, 'pos_class_num', args.max_rel_len)
    args.pos_no_symmetry = getattr(args, 'pos_no_symmetry', False)
    args.pos_repeat = getattr(args, 'pos_repeat', False)
    args.pos_enc_cls = getattr(args, 'pos_enc_cls', 'cond-rnn')
    args.pos_enc_hidden_dim = getattr(args, 'pos_enc_hidden_dim', args.decoder_ffn_embed_dim)
    args.pos_enc_num_heads = getattr(args, 'pos_enc_num_heads', args.decoder_attention_heads)
    args.pos_enc_inner_layers = getattr(args, 'pos_enc_inner_layers', 2)
    args.pos_enc_num_layers = getattr(args, 'pos_enc_num_layers', 2)
    args.pos_no_self_attn = getattr(args, 'pos_no_self_attn', False)
    args.pos_no_enc_bidir = getattr(args, 'pos_no_enc_bidir', False)
    args.pos_no_init = getattr(args, 'pos_no_init', False)
    args.pos_dec_cls = getattr(args, 'pos_dec_cls', 'reorder')
    args.pos_dec_hidden_dim = getattr(args, 'pos_dec_hidden_dim', args.pos_enc_hidden_dim)
    args.pos_dec_num_layers = getattr(args, 'pos_dec_num_layers', 2)
    args.pos_train_with_predict = getattr(args, 'pos_train_with_predict', False)
    args.pos_valid_with_predict = getattr(args, 'pos_valid_with_predict', False)
    args.pos_no_learning = getattr(args, 'pos_no_learning', False)
    args.pos_search_type = getattr(args, 'pos_search_type', 0)
    args.pos_decompose = getattr(args, 'pos_decompose', False)
    args.pos_no_search_norm = getattr(args, 'pos_no_search_norm', False)


@register_model_architecture('pos_nat', 'pos_nat_small')
def pos_nat_small(args):
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
    # relative position module args
    pos_base_architecture(args)


@register_model_architecture('pos_nat', 'pos_nat_prev_shadow')
def pos_nat_prev_shadow(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 400)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 800)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.dropout = getattr(args, 'dropout', 0.1)
    pos_nat_small(args)


@register_model_architecture('pos_nat', 'pos_nat_prev_iwslt16')
def pos_nat_prev_iwslt16(args):
    args.bridge_len_opt = getattr(args, 'bridge_len_opt', 'reference')
    args.bridge_input_select = getattr(args, 'bridge_input_select', 'hidden')
    args.bridge_input_how = getattr(args, 'bridge_input_how', 'copy')
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)  # TODO: NEED RESET DICTION
    args.share_rel_embed = getattr(args, 'share_rel_embed', True)
    args.residual_type = getattr(args, 'residual_type', 'highway')
    args.decoder_pos_attn = False
    pos_nat_prev_shadow(args)
