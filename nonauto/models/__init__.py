import nonauto.models.basic_nat
import nonauto.models.pos_nat
from fairseq.models.transformer import (
    base_architecture,
    register_model_architecture
)


@register_model_architecture('transformer', 'transformer_iwslt16_de_en')
def transformer_iwslt16_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 278)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 507)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 2)
    args.encoder_layers = getattr(args, 'encoder_layers', 5)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 278)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 507)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 2)
    args.decoder_layers = getattr(args, 'decoder_layers', 5)
    base_architecture(args)
