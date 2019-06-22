import torch
import torch.nn as nn

from nonauto.model_utils import (
    diagonal_mask, self_spread_mask
)
from nonauto.modules.blocks import (
    FeedForwardBlock, get_sublayer_cls
)
from nonauto.modules.pointer import PointerNet
from nonauto.position_utils import Position

LOGITS_KEY = 'predict_pos_logits'
PREDICT_KEY = 'predict_pos'
MASK_KEY = 'predict_masks'


class PointerRanker(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, repeat=False):
        super().__init__()
        self.pos_dec = PointerNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            repeat=repeat
        )

    def forward(self, inputs, mask_inputs, inputs_index=None, beam_size=1, **kwargs):
        """ batch first """
        if beam_size > 1:
            ranked_state_index = self.pos_dec.forward(
                inputs=inputs,
                mask_inputs=mask_inputs,
                inputs_index=inputs_index,
                beam_size=beam_size,
            )
            outputs = None
        else:
            outputs, ranked_state_index = self.pos_dec.forward(
                inputs,
                mask_inputs=mask_inputs,
                inputs_index=inputs_index,
            )
        if inputs_index is None:
            pos_predict = ranked_state_index.long().sort(dim=-1)[1]
        else:
            pos_predict = inputs_index
        position_ret = Position(
            abs_pos=pos_predict,
        )
        return {
            LOGITS_KEY: outputs,
            PREDICT_KEY: position_ret,
            MASK_KEY: mask_inputs,
        }


class FFNRelativeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, pos_class_num, symmetry):
        super().__init__()
        self.pos_dec = nn.Sequential(
            FeedForwardBlock(
                input_dim,
                hidden_dim,
                dropout
            ),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 2 * pos_class_num, bias=True)
        )
        self.pos_class_num = pos_class_num
        self.symmetry = symmetry

    def mask_non_predict(self, mask):
        batch, seq_len = mask.size()
        non_predict_mask = 1 - mask.long()
        index = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        if mask.is_cuda:
            index = index.cuda(mask.get_device())
        non_predict_pos = (index[:, :, None] - index[:, None, :]).clamp(-self.pos_class_num, self.pos_class_num)
        non_predict_mask = self_spread_mask(non_predict_mask)
        return non_predict_mask, non_predict_pos

    def ret_position(self, logits, mask=None):
        if self.symmetry:
            logits_mirror = logits.contiguous().transpose(1, 2).flip(dims=[3, ])
            logits = logits + logits_mirror
        predict = logits.max(dim=-1)[1]
        predict = predict - self.pos_class_num + predict.ge(self.pos_class_num).long()
        diagonal_zero = diagonal_mask(predict)
        predict = predict * diagonal_zero
        if mask is not None:
            non_predict_mask, non_predict_pos = self.mask_non_predict(mask)
            predict = non_predict_mask * non_predict_pos + (1 - non_predict_mask) * predict
        return logits, predict

    def forward(self, inputs, mask_inputs, **kwargs):
        """ batch first """
        batch_size, seq_len, _ = inputs.size()
        inputs = inputs.unsqueeze(-2) - inputs.unsqueeze(-3)
        predict_logits = self.pos_dec(inputs)
        post_predict_logits, relative_pos_predict = self.ret_position(predict_logits, mask_inputs)

        position_ret = Position(
            rel_pos=relative_pos_predict,
            cur_max_len=self.pos_class_num
        )
        return {
            LOGITS_KEY: predict_logits,
            PREDICT_KEY: position_ret,
            MASK_KEY: mask_inputs,
            'post_predict_pos_logits': post_predict_logits,
        }


class PositionDecoder(nn.Module):
    def __init__(self, block_cls, *block_args):
        super().__init__()
        self.pos_dec = block_cls(*block_args)

    @classmethod
    def build_model(cls, args):
        if args.pos_dec_cls == 'reorder':
            block_cls = PointerRanker
            block_args = [
                args.decoder_embed_dim,
                args.pos_dec_hidden_dim,
                args.pos_dec_num_layers,
                args.pos_dropout,
                args.pos_repeat,
            ]
        else:
            block_cls = FFNRelativeClassifier
            block_args = [
                args.decoder_embed_dim,
                args.pos_dec_hidden_dim,
                args.pos_dropout,
                args.pos_class_num,
                not args.pos_no_symmetry
            ]
        return PositionDecoder(block_cls, *block_args)

    def forward(self, inputs, mask_inputs, **kwargs):
        return self.pos_dec(inputs, mask_inputs, **kwargs)


class PositionEncoder(nn.Module):
    def __init__(self, block_cls, num_layers, *block_args):
        super().__init__()
        self.pos_enc = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.pos_enc.append(block_cls(*block_args))

    @classmethod
    def build_model(cls, args):
        if args.pos_enc_cls == 'cond-rnn':
            block_args = [
                args.decoder_embed_dim,
                args.pos_enc_hidden_dim,
                args.pos_dropout,
                args.pos_enc_num_heads,
                args.pos_enc_inner_layers,
                not args.pos_no_enc_bidir,
                not args.pos_no_self_attn,
                not args.attn_use_future,
                not args.attn_use_self,
            ]
        else:
            block_args = [
                args.decoder_embed_dim,
                args.pos_enc_hidden_dim,
                args.pos_dropout,
                args.pos_enc_num_heads,
                not args.pos_no_init,
                not args.pos_no_self_attn,
                not args.attn_use_future,
                not args.attn_use_self,
            ]
        block_cls = get_sublayer_cls(args.pos_enc_cls)
        num_layers = args.pos_enc_num_layers
        return PositionEncoder(block_cls, num_layers, *block_args)

    def forward(self, inputs, ctxs, mask_inputs, mask_ctxs, **kwargs):
        """
        TODO: DIMENSION CHECK
        :param inputs: dec_len, batch, depth
        :param ctxs: src_len, batch, depth
        :param mask_inputs: batch, dec_len
        :param mask_ctxs: batch, src_len
        :param kwargs:
        :return: batch_size, dec_len, depth
        """
        output = inputs
        for i in range(self.num_layers):
            output = self.pos_enc[i](inputs, ctxs, mask_inputs, mask_ctxs, **kwargs)
        output = output.permute(1, 0, 2)
        return output


class PositionPredictor(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.position_encoder = encoder
        self.position_decoder = decoder

    @staticmethod
    def add_args(parser):
        # universal parameter
        parser.add_argument('--pos-dropout', type=float, metavar='D')
        parser.add_argument('--pos-class-num', type=int, metavar='N')
        parser.add_argument('--pos-no-symmetry', action='store_true')
        parser.add_argument('--pos-repeat', action='store_true')

        # encoder parameter
        parser.add_argument('--pos-enc-cls', type=str, metavar='STR', choices=['cond-rnn', 'cond-attn'])
        parser.add_argument('--pos-enc-hidden-dim', type=int, metavar='N')
        parser.add_argument('--pos-enc-num-heads', type=int, metavar='N')
        parser.add_argument('--pos-enc-inner-layers', type=int, metavar='N')
        parser.add_argument('--pos-enc-num-layers', type=int, metavar='N')
        parser.add_argument('--pos-no-self-attn', action='store_true')
        parser.add_argument('--pos-no-enc-bidir', action='store_true')
        parser.add_argument('--pos-no-init', action='store_true')

        # decoder parameter
        parser.add_argument('--pos-dec-cls', type=str, metavar='STR', choices=['reorder', 'classifier'])
        parser.add_argument('--pos-dec-hidden-dim', type=int, metavar='N')
        parser.add_argument('--pos-dec-num-layers', type=int, metavar='N')

        # position training selection
        parser.add_argument('--pos-train-with-predict', action='store_true')
        parser.add_argument('--pos-valid-with-predict', action='store_true')
        parser.add_argument('--pos-no-learning', action='store_true')

        # position oracle search
        parser.add_argument('--pos-search-type', type=int, metavar='N')
        parser.add_argument('--pos-decompose', action='store_true')
        parser.add_argument('--pos-no-search-norm', action='store_true')

    @classmethod
    def build_model(cls, args):
        position_encoder = PositionEncoder.build_model(args)
        position_decoder = PositionDecoder.build_model(args)
        return PositionPredictor(position_encoder, position_decoder)

    def encode(self, dec, encoder_out, mask_dec, mask_enc):
        encoder_out = encoder_out['encoder_out']
        return self.position_encoder(dec.transpose(1, 0), encoder_out, mask_dec, mask_enc)

    def forward(self, dec, encoder_out, mask_dec, mask_enc, inputs_index=None, beam_size=1, **unused):
        output = self.encode(dec, encoder_out, mask_dec, mask_enc)
        return self.position_decoder(output, mask_dec, inputs_index=inputs_index, beam_size=beam_size)
