import torch
import torch.nn as nn

from nonauto.model_utils import (
    sequence_mask,
    Linear,
    matmul,
    softmax,
    INF,
    apply_mask,
)


def compute_for_dec(source_masks, decoder_masks, decoder_input_how):
    if decoder_input_how == "copy":
        max_trg_len = decoder_masks.size(1)

        src_lens = source_masks.sum(-1).float() - 1  # batch_size
        trg_lens = decoder_masks.sum(-1).float() - 1  # batch_size
        steps = src_lens / trg_lens  # batch_size

        index_s = torch.arange(max_trg_len).float()  # max_trg_len
        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())

        index_s = steps[:, None] * index_s[None, :]  # batch_size X max_trg_len
        index_s = index_s.round().long()
        return index_s

    elif decoder_input_how == "wrap":
        batch_size, max_src_len = source_masks.size()
        max_trg_len = decoder_masks.size(1)

        src_lens = source_masks.sum(-1).int()  # batch_size

        index_s = torch.arange(max_trg_len)[None, :]  # max_trg_len
        index_s = index_s.repeat(batch_size, 1)  # (batch_size, max_trg_len)

        for sin in range(batch_size):
            if src_lens[sin] + 1 < max_trg_len:
                index_s[sin, src_lens[sin]:2 * src_lens[sin]] = index_s[sin, :src_lens[sin]]

        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())

        # return Variable(index_s, requires_grad=False).long()
        return index_s.long()

    elif decoder_input_how == "pad":
        batch_size, max_src_len = source_masks.size()
        max_trg_len = decoder_masks.size(1)

        src_lens = source_masks.sum(-1).int() - 1  # batch_size

        index_s = torch.arange(max_trg_len)[None, :]  # max_trg_len
        index_s = index_s.repeat(batch_size, 1)  # (batch_size, max_trg_len)

        for sin in range(batch_size):
            if src_lens[sin] + 1 < max_trg_len:
                index_s[sin, src_lens[sin] + 1:] = index_s[sin, src_lens[sin]]

        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())

        # return Variable(index_s, requires_grad=False).long()
        return index_s.long()
    elif decoder_input_how == "interpolate":
        max_src_len = source_masks.size(1)
        max_trg_len = decoder_masks.size(1)
        src_lens = source_masks.sum(-1).float()  # batchsize
        trg_lens = decoder_masks.sum(-1).float()  # batchsize
        steps = src_lens / trg_lens  # batchsize
        index_t = torch.arange(0, max_trg_len).float()  # max_trg_len
        if decoder_masks.is_cuda:
            index_t = index_t.cuda(decoder_masks.get_device())
        index_t = steps[:, None] @ index_t[None, :]  # batch x max_trg_len
        index_s = torch.arange(0, max_src_len).float()  # max_src_len
        if decoder_masks.is_cuda:
            index_s = index_s.cuda(decoder_masks.get_device())
        index_matrix = (index_s[None, None, :] - index_t[:, :, None]) ** 2  # batch x max_trg x max_src
        index_matrix = softmax(-index_matrix.float() / 0.3 - INF * (1 - source_masks[:, None, :].float()))
        # batch x max_trg x max_src
        return index_matrix


class LengthPredictorBridge(nn.Module):
    """
    Bridge the encoder and non-auto decoder: predicting the target length and
        obtaining the decoder input, based on encoder output.

    Mode:
        fixed
        predict
        reference
    use_predict_len: Use predict or not

    """
    DECODE_INPUT_KEY = 'inputs'
    DECODE_MASK_KEY = 'masks'

    def __init__(self, args, dictionary, max_offset=-1):
        super().__init__()
        self.len_opt = args.bridge_len_opt

        self.train_with_predict = args.bridge_train_with_predict  # decide which length used while evaluating.
        self.valid_with_predict = args.bridge_valid_with_predict
        self.dictionary = dictionary
        self.decoder_input_select = args.bridge_input_select
        self.decoder_input_how = args.bridge_input_how
        self.max_offset = max_offset
        self.min_len = 2
        if self.len_opt == 'predict':
            self.length_predictor = nn.Sequential(
                nn.Dropout(args.bridge_dropout),
                Linear(args.encoder_embed_dim, 2 * max_offset + 1)
            )
        else:
            self.length_predictor = None
            self.len_ratio = args.bridge_len_ratio

    @staticmethod
    def add_args(parser):
        parser.add_argument('--bridge-len-opt', type=str, metavar='STR', choices=['predict', 'reference', 'fixed'],
                            help='bridge length option')
        parser.add_argument('--bridge-len-ratio', type=float, metavar='D', help='bridge length ratio for fixed option')
        parser.add_argument('--bridge-max-offset', type=int, metavar='N', help='bridge length predictor')
        parser.add_argument('--bridge-input-select', type=str, metavar='STR', choices=['embed', 'hidden'],
                            help='bridge input: select what for copy')
        parser.add_argument('--bridge-input-how', type=str, metavar='STR',
                            choices=['copy', 'pad', 'wrap', 'interpolate'],
                            help='bridge input: how to copy from source')
        parser.add_argument('--bridge-dropout', type=float, metavar='D', help='bridge dropout')
        parser.add_argument('--bridge-train-with-predict', action='store_true',
                            help='if set, use the predict decoder length instead of reference length while training')
        parser.add_argument('--bridge-valid-with-predict', action='store_true',
                            help='if set, use the predict decoder length instead of reference length while inference')

    def forward(self, encoder_out, prev_output_tokens=None):
        inputs_dict = self._prepare_inputs_dict(encoder_out, prev_output_tokens)
        src_masks, src_len = inputs_dict['source_masks'], inputs_dict['source_len']
        tgt_masks, tgt_len = inputs_dict['target_masks'], inputs_dict['target_len']

        encoding = encoder_out['encoder_history']
        predict_len = self._predict_len_info(encoding, inputs_dict)

        dec_len = self._select_len_info(predict_len, tgt_len)
        dec_masks = sequence_mask(sequence_len=dec_len)
        dec_inputs, dec_masks = self._transfer_to_dec(encoding, src_masks, dec_masks)

        inputs_dict['inputs'] = dec_inputs
        inputs_dict['masks'] = dec_masks
        inputs_dict['decoder_masks'] = dec_masks
        return inputs_dict

    def _transfer_to_dec(self, encoding, source_masks, decoder_masks):
        decoder_input_how = self.decoder_input_how
        decoder_input_select = self.decoder_input_select

        enc = encoding[0] if decoder_input_select == 'embed' else encoding[-1]
        d_model = enc.size()[-1]

        if self.decoder_input_how in ["copy", "pad", "wrap"]:
            attention = compute_for_dec(source_masks, decoder_masks, self.decoder_input_how)
            attention = apply_mask(attention, decoder_masks, p=1)  # p doesn't matter cos masked out
            attention = attention[:, :, None].expand(*attention.size(), d_model)
            # decoder_inputs = torch.gather(enc, dim=1, index=attention)
            decoder_inputs = enc.gather(1, attention)
        elif decoder_input_how == "interpolate":
            attention = compute_for_dec(source_masks, decoder_masks, self.decoder_input_how)
            decoder_inputs = matmul(attention, enc)  # batch x max_trg x size
        else:
            raise NotImplementedError

        return decoder_inputs, decoder_masks

    def _predict_len_info(self, encoding, inputs_dict):
        src_masks, src_len = inputs_dict['source_masks'], inputs_dict['source_len']
        if self.length_predictor is not None:
            min_len = self.min_len
            in_features = sum(encoding).mean(1)
            offset_logits = self.length_predictor(in_features)
            predict_offset = offset_logits.max(-1)[1] - self.max_offset
            predict_len = src_len + predict_offset
            predict_len = predict_len.lt(min_len).long() * min_len + predict_len.gt(min_len).long() * predict_len
            inputs_dict['predict_offset'] = predict_offset
            inputs_dict['predict_offset_logits'] = offset_logits
        elif self.len_opt == 'fixed':
            # fixed mapping the target length with ratio to source length
            predict_len = (src_len.float() * self.len_ratio).long()
        else:
            predict_len = None
        return predict_len

    def _select_len_info(self, predict_length, oracle_length=None):
        """ modularize the length return """
        if oracle_length is None:
            return predict_length
        if predict_length is None:
            return oracle_length
        if (self.train_with_predict and self.training) or (self.valid_with_predict and not self.training):
            return predict_length
        return oracle_length

    def _prepare_inputs_dict(self, encoder_out=None, prev_output_tokens=None):
        """ modularize the preprocess for length information and return dictionary"""
        inputs_dict = {}
        encoding = encoder_out['encoder_history']
        source_masks = encoder_out['encoder_padding_mask']
        if source_masks is not None:
            source_masks = 1 - source_masks
        else:
            batch_size, src_max_len, _ = encoding[0].size()
            source_masks = torch.ones(batch_size, src_max_len)
            if encoding[0].is_cuda:
                source_masks = source_masks.cuda(encoding[0].get_device())

        source_len = source_masks.sum(-1).long()
        target_pad, target_masks, target_len, reference_offset = None, None, None, None
        if prev_output_tokens is not None:
            target_pad = prev_output_tokens.eq(self.dictionary.pad())
            target_masks = 1 - target_pad.long()
            target_len = target_masks.sum(-1)
            reference_offset = (target_len - source_len + self.max_offset).clamp(0, 2 * self.max_offset)

        inputs_dict['target_padding_mask'] = target_pad
        inputs_dict['target_masks'] = target_masks
        inputs_dict['source_masks'] = source_masks
        inputs_dict['source_len'] = source_len
        inputs_dict['prev_output_tokens'] = prev_output_tokens
        inputs_dict['target_len'] = target_len
        inputs_dict['reference_offset'] = reference_offset
        return inputs_dict
