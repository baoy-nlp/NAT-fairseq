import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqDecoder,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
)
from nonauto.model_utils import (
    positional_encodings_like,
    Linear,
    OutLinear,
    rearrange_tensor_with_index
)
from nonauto.modules.attention import (
    MultiheadAttention
)
from nonauto.modules.blocks import (
    SelfAttentionBlock,
    InterAttentionBlock,
    FeedForwardBlock
)
from nonauto.modules.embed import RelativePositionEmbeddings
from nonauto.modules.length import LengthPredictorBridge
from nonauto.modules.oracle import PositionHeuristicSearcher
from nonauto.modules.position import PositionPredictor


class BasicNATDecoderLayer(nn.Module):
    def __init__(self, args, pos_self_attn=False, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if pos_self_attn:
            self.pos_self_attn_block = SelfAttentionBlock(
                self.embed_dim,
                args.decoder_attention_heads,
                dropout=args.attention_dropout,
                export=export
            )
        else:
            # self.pos_self_attn = None
            # self.pos_self_attn_layer_norm = None
            self.pos_self_attn_block = None

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
            incremental_state=None,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn.set_input_buffer(incremental_state, saved_state)
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        # if self.pos_self_attn is not None:
        #     residual = x
        #     pos_encoding, weights = positional_encodings_like(x), None
        #     x, attn = self.pos_self_attn(
        #         query=pos_encoding,
        #         key=pos_encoding,
        #         value=x,
        #         key_padding_mask=self_attn_padding_mask,
        #         need_weights=False,
        #         attn_mask=self_attn_mask,
        #     )
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        #     x = residual + x
        #     x = self.maybe_layer_norm(self.pos_self_attn_layer_norm, x, after=True)
        if self.pos_self_attn_block is not None:
            pos_encoding, weights = positional_encodings_like(x), None
            x = self.pos_self_attn_block(
                x=pos_encoding,
                value=x,
                padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )

        if self.encoder_attn is not None:
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn.set_input_buffer(incremental_state, saved_state)

            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn.get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class ShawDecoderLayer(nn.Module):
    def __init__(self, args, rel_key=None, rel_val=None, pos_self_attn=False, no_encoder_attn=False, add_bias_kv=False,
                 add_zero_attn=False):
        super().__init__()
        embed_dim = args.decoder_embed_dim
        head_nums = args.decoder_attention_heads
        dropout = args.dropout
        normalize_before = args.decoder_normalize_before
        export = getattr(args, 'char_inputs', False)

        self.attn_type = 'shaw_attn'
        self.rel_key_embed = rel_key
        self.rel_val_embed = rel_val
        self.self_attn_block = SelfAttentionBlock(
            input_dim=embed_dim,
            num_heads=head_nums,
            dropout=args.attention_dropout,
            normalize_before=normalize_before,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            attn_type=self.attn_type,
            export=export
        )

        self.pos_self_attn_block = None
        if pos_self_attn:
            self.pos_self_attn_block = SelfAttentionBlock(
                input_dim=embed_dim,
                num_heads=head_nums,
                dropout=args.attention_dropout,
                normalize_before=normalize_before,
                export=export
            )

        self.encoder_attn_block = None
        if not no_encoder_attn:
            self.encoder_attn_block = InterAttentionBlock(
                input_dim=embed_dim,
                num_heads=head_nums,
                dropout=args.attention_dropout,
                normalize_before=normalize_before,
                export=export
            )

        activation_dropout = getattr(args, 'activation_dropout', 0)
        if activation_dropout == 0:
            activation_dropout = getattr(args, 'relu_dropout', 0)

        self.ffn_block = FeedForwardBlock(
            input_dim=embed_dim,
            hidden_dim=args.decoder_ffn_embed_dim,
            dropout=dropout,
            activation_fn=getattr(args, 'activation_fn', 'relu'),
            activation_dropout=activation_dropout,
            normalize_before=normalize_before,
            export=export,
            residual_type=args.residual_type
        )

        self.need_attn = True
        self.onnx_trace = False

    def forward(self, x, encoder_out=None, encoder_padding_mask=None, self_attn_mask=None, self_attn_padding_mask=None,
                pos_key=None, pos_val=None, **kwargs):
        pos_key, pos_val = self._get_pos_repr(x, pos_key, pos_val)
        x, attn = self.self_attn_block(
            x=x,
            padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
            ret_attn=True,
            need_weights=False,
            pos_key=pos_key,
            pos_val=pos_val
        )
        if self.pos_self_attn_block is not None:
            pos_encoding, weights = positional_encodings_like(x), None
            x = self.pos_self_attn_block(
                x=pos_encoding,
                value=x,
                padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )

        if self.encoder_attn_block is not None:
            x, attn = self.encoder_attn_block(
                query=x,
                key=encoder_out,
                value=encoder_out,
                padding_mask=encoder_padding_mask,
                ret_attn=True,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
        x = self.ffn_block(x)
        return x, attn

    def _get_pos_repr(self, x, pos_key, pos_val):
        if pos_key is not None and pos_val is not None:
            # means use shared embedding
            return pos_key, pos_val
        elif pos_key is not None:
            # pos_val is None, pos_key is not None: means need query embedding: pos_key is relative position
            return self.rel_key_embed(pos_key), self.rel_val_embed(pos_val)
        else:
            # pos_key is None, pos_val is None: means pos_key is None, query from x
            length = x.size(0)
            return self.rel_key_embed(length), self.rel_val_embed(length)

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True


class BasicNATDecoder(FairseqDecoder):
    """
    - casual attention without mask
    - diagonal attention mask
    - position-self attention
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.mask_future = not args.attn_use_future
        self.mask_self = not args.attn_use_self
        input_embed_dim = args.decoder_embed_dim
        embed_dim = args.decoder_embed_dim
        self.no_encoder_attn = no_encoder_attn
        self.output_embed_dim = args.decoder_output_dim
        self.max_target_positions = args.max_target_positions
        self.embed_scale = math.sqrt(embed_dim)

        self.bridge = LengthPredictorBridge(args, dictionary=dictionary, max_offset=args.bridge_max_offset)
        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None
        self.embed_tokens = None
        self.decoder_layers = None
        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
        self.adaptive_softmax = None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not args.share_decoder_input_output_embed:
            self.embed_out = OutLinear(self.output_embed_dim, len(dictionary), bias=False, out_norm=args.out_norm)
        else:
            self.embed_tokens = embed_tokens
        self.register_buffer('version', torch.Tensor([2]))

        # append for basic non-auto
        self.use_enc_last = args.use_enc_last
        self._build_inner_layers(args)

    def forward(self, prev_output_tokens, encoder_out=None, **unused):
        x, extra = self.extract_features(prev_output_tokens, encoder_out, **unused)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
                    length-predict module output
                    position-search module output
                    position-predict module output
                    decoder module output
        """
        # embed positions
        inputs_dict = self._bridging(encoder_out=encoder_out, prev_output_tokens=prev_output_tokens)
        x = inputs_dict['inputs']
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        positions = positional_encodings_like(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)
        # attn = None
        # inner_states = [x]
        # decoder layers
        # self_attn_masks = self._buffered_nat_mask(x)
        # self_attn_padding_mask = (1 - inputs_dict[LengthPredictorBridge.DECODE_MASK_KEY]).byte()
        # for layer in self.decoder_layers:
        #     x, attn = layer(
        #         x,
        #         encoder_out=encoder_out['encoder_out'] if encoder_out is not None else None,
        #         encoder_padding_mask=encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
        #         self_attn_mask=self_attn_masks,
        #         self_attn_padding_mask=self_attn_padding_mask
        #     )
        #     inner_states.append(x)
        #
        # if self.normalize:
        #     x = self.layer_norm(x)
        #
        # # T x B x C -> B x T x C
        # x = x.transpose(0, 1)
        #
        # if self.project_out_dim is not None:
        #     x = self.project_out_dim(x)
        #
        # inputs_dict['attn'] = attn
        # inputs_dict['inner_states'] = inner_states
        x, inputs_dict = self._decoding(
            x,
            encoder_out,
            inputs_dict,
        )
        return x, inputs_dict

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return self.embed_tokens(features)
            else:
                return self.embed_out(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        # if self.embed_positions is None:
        return self.max_target_positions
        # return min(self.max_target_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        for i in range(len(self.decoder_layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict

    def get_normalized_probs(self, net_output, log_probs, sample, adaptive_softmax=True):
        """Get normalized probabilities (or log probs) from a net's output."""
        if adaptive_softmax:
            if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
                if sample is not None:
                    assert 'target' in sample
                    target = sample['target']
                else:
                    target = None
                out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
                return out.exp_() if not log_probs else out

        # judge for extend the previous
        logits = net_output[0] if isinstance(net_output, list) else net_output
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    # append for non-auto-regressive decoder
    def _build_inner_layers(self, args):
        self.decoder_layers = nn.ModuleList([])
        self.decoder_layers.extend([
            BasicNATDecoderLayer(args, args.decoder_pos_attn, self.no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

    def _bridging(self, prev_output_tokens, encoder_out=None, ):
        return self.bridge.forward(encoder_out=encoder_out, prev_output_tokens=prev_output_tokens)

    def _decoding(self, x, enc_dict, inputs_dict, **unused):
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]
        attn_states = [attn]
        self_attn_masks = self._buffered_nat_mask(x)
        self_attn_padding_mask = (1 - inputs_dict[LengthPredictorBridge.DECODE_MASK_KEY]).byte()
        encoder_padding_mask = enc_dict['encoder_padding_mask'] if enc_dict is not None else None
        for idx, layer in enumerate(self.decoder_layers):
            enc = enc_dict['encoder_out'] if self.use_enc_last else enc_dict['encoder_history'][idx + 1]
            x, attn = layer(
                x,
                encoder_out=enc,
                encoder_padding_mask=encoder_padding_mask,
                self_attn_mask=self_attn_masks,
                self_attn_padding_mask=self_attn_padding_mask,
                **unused
            )
            inner_states.append(x)
            attn_states.append(attn)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        inputs_dict['attn'] = attn
        inputs_dict['inner_states'] = inner_states
        inputs_dict['attn_states'] = attn_states
        return x, inputs_dict

    def _buffered_nat_mask(self, key):
        sequence_length = key.size(0)
        self_masks = key.data.new(sequence_length, sequence_length).fill_(0).float()
        if self.mask_future:
            self_masks.fill_(float('-inf')).triu(1)
        if self.mask_self:
            diag_masks = torch.eye(sequence_length)
            if key.is_cuda:
                diag_masks = diag_masks.cuda(key.get_device())
            self_masks = self_masks + float('-inf') * diag_masks
        return self_masks


class PosNATDecoder(BasicNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, final_norm=True):
        super(BasicNATDecoder, self).__init__(dictionary)
        self.max_rel_len = args.max_rel_len
        self.relative_attn_type = args.relative_attn_type
        self._relative_keys = None
        self._relative_vals = None
        super(PosNATDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn, final_norm)

        self.pos_train_with_predict = args.pos_train_with_predict
        self.pos_valid_with_predict = args.pos_valid_with_predict
        self.pos_learning = not args.pos_no_learning and args.pos_loss_scale > 0
        embed_dim = args.decoder_embed_dim
        output = self.embed_tokens if self.embed_tokens is not None else self.embed_out
        self.position_oracle_searcher = PositionHeuristicSearcher(
            embed_dim, args.dropout, args.pos_search_type, args.pos_decompose, not args.pos_no_search_norm, out=output
        )
        if self.pos_train_with_predict or self.pos_valid_with_predict:
            self.in_position_predictor = PositionPredictor.build_model(args)
        else:
            self.in_position_predictor = None

        if getattr(args, 'state_bow_loss', 0) > 0:
            if args.state_share_output:
                self.state_out = OutLinear(args.decoder_output_dim, len(dictionary), bias=False)
            else:
                self.state_out = OutLinear(args.decoder_output_dim, len(dictionary), bias=False, _weight=output.weight)

        if getattr(args, 'layer_bow_loss', 0) > 0:
            if not args.layer_share_output:
                self.layer_out = OutLinear(args.decoder_output_dim, len(dictionary), bias=False)
            else:
                self.layer_out = OutLinear(args.decoder_output_dim, len(dictionary), bias=False, _weight=output.weight)

    def extract_features(self, prev_output_tokens, encoder_out=None, target_output_tokens=None, **unused):
        inputs_dict = self._bridging(encoder_out=encoder_out, prev_output_tokens=target_output_tokens)
        x = inputs_dict[LengthPredictorBridge.DECODE_INPUT_KEY]
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        # position info
        pos_oracle = self._search_pos_oracle(x, target_output_tokens, mask_dec=inputs_dict['decoder_masks'],
                                             mask_tgt=inputs_dict['target_masks'])
        pos_predict = self._predict_pos(
            self.in_position_predictor, x, encoder_out, mask_dec=inputs_dict['decoder_masks'],
            mask_enc=inputs_dict['source_masks'], pos_oracle=pos_oracle,
            beam_size=1 if 'beam_size' not in unused else unused['beam_size']
        )
        inputs_dict['position_oracle'] = pos_oracle
        inputs_dict['position_predict'] = pos_predict
        pos_use = self._get_position_info(pos_predict, pos_oracle)
        x, pos_key, pos_val = self._prepare_input(x, pos_use)

        # layer computing
        x, inputs_dict = self._decoding(
            x,
            enc_dict=encoder_out,
            inputs_dict=inputs_dict,
            pos_key=pos_key,
            pos_val=pos_val,
        )

        return x, inputs_dict

    def _search_pos_oracle(self, dec, tgt, mask_dec, mask_tgt):
        # if self.share_input_output_embed:
        #     tgt_embed = F.embedding(tgt, self.embed_tokens.weight * self.embed_scale)
        # else:
        #     tgt_embed = F.embedding(tgt, self.embed_out.weight * self.embed_scale)
        # return self.position_oracle_searcher(dec, tgt_embed, mask_dec, mask_tgt)
        return self.position_oracle_searcher(dec, tgt, mask_dec, mask_tgt, use_embed=False)

    def _predict_pos(self, model, x, encoder_out, mask_dec=None, mask_enc=None, pos_oracle=None, beam_size=1):
        inputs_index = None
        if self.pos_learning or not ((self.pos_train_with_predict and self.training) or (
                self.pos_valid_with_predict and not self.training)):
            # use oracle or learning with oracle
            if pos_oracle is not None:
                inputs_index = pos_oracle['os_pos'].absolute_pos
        if model is not None and self.pos_learning:
            predict_pos_ret = model(
                x, encoder_out, mask_dec, mask_enc,
                inputs_index=inputs_index,
                beam_size=beam_size
            )
        else:
            predict_pos_ret = None
        return predict_pos_ret

    def _get_position_info(self, predict_pos_ret, oracle_pos_ret=None):
        # consider which position should be use: position oracle or position predict
        if oracle_pos_ret is None:
            return predict_pos_ret['predict_pos']
        if predict_pos_ret is None:
            return oracle_pos_ret['os_pos']
        if (self.pos_train_with_predict and self.training) or (self.pos_valid_with_predict and not self.training):
            return predict_pos_ret['predict_pos']
        return oracle_pos_ret['os_pos']

    def _prepare_input(self, x, pos):
        if self.relative_attn_type == "shaw":
            absolute_pos = pos.absolute_pos
            output = rearrange_tensor_with_index(x, absolute_pos)
            return output, None, None
        else:
            relative_pos = pos.relative_pos(self.max_rel_len)
            if self._relative_keys is not None:
                return x, self._relative_keys(relative_pos), self._relative_vals(relative_pos)
            else:
                return x, relative_pos, None

    def _build_inner_layers(self, args):
        # construct the inner layer, there is two selection: one is BasicNATDecoderLayer, another is ShawDecoderLayer
        max_rel_len = self.max_rel_len
        embedding_dim = args.decoder_embed_dim // args.decoder_attention_heads
        block_cls = ShawDecoderLayer if args.relative_attn_type == 'shaw' else None

        def build_relative_embed():
            return RelativePositionEmbeddings(
                max_relative_position=max_rel_len,
                embedding_dim=embedding_dim,
                dropout=args.attention_dropout,
                direction=True
            )

        if args.share_rel_embed:
            self._relative_keys = build_relative_embed()
            self._relative_vals = build_relative_embed()
            self.decoder_layers = nn.ModuleList(
                # [
                #     block_cls(args, SharedModule(self._relative_keys), SharedModule(self._relative_vals),
                #               args.decoder_pos_attn, self.no_encoder_attn)
                #     for _ in range(args.decoder_layers)
                # ]
                [
                    block_cls(args, self._relative_keys, self._relative_vals,
                              args.decoder_pos_attn, self.no_encoder_attn)
                    for _ in range(args.decoder_layers)
                ]
            )

        else:
            self.decoder_layers = nn.ModuleList(
                [
                    block_cls(args, build_relative_embed(), build_relative_embed(), args.decoder_pos_attn,
                              self.no_encoder_attn)
                    for _ in range(args.decoder_layers)
                ]
            )
