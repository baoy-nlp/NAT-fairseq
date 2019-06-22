import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import register_criterion
from nonauto.criterions.nat_loss import BasicNATLoss
from nonauto.loss_utils import raw_predict_loss


@register_criterion('pos_nat_loss')
class PosNATLoss(BasicNATLoss):
    """
    Jointly Training
    append:
        position-aware loss
    """

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--output-loss-scale', type=float, metavar='D', default=1.0)
        parser.add_argument('--length-loss-scale', type=float, metavar='D', default=0.0)
        parser.add_argument('--pos-loss-scale', type=float, metavar='D', default=0.0)
        parser.add_argument('--embed-loss-scale', type=float, metavar='D', default=0.0)
        parser.add_argument('--prior-loss-scale', type=float, metavar='D', default=0.0)
        parser.add_argument('--state-bow-loss', type=float, metavar='D', default=0.0)
        parser.add_argument('--state-bow-weight', type=float, nargs="*", metavar='D', default=[0.0])
        parser.add_argument('--state-share-output', action='store_true')
        parser.add_argument('--layer-bow-loss', type=float, metavar='D', default=0.0)
        parser.add_argument('--layer-bow-weight', type=float, nargs="*", metavar='D', default=[0.0])
        parser.add_argument('--layer-share-output', action='store_true')

    def forward(self, model, sample, reduce=True):
        output_predict_logits, outputs_dict = model(**sample['net_input'], target_output_tokens=sample['target'])

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        output_loss = self.output_loss(model
                                       , output_predict_logits, sample) / sample_size
        length_loss = self.length_loss(model, outputs_dict) / sample['target'].size(0)
        position_loss = self.position_loss(outputs_dict, reduce) / sample_size
        embed_loss = self.embedding_loss(outputs_dict, reduce) / sample_size
        prior_loss = self.prior_loss(outputs_dict, reduce) / sample_size
        loss = output_loss
        loss += length_loss
        loss += position_loss
        loss += embed_loss
        loss += prior_loss
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def position_loss(self, outputs_dict, reduce=True):
        if not self.args.pos_loss_scale > 0 or outputs_dict['position_predict'] is None:
            return 0.0
        logits = outputs_dict['position_predict']['predict_pos_logits']
        masks = outputs_dict['position_predict']['predict_masks']
        target_pos = outputs_dict['position_oracle']['os_pos']

        if logits.dim() > 3:
            # TODO: RELATIVE POSITION LOSS MODULE
            # relative position
            # pos_class_num = logits.size(-1)
            # pos_oracle = target_pos.relative_oracle(pos_class_num)
            loss = 0.0
        else:
            ref_order = target_pos.absolute_pos
            batch_size, seq_len = ref_order.size()
            tgt_index = ref_order.long().sort(dim=-1)[1]
            loss = F.cross_entropy(
                input=logits.contiguous().view(batch_size * seq_len, -1) + 1e-10,
                target=tgt_index.contiguous().view(-1),
                reduction='none',
            ) * self.args.pos_loss_scale  # return is [batch, seq_len]
            loss = loss.contiguous().view(batch_size, seq_len)
        if masks is not None:
            loss = loss * masks.float()
        if reduce:
            loss = loss.sum()
        return loss

    def embedding_loss(self, outputs_dict, reduce=True):
        if not self.args.embed_loss_scale > 0 or outputs_dict['position_oracle'] is None:
            return 0.0
        logits = outputs_dict['position_oracle']['os_logits']
        tgt_var = outputs_dict['position_oracle']['os_pos'].absolute_pos
        masks = outputs_dict['position_oracle']['os_dec_mask']
        loss = raw_predict_loss(
            logits,
            tgt_var,
            masks
        )
        if reduce:
            loss = loss.sum()

        return loss * self.args.embed_loss_scale

    def prior_loss(self, outputs_dict, reduce=True):
        if not self.args.prior_loss_scale > 0 or outputs_dict['position_oracle'] is None:
            return 0.0
        logits = outputs_dict['position_oracle']['os_logits']
        tgt_var = torch.arange(logits.size(1)).unsqueeze(0).expand(logits.size(0), -1)
        masks = outputs_dict['position_oracle']['os_dec_mask']
        if logits.is_cuda:
            tgt_var = tgt_var.cuda(logits.get_device())

        loss = raw_predict_loss(
            logits,
            tgt_var,
            masks
        )
        if reduce:
            loss = loss.sum()

        return loss * self.args.prior_loss_scale

    def layer_bow_loss(self, model, outputs_dict, reduce=True):
        if not self.args.layer_bow_loss > 0:
            return 0.0
        inner_states = outputs_dict['inner_states']

        raise NotImplementedError

    def state_bow_loss(self, model, outputs_dict, reduct=True):
        if not self.args.state_bow_loss > 0:
            return 0.0
        inner_states = outputs_dict['inner_states']
        raise NotImplementedError
