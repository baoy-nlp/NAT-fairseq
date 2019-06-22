import torch.nn.functional as F

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion


def prepare_cost(targets, out, target_mask=None, return_mask=None, PAD_ID=1):
    # targets : batch_size, seq_len
    # out     : batch_size, seq_len, vocab_size
    # target_mask : batch_size, seq_len
    if target_mask is None:
        target_mask = (targets != PAD_ID)

    if targets.size(1) < out.size(1):
        out = out[:, :targets.size(1), :]
    elif targets.size(1) > out.size(1):
        targets = targets[:, :out.size(1)]
        target_mask = target_mask[:, :out.size(1)]

    out_mask = target_mask.unsqueeze(-1).expand_as(out)

    if return_mask:
        return targets[target_mask], out[out_mask].view(-1, out.size(-1)), out_mask
    else:
        return targets[target_mask], out[out_mask].view(-1, out.size(-1))


@register_criterion('basic_nat_loss')
class BasicNATLoss(FairseqCriterion):
    """
    including loss item:
        length predict loss
        decoder output loss
    TODO: ADD Mode:
        BLEU Validation
    """

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--output-loss-scale', type=float, metavar='D', default=1.0)
        parser.add_argument('--length-loss-scale', type=float, metavar='D', default=1.0)

    def forward(self, model, sample, reduce=True):
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        output_predict_logits, outputs_dict = model(**sample['net_input'])
        output_loss = self.output_loss(model, output_predict_logits, sample) / sample_size
        length_loss = self.length_loss(model, outputs_dict) / sample['target'].size(0)
        loss = output_loss + length_loss

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def output_loss(self, model, output_predict_logits, sample):
        if not self.args.output_loss_scale > 0:
            return 0.0
        token_predict_loss = F.cross_entropy(
            # token_predict_loss = self.cross_entropy_loss(
            # model,
            output_predict_logits.view(-1,output_predict_logits.size(-1)),
            target=sample['target'].view(-1),
            ignore_index=self.padding_idx,
            reduction='sum'
        )
        # final_tgts, final_output = prepare_cost(sample['target'], out=output_predict_logits, PAD_ID=self.padding_idx)
        # token_predict_loss = F.cross_entropy(final_output, final_tgts, reduction='sum')
        return token_predict_loss

    def length_loss(self, model, outputs_dict):
        if 'predict_offset' not in outputs_dict or not self.args.length_loss_scale > 0:
            return 0.0
        else:
            # TODO: DIMENSION CHECK
            return self.cross_entropy_loss(
                model,
                outputs_dict['predict_offset_logits'],
                target=outputs_dict['reference_offset'],
                adaptive_softmax=False
            ) * self.args.length_loss_scale

    def cross_entropy_loss(self, model, logits, target, reduce=True, adaptive_softmax=True):
        log_probs = model.get_normalized_probs(logits, log_probs=True, adaptive_softmax=adaptive_softmax)
        log_probs = log_probs.contiguous().view(-1, log_probs.size(-1))
        reduction = 'sum' if reduce else 'none'
        loss = F.nll_loss(log_probs, target.contiguous().view(-1), ignore_index=self.padding_idx, reduction=reduction)
        # TODO: check the output of this mode
        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum,  # / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        # if sample_size != ntokens:
        #     agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
