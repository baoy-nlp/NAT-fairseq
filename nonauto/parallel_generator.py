import torch

from fairseq.sequence_generator import (
    SequenceGenerator,
    EnsembleModel
)
from nonauto.model_utils import apply_mask, log_softmax, INF


def beam_search(logits, mask_src, N=100):
    # prepare data
    nlogP = -log_softmax(logits).data
    maxL = nlogP.size(-1)
    overmask = torch.cat([mask_src[:, :, None], (1 - mask_src[:, :, None]).expand(*mask_src.size(), maxL - 1) * INF
                          + mask_src[:, :, None]], 2)
    nlogP = nlogP * overmask

    batch_size, src_len, L = logits.size()
    _, R = nlogP.sort(-1)

    def get_score(data, index):
        # avoid all zero
        # zero_mask = (index.sum(-2) == 0).float() * INF
        return data.gather(-1, index).sum(-2)

    heap_scores = torch.ones(batch_size, N) * INF
    heap_inx = torch.zeros(batch_size, src_len, N).long()
    heap_scores[:, :1] = get_score(nlogP, R[:, :, :1])
    if nlogP.is_cuda:
        heap_scores = heap_scores.cuda(nlogP.get_device())
        heap_inx = heap_inx.cuda(nlogP.get_device())

    def span(ins):
        inds = torch.eye(ins.size(1)).long()
        if ins.is_cuda:
            inds = inds.cuda(ins.get_device())
        return ins[:, :, None].expand(ins.size(0), ins.size(1), ins.size(1)) + inds[None, :, :]

    # iteration starts
    for k in range(1, N):
        cur_inx = heap_inx[:, :, k - 1]
        I_t = span(cur_inx).clamp(0, L - 1)  # B x N x N
        S_t = get_score(nlogP, R.gather(-1, I_t))
        S_t, _inx = torch.cat([heap_scores[:, k:], S_t], 1).sort(1)
        S_t[:, 1:] += ((S_t[:, 1:] - S_t[:, :-1]) == 0).float() * INF  # remove duplicates
        S_t, _inx2 = S_t.sort(1)
        I_t = torch.cat([heap_inx[:, :, k:], I_t], 2).gather(
            2, _inx.gather(1, _inx2)[:, None, :].expand(batch_size, src_len, _inx.size(-1)))
        heap_scores[:, k:] = S_t[:, :N - k]
        heap_inx[:, :, k:] = I_t[:, :, :N - k]

    # get the searched
    output = R.gather(-1, heap_inx)
    output = output.transpose(2, 1).contiguous().view(batch_size * N, src_len)  # (B x N) x Ts
    output = output
    mask_src = mask_src[:, None, :].expand(batch_size, N, src_len).contiguous().view(batch_size * N, src_len)

    return output, mask_src


class EnsembleParallelModel(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def has_decoder(self):
        return hasattr(self.models[0], 'decoder')

    def forward(self, *input):
        raise NotImplementedError

    @torch.no_grad()
    def forward_decoder(self, tokens=None, encoder_outs=None, targets=None, temperature=1.):
        if not self.has_decoder():
            return None
        return [model.decoder(tokens, encoder_outs[0], target_output_tokens=targets) for model in self.models]


class ParallelGenerator(SequenceGenerator):
    @torch.no_grad()
    def generate(
            self,
            models,
            sample,
            prefix_tokens=None,
            bos_token=None,
            **kwargs
    ):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
        """
        model = EnsembleParallelModel(models)
        if not self.retain_dropout:
            model.eval()

        encoder_input = {k: v for k, v in sample['net_input'].items() if k != 'prev_output_tokens'}
        encoder_outs = model.forward_encoder(encoder_input)
        decoder_outs = model.forward_decoder(encoder_outs=encoder_outs, targets=sample['target'])
        finalized = []

        def process_finalized(outputs, masks):
            position_scores, tokens = outputs.max(dim=-1)
            tokens = apply_mask(tokens, masks)
            position_scores = apply_mask(position_scores, masks.float())
            sequence_length = masks.long().sum(dim=-1)
            for idx, seq_len in enumerate(sequence_length.tolist()):
                _temp_scores = position_scores[idx, :seq_len].squeeze(0)
                _temp_tokens = tokens[idx, :seq_len].squeeze(0)
                temp = [
                    {
                        'alignment': None,
                        'positional_scores': _temp_scores,
                        'score': _temp_scores.sum() / seq_len,
                        'tokens': _temp_tokens
                    }
                ]
                finalized.append(temp)

        outputs = decoder_outs[0][0]
        masks = decoder_outs[0][1]['masks']
        process_finalized(outputs, masks)

        return finalized
