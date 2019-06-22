import torch
import torch.nn.functional as F


class LogCumsumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        # inputs = kwargs['inputs']
        m, _ = torch.max(inputs, dim=1, keepdim=True)
        # a transformation aiming for higher stability when computing softmax() with exp()
        y = inputs - m
        y = torch.exp(y)
        y_cumsum_t2h = torch.flip(torch.cumsum(torch.flip(y, dims=[1]), dim=1), dims=[1])
        # row-wise cumulative sum, from tail to head
        fd_output = torch.log(y_cumsum_t2h) + m  # corresponding to the '-m' operation
        ctx.save_for_backward(inputs, fd_output)

        return fd_output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_outputs[0]
        inputs, fd_output = ctx.saved_tensors
        bk_output = grad_output * (torch.exp(inputs) * torch.cumsum(torch.exp(-fd_output), dim=1))
        return bk_output


apply_LogCumsumExp = LogCumsumExp.apply


def listwise_loss(inputs, tgt_var, mask=None, scale=1.0):
    """

    Args:
        inputs: [batch_size, seq_len]
        tgt_var: [batch_size, tgt_inxdex]
        mask:
    Returns:
    """
    reindex = tgt_var.long().sort(dim=-1)[1]
    batch_preds = inputs.gather(dim=1, index=reindex)
    batch_logcumsumexps = apply_LogCumsumExp(batch_preds)
    batch_loss = torch.sum(batch_logcumsumexps - batch_preds)
    return batch_loss * scale


def pairwise_loss(inputs, tgt_var, mask=None, exp=False, scale=1.0):
    if scale > 0.0:
        diff = inputs[:, :, None] - inputs[:, None, :]
        target_diff = ((tgt_var[:, :, None] - tgt_var[:, None, :]) > 0).float()

        if exp:
            loss = torch.exp(F.relu(target_diff - diff)) - 1
        else:
            loss = F.relu(target_diff - diff)  # [batch_size,seq_len,seq_len]
        if mask is not None:
            mask = (mask[:, :, None] * mask[:, None, :]).float() * target_diff
        else:
            mask = target_diff
        loss = (loss * mask).sum() / (mask.sum() + 1e-9)
        return loss * scale
    else:
        return 0.0


def pointwise_loss(inputs, tgt_var, mask=None, scale=1.0):
    """

    Args:
        inputs: [batch_size, seq_len]
        tgt_var: [batch_size, seq_len]
        mask: [batch_size, seq_len]
        scale:

    Returns:

    """

    loss = F.mse_loss(inputs, tgt_var.float(), reduction='none')
    if mask is not None:
        origin_length = mask.sum(dim=-1).float()
        loss = (loss * mask.float()).sum(dim=-1) / (origin_length + 1e-9)
    return loss.mean() * scale


def indigo_tgt_var(tgt_var):
    batch_size, seq_len = tgt_var.size()
    ingigo_targets = []
    flatten_tgt_var = tgt_var.unsqueeze(-1).expand(batch_size, seq_len, 2).contiguous().view(batch_size, seq_len * 2)
    for step in range(1, seq_len):
        cur_index = tgt_var[:, step:step + 1]
        cur_tgt = flatten_tgt_var[:, :2 * step - 1].lt(cur_index).long().sum(dim=-1).unsqueeze(dim=-1)
        ingigo_targets.append(cur_tgt)
    return ingigo_targets


def indigo_loss(inputs, tgt_var, scale=1.0):
    batch_size, seq_len = tgt_var.size()
    indigo_targets = indigo_tgt_var(tgt_var)
    loss = 0
    for logits, target in zip(inputs, indigo_targets):
        logits = logits.log_softmax(dim=-1)
        loss += logits.gather(dim=1, index=target)
    loss /= seq_len
    return -loss.mean() * scale


def fix_indigo_tgt_var(tgt_var):
    batch_size, seq_len = tgt_var.size()
    ingigo_targets = []
    for step in range(1, seq_len):
        cur_index = tgt_var[:, step:step + 1]
        cur_candidate = tgt_var[:, :step]
        cur_diff = cur_index - cur_candidate
        cur_select = abs(cur_diff).min(dim=-1)[1].unsqueeze(dim=-1)
        cur_tgt = cur_select * 2 + cur_diff.gather(dim=1, index=cur_select).gt(0).long()
        ingigo_targets.append(cur_tgt)
    return ingigo_targets


def fix_indigo_loss(inputs, tgt_var, scale=1.0):
    batch_size, seq_len = tgt_var.size()
    indigo_targets = fix_indigo_tgt_var(tgt_var)
    loss = 0
    for logits, target in zip(inputs, indigo_targets):
        logits = logits.log_softmax(dim=-1)
        loss += logits.gather(dim=1, index=target)
    loss /= seq_len
    return -loss.mean() * scale


def predict_loss(inputs: torch.Tensor, tgt_var, masks=None, pad_id=-1):
    """

    Args:
        inputs: batch_size, seq_len, vocab_size
        tgt_var: batch_size, seq_len
        masks:  batch_size, seq_len
        pad_id:

    Returns:

    """
    inputs = inputs.log_softmax(dim=-1)
    inputs = inputs.contiguous().view(-1, inputs.size(-1))
    if masks is not None:
        masks = masks.float() * tgt_var.ne(pad_id).float()
    else:
        masks = tgt_var.ne(pad_id).float()
    batch_size, seq_len = masks.size()
    select_logits = inputs[range(batch_size * seq_len), tgt_var.contiguous().view(1, -1)]
    select_logits = select_logits.contiguous().view(batch_size, seq_len) * masks
    select_logits = select_logits.sum(dim=-1) / masks.sum(dim=-1)
    return -select_logits.mean()


def cross_entropy_loss(inputs, tgt_var, mask, scale=1.0):
    if scale > 0.0:
        mask = mask.float()
        inputs = inputs.contiguous().view(-1, inputs.size(-1))
        tgt_var = tgt_var.contiguous().view(-1)
        raw_loss = F.cross_entropy(inputs, tgt_var, reduction='none').view(*mask.size())
        raw_loss = (raw_loss * mask).sum(dim=tuple(range(mask.dim())[1:])) / mask.sum(dim=tuple(range(mask.dim())[1:]))
        return raw_loss.mean()
    else:
        return 0.0


def position_predict_loss(inputs, tgt_var=None, scale=1.0):
    """
    predict the defalut order
    Args:
        inputs: batch_size, nag_len, tgt_len
        tgt_var:
        scale:
    Returns:
    """
    if scale > 0.0:
        batch_size, nag_len, tgt_len = tgt_var.size()
        # if tgt_var is None:
        #     tgt_var = torch.Tensor(range(nag_len)).long()
        #     tgt_var = tgt_var.unsqueeze(0).expand(batch_size, nag_len)
        #     if inputs.is_cuda:
        #         tgt_var = tgt_var.cuda(inputs.get_device())
        #         tgt_var.requires_grad = False

        return F.cross_entropy(
            inputs.contiguous().view(-1, inputs.size(-1)),
            tgt_var.contiguous().view(-1)
        ) * scale
    else:
        return 0.0


def rel_rank_loss(inputs, tgt_var, mask=None, exp=False, max_rel=1, scale=1.0):
    if scale > 0.0:
        diff = inputs[:, :, None] - inputs[:, None, :] + max_rel
        target_diff = ((tgt_var[:, :, None] - tgt_var[:, None, :] + max_rel) > 0).float()

        if exp:
            loss = torch.exp(F.relu(target_diff - diff)) - 1
        else:
            loss = F.relu(target_diff - diff)  # [batch_size,seq_len,seq_len]
        if mask is not None:
            mask = (mask[:, :, None] * mask[:, None, :]).float() * target_diff
            loss = (loss * mask).sum() / (mask.sum() + 1e-9)
        else:
            loss = loss.sum()
        return loss * scale
    else:
        return 0.0


def relative_regression_loss(inputs, tgt_var, mask=None, exp=False, max_rel=1, scale=1.0):
    if scale > 0.0:
        rel_diff = inputs[:, :, None] - inputs[:, None, :].clamp(-max_rel, max_rel) + max_rel
        tgt_diff = (tgt_var[:, :, None] - tgt_var[:, None, :]).clamp(-max_rel, max_rel).float() + max_rel
        if exp:
            loss = torch.exp(F.mse_loss(rel_diff, tgt_diff, reduction='none')) - 1
        else:
            loss = F.mse_loss(rel_diff, tgt_diff, reduction='none')  # [batch_size,seq_len,seq_len]
        if mask is not None:
            mask = (mask[:, :, None] * mask[:, None, :]).float()
            loss = (loss * mask).sum() / (mask.sum() + 1e-9)
        else:
            loss = loss.sum()
        return loss * scale
    else:
        return 0.0


def rel_distance_loss(inputs, tgt_var, mask=None, max_rel=0, scale=1.0):
    if scale > 0:
        tgt_var = tgt_var.float()
        mask = mask.float()
        diff = inputs[:, :, None] - inputs[:, None, :]
        target_diff = tgt_var[:, :, None] - tgt_var[:, None, :]
        _mask = (mask.float().unsqueeze(-1) * mask.float().unsqueeze(-2))
        _dst_loss = F.mse_loss(diff, target_diff, reduction='none') * _mask
        _dst_loss = _dst_loss.sum(dim=(-1, -2)) / (_mask.sum(dim=(-1, -2)) + 1e-9)
        return _dst_loss.mean() * scale
    else:
        return 0.0


def origin_rankloss(inputs, target, mask, exp=False):
    diff = inputs[:, :, None] - inputs[:, None, :]
    target_diff = ((target[:, :, None] - target[:, None, :]) > 0).float()
    mask = mask[:, :, None] * mask[:, None, :] * target_diff

    if exp:
        loss = torch.exp(F.relu(target_diff - diff)) - 1
    else:
        loss = F.relu(target_diff - diff)
    loss = (loss * mask).sum() / (mask.sum() + 1e-9)

    return loss


def reorder_loss(inputs, scale=1.0, normalize=False):
    """

    Args:
        inputs: [batch_size,length]
        scale:
        normalize
    Returns:
    """
    if scale > 0.0:
        batch_size, max_len = inputs.size()
        inputs_diff = inputs[:, :, None] - inputs[:, None, :]
        target = torch.arange(max_len).unsqueeze(0).expand(batch_size, max_len).float()
        if inputs.is_cuda:
            target = target.cuda(inputs.get_device())
        target_diff = target[:, :, None] - target[:, None, :]
        select = (inputs_diff > 0) * (target_diff < 0)
        loss = (inputs_diff * select.float()).sum(dim=(-1, -2))
        if normalize:
            norm_select = target_diff * (target_diff > 0).float()
            norm_weight = norm_select.sum(dim=(-1, -2))
            loss /= norm_weight
        return loss.mean()
    else:
        return 0.0


def diversity_loss(inputs, scale=1.0):
    """
    a average
    Args:
        inputs: batch_size, seq_length, len_size
        scale
    Returns:
    """
    logits = inputs.softmax(dim=-1)
    logits = logits.mean(dim=1)  # batch_size, len_size
    mean_prob = 1.0 / logits.size(1)
    loss = ((logits - mean_prob) ** 2).sum(dim=-1).mean()
    return loss * scale


def batch_index_loss(inputs, tgt_var, ignore_index=1, tgt_mask=None):
    """

    Args:
        inputs: [batch,vocab_size]
        tgt_var: [batch,tgt_len]
        ignore_index:
        tgt_mask:

    Returns:

    """
    inputs = inputs.log_softmax(dim=-1)
    if tgt_mask is not None:
        tgt_mask = tgt_mask.float()
        tgt_mask *= tgt_var.ne(ignore_index).float()
    else:
        tgt_mask = tgt_var.ne(ignore_index).float()
    loss_len = tgt_mask.sum(dim=-1)
    log_loss = inputs.gather(dim=1, index=tgt_var)  # [batch,tgt_len]
    log_avg = ((log_loss * tgt_mask).sum(-1) + 1e-12) / (loss_len + 1e-9)
    loss = -log_avg
    return loss


def bag_of_word_loss(inputs, tgt_var, pad_id=1):
    """

    Args:
        inputs: [batch, vocab_size]
        tgt_var: [batch,seq_len]
        pad_id: default is 1
    Returns:
    """
    return batch_index_loss(inputs, tgt_var, pad_id).mean()


def batch_bag_of_word_loss(inputs, tgt_var, pad_id=1, average=True, tgt_mask=None):
    """
    Args:
        inputs: [batch, seq_len, vocab_size]
        tgt_var: [batch, tgt_len]
        pad_id: default is 1,
        average:
        tgt_mask: [batch, seq_len, tgt_len]
    Returns:
    """
    batch_size, seq_len, vocab_size = inputs.size()
    tgt_len = tgt_var.size(1)
    flatten_tgt_var = tgt_var.unsqueeze(1).expand(batch_size, seq_len, tgt_len).contiguous().view(-1, tgt_len)
    flatten_probs = inputs.contiguous().view(-1, vocab_size)
    loss = batch_index_loss(flatten_probs, flatten_tgt_var, pad_id, tgt_mask=tgt_mask)
    if average:
        return loss.mean()
    else:
        return loss.sum()


def window_bow_loss(inputs, tgt_var, window=1, pad_id=1, average=True):
    """
    Args:
        inputs: [batch_size,seq_len,vocab_size]
        tgt_var: [batch_size, tgt_len]
        window:
        pad_id:
        average:
    Returns:

    """
    if window == -1:
        return batch_bag_of_word_loss(inputs=inputs, tgt_var=tgt_var, pad_id=pad_id, average=average)
    if window == 0:
        return 0.0
    batch_size, tgt_len = tgt_var.size()
    _, seq_len, vocab_size = inputs.size()

    padded = torch.ones(batch_size, window).long() * pad_id
    if tgt_var.is_cuda:
        padded = padded.cuda(tgt_var.get_device())
    shifted_tgt_var = torch.cat([padded, tgt_var, padded], dim=-1)

    flatten_probs = inputs.contiguous()
    surround_tgt_var = [shifted_tgt_var[:, i:i + window * 2 + 1] for i in range(seq_len)]
    surround_tgt_var = torch.cat(surround_tgt_var, dim=-1).contiguous()
    loss = batch_index_loss(
        inputs=flatten_probs.view(-1, vocab_size),
        tgt_var=surround_tgt_var.view(batch_size * seq_len, -1),
        ignore_index=pad_id,
    )  # .view(batch_size, -1).sum(dim=-1)
    if average:
        loss = loss.mean()
        return loss
    return loss.sum()


def index_window_loss(inputs, tgt_var, index=None, window=1, pad_id=1, average=True):
    """

    Args:
        inputs: batch,log_len,vocab_size
        tgt_var: batch,tgt_var
        index: batch,log_len
        window: -1,0, or other
        pad_id: ignore index for loss
        average: size average for distributed
    Returns:

    """
    if window == -1:
        return batch_bag_of_word_loss(inputs=inputs, tgt_var=tgt_var, pad_id=pad_id, average=average)
    batch_size, tgt_len = tgt_var.size()
    log_len = inputs.size(1)
    array = list(range(-window, window + 1))
    if index is None:
        index = torch.arange(tgt_len).unsqueeze(0).expand(batch_size, tgt_len)
        if tgt_var.is_cuda:
            index = index.cuda(tgt_var.get_device())
    flatten_index = index.view(-1)  # [batch*log_len]
    flatten_index = torch.cat([(flatten_index + a) for a in array], dim=-1).contiguous(batch_size, -1)
    flatten_index = flatten_index.clamp(0, tgt_len)
    flatten_tgt_var = tgt_var.gather(1, flatten_index).view(batch_size * log_len, -1)
    flatten_probs = inputs.contiguous().view(batch_size * log_len, -1)
    loss = batch_index_loss(
        inputs=flatten_probs,
        tgt_var=flatten_tgt_var,
        ignore_index=pad_id
    )
    if average:
        loss = loss.mean()
        return loss
    return loss.sum()


def get_context_index(index, window, mask_center=True):
    """

    Args:
        index: batch_size, seq_len
        window:
        mask_center

    Returns:
        batch_size, seq_len, window*2
    """
    offset = list(range(-window, window + 1))
    if mask_center:
        offset.remove(0)
    offset = torch.Tensor(offset).long()
    if index.is_cuda:
        offset = offset.cuda(index.get_device())
    max_len = index.size(1)
    pri_ctx_index = index.unsqueeze(-1) + offset
    ctx_index = pri_ctx_index.clamp(0, max_len - 1)
    ctx_mask = ctx_index.eq(pri_ctx_index)
    return ctx_index, ctx_mask


def get_context_tgt_var(tgt_var, ctx_index):
    """

    Args:
        tgt_var: [batch_size, seq_len]
        ctx_index: [batch_size, seq_len, window*2]

    Returns:
        [batch_size, seq_len * window * 2]

    """
    batch_size, seq_len = tgt_var.size()
    flatten_ctx_index = ctx_index.view(batch_size, -1)
    flatten_ctx_tgt_var = tgt_var.gather(dim=1, index=flatten_ctx_index).contiguous().view(*ctx_index.size())
    return flatten_ctx_tgt_var


def batch_sequence_select(inputs, tgt_var):
    """

    Args:
        inputs: [batch_size, seq_len , vocab_size]
        tgt_var: [batch_size, seq_len , window*2]

    Returns:

    """
    vocab_size = inputs.size(-1)
    select_size = tgt_var.size(-1)
    inputs = inputs.contiguous().view(-1, vocab_size)
    select = tgt_var.contiguous().view(-1, select_size)

    slt_loss = inputs.gather(dim=1, index=select).contiguous().view(*tgt_var.size())
    return slt_loss


def skip_gram_loss(inputs, tgt_var=None, index=None, window=1, pad_id=-1, average=True):
    """
    for context predicting: window size for each index
    Args:
        inputs: [batch_size, seq_len, vocab_size]
        tgt_var: [batch_size, seq_len]
        index: [batch_size, seq_len]
        window: int
        pad_id: to set masks
        average:

    Returns:

    """
    if window <= 0:
        return 0.0
    batch, seq_len, _ = inputs.size()
    if tgt_var is None:
        tgt_var = torch.arange(seq_len).unsqueeze(0).expand(batch, -1).contiguous()
        if inputs.is_cuda:
            tgt_var = tgt_var.cuda(inputs.get_device())

    if index is None:
        index = torch.arange(seq_len).unsqueeze(0).expand(batch, -1).contiguous()
        if inputs.is_cuda:
            index = index.cuda(inputs.get_device())

    ctx_logits = inputs.log_softmax(dim=-1)
    ctx_index, ctx_index_mask = get_context_index(index=index, window=window)
    ctx_tgt_var = get_context_tgt_var(tgt_var, ctx_index=ctx_index)
    ctx_mask = ctx_index_mask.float() * (ctx_tgt_var.ne(pad_id).float())

    ctx_loss = -batch_sequence_select(inputs=ctx_logits, tgt_var=ctx_tgt_var)
    ctx_sum = ctx_mask.sum(dim=(1, 2))

    loss = (ctx_loss * ctx_mask).sum(dim=(1, 2)) / ctx_mask.sum(dim=(1, 2))
    if average:
        return loss.mean()
    return loss.sum()


def state_skip_gram_loss(inputs, threshold, tgt_var=None, index=None, window=1, average=True):
    """
    for context predicting: window size for each index
    Args:
        inputs: [batch_size, seq_len, vocab_size]
        tgt_var: [batch_size, seq_len]
        index: [batch_size, seq_len]
        window: int
        pad_id: to set masks
        average:

    Returns:

    """
    if window <= 0:
        return 0.0
    batch, seq_len, _ = inputs.size()
    if tgt_var is None:
        tgt_var = torch.arange(seq_len).unsqueeze(0).expand(batch, -1).contiguous()
        if inputs.is_cuda:
            tgt_var = tgt_var.cuda(inputs.get_device())

    if index is None:
        index = torch.arange(seq_len).unsqueeze(0).expand(batch, -1).contiguous()
        if inputs.is_cuda:
            index = index.cuda(inputs.get_device())

    ctx_logits = inputs.log_softmax(dim=-1)
    ctx_index, ctx_index_mask = get_context_index(index=index, window=window)
    ctx_tgt_var = get_context_tgt_var(tgt_var, ctx_index=ctx_index)
    if threshold is not None:
        threshold = threshold.contiguous().view(-1, 1, 1).expand(*ctx_tgt_var.size())
    else:
        threshold = seq_len
    ctx_mask = ctx_index_mask.float() * (ctx_tgt_var.lt(threshold).float())

    ctx_loss = -batch_sequence_select(inputs=ctx_logits, tgt_var=ctx_tgt_var)
    ctx_sum = ctx_mask.sum(dim=(1, 2))

    loss = (ctx_loss * ctx_mask).sum(dim=(1, 2)) / ctx_mask.sum(dim=(1, 2))
    if average:
        return loss.mean()
    return loss.sum()


def raw_predict_loss(inputs, tgt_var, masks=None, pad_id=-1):
    """

    Args:
        inputs: batch_size, seq_len, vocab_size
        tgt_var: batch_size, seq_len
        masks:  batch_size, seq_len
        pad_id:

    Returns:

    """

    inputs = inputs.log_softmax(dim=-1)
    inputs = inputs.contiguous().view(-1, inputs.size(-1))
    if masks is not None:
        masks = masks.float() * tgt_var.ne(pad_id).float()
    else:
        masks = tgt_var.ne(pad_id).float()
    batch_size, seq_len = masks.size()
    select_logits = inputs[range(batch_size * seq_len), tgt_var.contiguous().view(1, -1)]
    select_logits = select_logits.contiguous().view(batch_size, seq_len) * masks
    # select_logits = select_logits.sum(dim=-1) / masks.sum(dim=-1)
    # return -select_logits.mean()
    return -select_logits
