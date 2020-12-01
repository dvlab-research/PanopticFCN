import torch
import torch.nn.functional as F

__all__ = ['sigmoid_focal_loss', 'weighted_dice_loss']


def weighted_dice_loss(
    prediction,
    target_seg,
    gt_num,
    index_mask,
    instance_num: int = 0,
    weighted_val: float = 1.0,
    weighted_num: int = 1,
    mode: str = "thing",
    reduction: str = "sum",
    eps: float = 1e-8,
):
    """
    Weighted version of Dice Loss used in PanopticFCN for multi-positive optimization.

    Args:
        prediction: prediction for Things or Stuff,
        target_seg: segmentation target for Things or Stuff,
        gt_num: ground truth number for Things or Stuff,
        index_mask: positive index mask for Things or Stuff,
        instance_num: instance number of Things or Stuff,
        weighted_val: values of k positives,
        weighted_num: number k for weighted loss,
        mode: used for things or stuff,
        reduction: 'none' | 'mean' | 'sum'
                   'none': No reduction will be applied to the output.
                   'mean': The output will be averaged.
                   'sum' : The output will be summed.
        eps: the minimum eps,
    """
    # avoid Nan
    if gt_num == 0:
        loss = prediction[0][0].sigmoid().mean() + eps
        return loss * gt_num
    
    n, _, h, w = target_seg.shape
    if mode == "thing":
        prediction = prediction.reshape(n, instance_num, weighted_num, h, w)
        prediction = prediction.reshape(-1, weighted_num, h, w)[index_mask,...]
        target_seg = target_seg.unsqueeze(2).expand(n, instance_num, weighted_num, h, w)
        target_seg = target_seg.reshape(-1, weighted_num, h, w)[index_mask,...]
        weighted_val = weighted_val.reshape(-1, weighted_num)[index_mask,...]
        weighted_val = weighted_val / torch.clamp(weighted_val.sum(dim=-1,keepdim=True), min=eps)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.reshape(int(gt_num), weighted_num, h*w)
        target_seg = target_seg.reshape(int(gt_num), weighted_num, h*w)
    elif mode == "stuff":
        prediction = prediction.reshape(-1, h, w)[index_mask,...]
        target_seg = target_seg.reshape(-1, h, w)[index_mask,...]
        prediction = torch.sigmoid(prediction)
        prediction = prediction.reshape(int(gt_num), h*w)
        target_seg = target_seg.reshape(int(gt_num), h*w)
    else: 
        raise ValueError
    
    # calculate dice loss
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)
    # normalize the loss
    loss = loss * weighted_val

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    return loss


def sigmoid_focal_loss(
    inputs,
    targets,
    mode: str = "thing",
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        mode: A string used to indicte the optimization mode.
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # pixel-wise loss for stuff
    if mode == "stuff":
        loss = loss.reshape(*loss.shape[:2], -1).mean(dim=-1)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
