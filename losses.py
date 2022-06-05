import torch
import config
import torch.nn.functional as F
import torch.nn as nn

def geometry_loss(pred, gt, smooth=0.1):
    '''

    Args:
        pred: torch.float [N, C]
        gt: torch.long [N]

    Returns:
        occupancy loss
    '''
    _gt = (((gt > 0) * (gt != 255))).long()
    pred_empty = pred[:, 0].unsqueeze(1)
    pred_ssc = pred[:, 1:]
    pred_occ = torch.sum(pred_ssc, 1).unsqueeze(1)

    pred_occ = torch.cat([pred_empty, pred_occ], dim=1)
    loss = torch.nn.CrossEntropyLoss(weight=config.class_weights_geo, ignore_index=255, label_smoothing=smooth).cuda()
    return loss(pred_occ, _gt)

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=255)
        return linear_combination(loss / n, nll, self.epsilon)