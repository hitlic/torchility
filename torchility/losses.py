import torch.nn.functional as F
import torch
# --- losses

def masked_nll_loss(preds, targets):
    return F.nll_loss(preds[targets.mask], targets.data[targets.mask])


def masked_TOP_loss(preds, targets):
    """
    Top One Probability(TOP) loss，from <<Learning to Rank: From Pairwise Approach to Listwise Approach>>
    """
    preds = torch.squeeze(preds[targets.mask])
    targets = targets.data[targets.mask]
    preds_p = torch.softmax(preds, 0)
    targets_p = torch.softmax(targets, 0)
    loss = torch.mean(-torch.sum(targets_p*torch.log(preds_p)))
    return loss
