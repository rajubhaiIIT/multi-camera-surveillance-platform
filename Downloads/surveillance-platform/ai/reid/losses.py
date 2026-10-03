from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothCE(nn.Module):
    """Cross-entropy with label smoothing (stops the model becoming over-confident on 751 training people)."""

    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        return ((1 - self.eps) * nll + self.eps * (-logp.mean(dim=1))).mean()


def batch_hard_triplet(feats, labels, margin: float = 0.3):
    """For every image: pull its hardest same-person image closer than its hardest different-person image."""
    f = F.normalize(feats.float(), dim=1)
    d = torch.cdist(f, f)
    same = (labels[:, None] == labels[None, :]).float()
    d_pos = (d * same).max(dim=1).values
    d_neg = (d + 1e5 * same).min(dim=1).values
    return F.relu(d_pos - d_neg + margin).mean()
