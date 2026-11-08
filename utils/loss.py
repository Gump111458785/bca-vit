"""Uncertainty-Calibrated Margin Loss (UCM Loss) for Transfer Transformers.
Minimizes out-of-domain transfer miscalibration and excessive target confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyCalibratedMarginLoss(nn.Module):
    def __init__(self, smoothing: float = 0.05, cal_weight: float = 0.20, use_ucm: bool = True):
        super().__init__()
        self.smoothing = smoothing
        self.cal_weight = cal_weight
        self.use_ucm = use_ucm
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not self.use_ucm:
            return self.ce(logits, targets)

        c = logits.size(1)
        log_preds = F.log_softmax(logits, dim=-1)
        loss_ce = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, targets, reduction="mean")
        loss_smooth = (1.0 - self.smoothing) * nll + (self.smoothing / c) * loss_ce

        # Calibrated margin constraint
        probs = F.softmax(logits, dim=1)
        top_probs, _ = torch.topk(probs, k=min(2, c), dim=1)
        if c >= 2:
            margin = top_probs[:, 0] - top_probs[:, 1]
            cal_penalty = torch.relu(margin - 0.88).mean()
        else:
            cal_penalty = torch.tensor(0.0, device=logits.device)

        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        return loss_smooth + self.cal_weight * cal_penalty - 0.03 * entropy
