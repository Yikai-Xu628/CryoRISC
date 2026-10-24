from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def projective_angles(viewdirs: torch.Tensor) -> torch.Tensor:

    dots = (viewdirs @ viewdirs.T).abs().clamp(0.0, 1.0)
    return torch.rad2deg(torch.acos(dots))


def supervised_direction_contrastive_loss(
    embeddings, viewdirs, group_ids, temperature: float = 0.10,
    negative_min_angle_deg: float = 15.0, structure_ids=None,
):

    logits = (embeddings @ embeddings.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    n = len(embeddings)
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    positive = group_ids[:, None].eq(group_ids[None, :]) & ~eye
    angles = projective_angles(viewdirs)
    if structure_ids is None:
        same_structure = torch.ones_like(eye)
    else:
        same_structure = structure_ids[:, None].eq(structure_ids[None, :])
    positive &= same_structure
    negative = (angles >= negative_min_angle_deg) & same_structure & ~eye
    allowed = positive | negative
    exp_logits = torch.exp(logits) * allowed
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    mean_positive_log_prob = (
        (log_prob * positive).sum(dim=1) / positive_count.clamp_min(1)
    )
    return -mean_positive_log_prob[valid].mean()

