from __future__ import annotations

import torch


def trajectory_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    object_mask: torch.Tensor,
    target_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute displacement errors.

    Args:
        pred: [B, N, H, 3]
        target: [B, N, H, 3]
        object_mask: [B, N]
        target_index: optional [B] pushed-object indices.
    """
    dist = torch.linalg.vector_norm(pred - target, dim=-1)
    mask = object_mask.bool().unsqueeze(-1).expand_as(dist)
    denom = mask.sum().clamp_min(1)
    ade = dist.masked_select(mask).sum() / denom

    final = dist[..., -1]
    final_mask = object_mask.bool()
    fde = final.masked_select(final_mask).mean()

    out = {"ade": ade, "fde": fde}
    if target_index is not None:
        b = torch.arange(pred.shape[0], device=pred.device)
        target_dist = dist[b, target_index.long()]
        out["target_ade"] = target_dist.mean()
        out["target_fde"] = target_dist[:, -1].mean()
    return out
