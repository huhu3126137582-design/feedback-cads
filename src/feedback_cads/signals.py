"""Group-level online diversity signals for Feedback-CADS."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_pairwise_cosine_diversity(
    value: torch.Tensor,
    *,
    group_size: int,
    pool_size: int = 8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return one mean unique-pair cosine distance per prompt group.

    ``value`` must be prompt-major: all ``group_size`` candidates for the
    first prompt, followed by all candidates for the next prompt. Spatial
    tensors are pooled to ``pool_size`` square cells before flattening. The
    returned distance is ``(1-cosine)/2`` and is calculated in FP32.
    """

    if value.ndim != 4:
        raise ValueError("Expected a [batch, channels, height, width] tensor.")
    if group_size < 2:
        raise ValueError("group_size must be at least two.")
    if value.shape[0] % group_size != 0:
        raise ValueError("Batch size must be divisible by group_size.")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive.")

    pooled = F.adaptive_avg_pool2d(
        value.float(), output_size=(pool_size, pool_size)
    )
    features = pooled.flatten(start_dim=1)
    group_count = features.shape[0] // group_size
    features = features.reshape(group_count, group_size, -1)
    norms = torch.linalg.vector_norm(features, dim=-1)
    dots = features @ features.transpose(-1, -2)
    denominators = norms.unsqueeze(-1) * norms.unsqueeze(-2)
    similarities = dots / denominators.clamp_min(eps)
    both_zero = (norms.unsqueeze(-1) <= eps) & (
        norms.unsqueeze(-2) <= eps
    )
    similarities = torch.where(
        both_zero,
        torch.ones_like(similarities),
        similarities,
    ).clamp(-1.0, 1.0)

    pair_indices = torch.triu_indices(
        group_size,
        group_size,
        offset=1,
        device=value.device,
    )
    pair_similarities = similarities[
        :, pair_indices[0], pair_indices[1]
    ]
    return ((1.0 - pair_similarities) / 2.0).mean(dim=1)

