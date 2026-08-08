"""Unit tests for group-level Feedback-CADS signals."""

from __future__ import annotations

import pytest
import torch

from feedback_cads import grouped_pairwise_cosine_diversity


def test_identical_candidates_have_zero_diversity() -> None:
    sample = torch.randn(1, 4, 3, 5)
    candidates = sample.repeat(8, 1, 1, 1)
    result = grouped_pairwise_cosine_diversity(
        candidates, group_size=8, pool_size=8
    )
    torch.testing.assert_close(result, torch.zeros(1), atol=1e-7, rtol=0)


def test_zero_candidates_have_zero_diversity() -> None:
    result = grouped_pairwise_cosine_diversity(
        torch.zeros(8, 4, 2, 2), group_size=8
    )
    assert result.item() == 0.0


def test_unique_pair_definition_matches_expected_half_cosine_scale() -> None:
    features = torch.tensor(
        [
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
            [[[-1.0]], [[0.0]]],
        ]
    )
    result = grouped_pairwise_cosine_diversity(
        features, group_size=3, pool_size=1
    )
    assert result.item() == pytest.approx(2.0 / 3.0)


def test_prompt_groups_are_not_mixed() -> None:
    first = torch.ones(4, 1, 2, 2)
    second = -torch.ones(4, 1, 2, 2)
    result = grouped_pairwise_cosine_diversity(
        torch.cat([first, second]), group_size=4, pool_size=2
    )
    torch.testing.assert_close(result, torch.zeros(2), atol=1e-7, rtol=0)


def test_fp16_inputs_use_finite_fp32_statistics_in_unit_interval() -> None:
    generator = torch.Generator().manual_seed(20260808)
    value = torch.randn(16, 4, 8, 8, generator=generator).half()
    result = grouped_pairwise_cosine_diversity(
        value, group_size=8, pool_size=8
    )
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    assert ((0.0 <= result) & (result <= 1.0)).all()


@pytest.mark.parametrize(
    "value,group_size,pool_size",
    [
        (torch.zeros(3, 4, 2), 3, 8),
        (torch.zeros(3, 4, 2, 2), 1, 8),
        (torch.zeros(3, 4, 2, 2), 2, 8),
        (torch.zeros(3, 4, 2, 2), 3, 0),
    ],
)
def test_invalid_group_shapes_are_rejected(
    value: torch.Tensor, group_size: int, pool_size: int
) -> None:
    with pytest.raises(ValueError):
        grouped_pairwise_cosine_diversity(
            value, group_size=group_size, pool_size=pool_size
        )

