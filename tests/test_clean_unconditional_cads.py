"""Unit tests for stage A clean-unconditional CADS."""

from __future__ import annotations

import pytest
import torch

from feedback_cads import (
    CleanUnconditionalCADSConfig,
    ConditionNoiseStream,
    build_content_token_mask,
    corrupt_condition,
    expand_content_token_mask,
)
from feedback_cads.schedules import progress_pollution_cap


class _FakeTokenizer:
    model_max_length = 6

    def __call__(self, prompt: object, **kwargs: object) -> dict:
        del prompt, kwargs
        return {
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]]
            ),
            "special_tokens_mask": torch.tensor(
                [[1, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
            ),
        }


def _masked_stats(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    means = []
    stds = []
    for sample, sample_mask in zip(value.float(), mask):
        selected = sample[sample_mask].flatten()
        means.append(selected.mean())
        stds.append(selected.std(correction=0))
    return torch.stack(means), torch.stack(stds)


def test_stage_a_defaults_match_frozen_scheme() -> None:
    config = CleanUnconditionalCADSConfig()
    assert config.fixed_rho_a == pytest.approx(0.55)
    assert config.hold_until_progress == pytest.approx(0.10)
    assert config.hard_off_progress == pytest.approx(0.60)
    assert config.noise_scale == pytest.approx(0.20)
    assert config.rescale_mix == pytest.approx(0.70)
    assert config.content_tokens_only is True
    assert config.clean_unconditional is True


def test_progress_cap_has_exact_boundaries_and_correct_direction() -> None:
    values = [
        progress_pollution_cap(
            progress,
            hold_until=0.10,
            hard_off=0.60,
        )
        for progress in [0.0, 0.10, 0.35, 0.60, 1.0]
    ]
    assert values == pytest.approx([1.0, 1.0, 0.5, 0.0, 0.0])
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_content_mask_excludes_bos_eos_and_padding() -> None:
    mask = build_content_token_mask(_FakeTokenizer(), ["words", ""])
    assert torch.equal(
        mask,
        torch.tensor(
            [
                [False, True, True, False, False, False],
                [False, False, False, False, False, False],
            ]
        ),
    )


def test_content_mask_expands_in_prompt_major_order() -> None:
    mask = torch.tensor(
        [[False, True, False], [False, False, True]]
    )
    expanded = expand_content_token_mask(
        mask,
        prompt_batch_size=2,
        num_images_per_prompt=2,
        sequence_length=3,
        device="cpu",
    )
    assert torch.equal(expanded, mask.repeat_interleave(2, dim=0))


def test_masked_corruption_changes_only_content_and_restores_stats() -> None:
    generator = torch.Generator().manual_seed(808)
    clean = torch.randn(2, 6, 9, generator=generator)
    clean[1] = clean[1] * 2.5 + 4.0
    noise = torch.randn(clean.shape, generator=generator)
    mask = torch.tensor(
        [
            [False, True, True, True, False, False],
            [False, True, True, False, False, False],
        ]
    )
    result = corrupt_condition(
        clean,
        pollution=0.6,
        noise=noise,
        noise_scale=0.2,
        rescale_mix=1.0,
        mask=mask,
    )

    assert torch.equal(result[~mask], clean[~mask])
    assert not torch.equal(result[mask], clean[mask])
    clean_mean, clean_std = _masked_stats(clean, mask)
    result_mean, result_std = _masked_stats(result, mask)
    torch.testing.assert_close(result_mean, clean_mean, atol=1e-5, rtol=0)
    torch.testing.assert_close(result_std, clean_std, atol=1e-5, rtol=0)


def test_empty_content_mask_is_exact_identity_and_finite() -> None:
    clean = torch.randn(2, 6, 9, dtype=torch.float16)
    stream = ConditionNoiseStream(seed=19, batch_size=2, device="cpu")
    result = corrupt_condition(
        clean,
        pollution=0.6,
        noise=stream.sample(clean.shape),
        noise_scale=0.2,
        rescale_mix=0.7,
        mask=torch.zeros((2, 6), dtype=torch.bool),
    )
    assert torch.equal(result, clean)
    assert torch.isfinite(result).all()


def test_batched_q_zero_bypasses_noise_mask_and_rescaling_exactly() -> None:
    clean = torch.randn(2, 6, 9, dtype=torch.float16)
    noise = torch.full(clean.shape, float("nan"), dtype=torch.float32)
    mask = torch.tensor(
        [
            [False, True, True, False, False, False],
            [False, True, False, False, False, False],
        ]
    )
    result = corrupt_condition(
        clean,
        pollution=torch.zeros(2),
        noise=noise,
        noise_scale=0.2,
        rescale_mix=0.7,
        rescale=True,
        mask=mask,
    )
    assert result.data_ptr() == clean.data_ptr()
    assert torch.equal(result, clean)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fixed_rho_a": -0.1},
        {"fixed_rho_a": 1.1},
        {"hold_until_progress": 0.6, "hard_off_progress": 0.6},
        {"content_tokens_only": False},
        {"clean_unconditional": False},
    ],
)
def test_invalid_stage_a_config_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CleanUnconditionalCADSConfig(**kwargs)
