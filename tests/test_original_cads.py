"""Unit tests for the audited Original CADS adaptation."""

from __future__ import annotations

import pytest
import torch

from feedback_cads.condition_noise import (
    ConditionNoiseStream,
    PaperCADSConfig,
    corrupt_cfg_conditions,
    corrupt_condition,
    negative_seed_from_positive,
)
from feedback_cads.schedules import paper_gamma, paper_pollution


def _stats(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = value.float().flatten(start_dim=1)
    return flattened.mean(dim=1), flattened.std(dim=1, correction=0)


def test_stable_diffusion_hyperparameters_match_paper_table_13() -> None:
    config = PaperCADSConfig()
    assert config.tau1 == pytest.approx(0.60)
    assert config.tau2 == pytest.approx(0.90)
    assert config.noise_scale == pytest.approx(0.25)
    assert config.rescale_mix == pytest.approx(1.0)
    assert config.source_model == "stable-diffusion-v2-1"
    assert config.source_sampler == "DDPM"
    assert config.source_num_steps == 100
    assert config.source_cfg_scale == pytest.approx(9.0)
    assert config.target_model == "stable-diffusion-v1-5"
    assert config.target_sampler == "DDIM"


def test_paper_schedule_uses_scheduler_timestep_and_correct_direction() -> None:
    assert paper_gamma(1.0, tau1=0.6, tau2=0.9) == pytest.approx(0.0)
    assert paper_gamma(0.9, tau1=0.6, tau2=0.9) == pytest.approx(0.0)
    assert paper_gamma(0.75, tau1=0.6, tau2=0.9) == pytest.approx(0.5)
    assert paper_gamma(0.6, tau1=0.6, tau2=0.9) == pytest.approx(1.0)
    assert paper_gamma(0.0, tau1=0.6, tau2=0.9) == pytest.approx(1.0)

    assert paper_pollution(
        999,
        num_train_timesteps=1000,
        tau1=0.6,
        tau2=0.9,
    ) == pytest.approx(1.0)
    assert paper_pollution(
        749.25,
        num_train_timesteps=1000,
        tau1=0.6,
        tau2=0.9,
    ) == pytest.approx(0.5)
    assert paper_pollution(
        0,
        num_train_timesteps=1000,
        tau1=0.6,
        tau2=0.9,
    ) == pytest.approx(0.0)


def test_zero_pollution_is_bitwise_identity() -> None:
    clean = torch.randn(2, 6, 8, dtype=torch.float16)
    result = corrupt_condition(
        clean,
        pollution=0.0,
        noise=torch.randn_like(clean),
        noise_scale=0.25,
        rescale_mix=1.0,
    )
    assert result.data_ptr() == clean.data_ptr()
    assert torch.equal(result, clean)


def test_rescaling_restores_scalar_stats_for_each_cfg_branch() -> None:
    generator = torch.Generator().manual_seed(31)
    positive = torch.randn(2, 7, 11, generator=generator)
    positive[1] = positive[1] * 3.0 + 8.0
    negative = torch.randn(2, 7, 11, generator=generator) * 0.5 - 4.0
    positive_noise = torch.randn(positive.shape, generator=generator)
    negative_noise = torch.randn(negative.shape, generator=generator)

    noisy_positive, noisy_negative = corrupt_cfg_conditions(
        positive,
        negative,
        pollution=0.5,
        positive_noise=positive_noise,
        negative_noise=negative_noise,
        config=PaperCADSConfig(),
    )
    for clean, noisy in (
        (positive, noisy_positive),
        (negative, noisy_negative),
    ):
        clean_mean, clean_std = _stats(clean)
        noisy_mean, noisy_std = _stats(noisy)
        torch.testing.assert_close(noisy_mean, clean_mean, atol=1e-5, rtol=0)
        torch.testing.assert_close(noisy_std, clean_std, atol=1e-5, rtol=0)

    assert not torch.equal(noisy_positive, positive)
    assert not torch.equal(noisy_negative, negative)


def test_original_cads_corrupts_all_tokens_in_both_branches() -> None:
    positive = torch.linspace(-1, 1, 4 * 5 * 6).reshape(4, 5, 6)
    negative = positive.flip(-1) + 2.0
    positive_noise = torch.randn_like(positive)
    negative_noise = torch.randn_like(negative)
    noisy_positive, noisy_negative = corrupt_cfg_conditions(
        positive,
        negative,
        pollution=0.8,
        positive_noise=positive_noise,
        negative_noise=negative_noise,
        config=PaperCADSConfig(),
    )
    positive_changed = (noisy_positive != positive).flatten(start_dim=1)
    negative_changed = (noisy_negative != negative).flatten(start_dim=1)
    assert positive_changed.all()
    assert negative_changed.all()


def test_positive_and_negative_streams_are_independent_and_reproducible() -> None:
    global_state = torch.random.get_rng_state()
    positive_a = ConditionNoiseStream(seed=41, batch_size=2, device="cpu")
    positive_b = ConditionNoiseStream(seed=41, batch_size=2, device="cpu")
    negative = ConditionNoiseStream(
        seed=negative_seed_from_positive(41),
        batch_size=2,
        device="cpu",
    )
    shape = (2, 5, 32)
    draw_a = positive_a.sample(shape)
    draw_b = positive_b.sample(shape)
    negative_draw = negative.sample(shape)

    assert torch.equal(draw_a, draw_b)
    assert not torch.equal(draw_a[0], draw_a[1])
    assert not torch.equal(draw_a, negative_draw)
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_fp16_degenerate_rescaling_is_finite() -> None:
    clean = torch.full((2, 5, 16), 30_000.0, dtype=torch.float16)
    clean[:, :, 1::2] = 30_016.0
    result = corrupt_condition(
        clean,
        pollution=1.0,
        noise=torch.zeros_like(clean),
        noise_scale=0.0,
        rescale_mix=1.0,
        rescale=True,
    )
    assert result.dtype == torch.float16
    assert torch.isfinite(result).all()
