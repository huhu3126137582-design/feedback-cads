"""Paper-algorithm-faithful CADS condition corruption primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch


_SEED_MODULUS = 2**63 - 1
_SAMPLE_SEED_STRIDE = 1_000_003
_NEGATIVE_SEED_OFFSET = 10_000_019


@dataclass(frozen=True)
class PaperCADSConfig:
    """CADS hyperparameters reported for Stable Diffusion in Table 13.

    The source experiment used SD v2.1, DDPM 100 steps, and CFG 9. This
    project applies the same CADS algorithm and hyperparameters to SD v1.5,
    DDIM 50 steps, and CFG 7.5; it does not claim numerical reproduction of
    the paper's SD v2.1 experiment.
    """

    tau1: float = 0.60
    tau2: float = 0.90
    noise_scale: float = 0.25
    rescale_mix: float = 1.0
    rescale: bool = True
    eps: float = 1e-6

    source_model: str = "stable-diffusion-v2-1"
    source_sampler: str = "DDPM"
    source_num_steps: int = 100
    source_cfg_scale: float = 9.0
    target_model: str = "stable-diffusion-v1-5"
    target_sampler: str = "DDIM"
    target_num_steps: int = 50
    target_cfg_scale: float = 7.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau1 < self.tau2 <= 1.0:
            raise ValueError("Expected 0 <= tau1 < tau2 <= 1.")
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative.")
        if not 0.0 <= self.rescale_mix <= 1.0:
            raise ValueError("rescale_mix must be in [0, 1].")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")


@dataclass(frozen=True)
class CleanUnconditionalCADSConfig:
    """Frozen open-loop A configuration shared by later B/C stages."""

    noise_scale: float = 0.20
    rescale_mix: float = 0.70
    fixed_rho_a: float = 0.55
    hold_until_progress: float = 0.10
    hard_off_progress: float = 0.60
    content_tokens_only: bool = True
    clean_unconditional: bool = True
    rescale: bool = True
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative.")
        if not 0.0 <= self.rescale_mix <= 1.0:
            raise ValueError("rescale_mix must be in [0, 1].")
        if not 0.0 <= self.fixed_rho_a <= 1.0:
            raise ValueError("fixed_rho_a must be in [0, 1].")
        if not (
            0.0
            <= self.hold_until_progress
            < self.hard_off_progress
            <= 1.0
        ):
            raise ValueError(
                "Expected 0 <= hold_until_progress "
                "< hard_off_progress <= 1."
            )
        if not self.content_tokens_only:
            raise ValueError(
                "Stage A requires content_tokens_only=True; full-embedding "
                "corruption belongs to an explicit ablation."
            )
        if not self.clean_unconditional:
            raise ValueError(
                "Stage A requires clean_unconditional=True; perturbing both "
                "CFG branches belongs to Original CADS."
            )
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")


def build_content_token_mask(tokenizer: object, prompt: object) -> torch.Tensor:
    """Tokenize prompts and select non-special, non-padding CLIP tokens."""

    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    attention_mask = encoded["attention_mask"].to(dtype=torch.bool)
    special_mask = encoded["special_tokens_mask"].to(dtype=torch.bool)
    if attention_mask.shape != special_mask.shape:
        raise RuntimeError(
            "Tokenizer attention and special-token masks must have the same "
            "shape."
        )
    return attention_mask & ~special_mask


def expand_content_token_mask(
    mask: torch.Tensor,
    *,
    prompt_batch_size: int,
    num_images_per_prompt: int,
    sequence_length: int,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Expand a prompt-level mask in Diffusers' prompt-major batch order."""

    if prompt_batch_size <= 0 or num_images_per_prompt <= 0:
        raise ValueError("Batch sizes must be positive.")
    value = torch.as_tensor(mask, dtype=torch.bool)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] != sequence_length:
        raise ValueError(
            "Content mask must have shape [prompt_batch, sequence_length]."
        )

    effective_batch = prompt_batch_size * num_images_per_prompt
    if value.shape[0] == prompt_batch_size:
        value = value.repeat_interleave(num_images_per_prompt, dim=0)
    elif value.shape[0] != effective_batch:
        raise ValueError(
            "Content mask batch must equal prompt batch or effective batch."
        )
    return value.to(device=device)


def negative_seed_from_positive(seed: int) -> int:
    """Derive a deterministic, named negative-condition stream seed."""

    return (int(seed) + _NEGATIVE_SEED_OFFSET) % _SEED_MODULUS


class ConditionNoiseStream:
    """Independent per-candidate condition noise without global RNG use."""

    def __init__(
        self,
        *,
        seed: int,
        batch_size: int,
        device: Union[str, torch.device],
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.draw_calls = 0
        self._generators = []
        for sample_index in range(self.batch_size):
            sample_seed = (
                self.seed + _SAMPLE_SEED_STRIDE * sample_index
            ) % _SEED_MODULUS
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sample_seed)
            self._generators.append(generator)

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        """Draw FP32 Gaussian noise with one generator per candidate."""

        shape = tuple(int(dimension) for dimension in shape)
        if not shape or shape[0] != self.batch_size:
            raise ValueError(
                "Noise batch dimension must equal stream batch_size."
            )
        sample_shape = (1, *shape[1:])
        result = torch.cat(
            [
                torch.randn(
                    sample_shape,
                    generator=generator,
                    device=self.device,
                    dtype=torch.float32,
                )
                for generator in self._generators
            ],
            dim=0,
        )
        self.draw_calls += 1
        return result


def _broadcast_pollution(
    pollution: Union[float, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    value = torch.as_tensor(
        pollution,
        dtype=torch.float32,
        device=device,
    )
    if value.ndim == 0:
        value = value.expand(batch_size)
    if value.ndim == 1 and value.shape[0] == batch_size:
        value = value.view(batch_size, 1, 1)
    if value.shape != (batch_size, 1, 1):
        raise ValueError(
            "pollution must be scalar, [batch], or [batch, 1, 1]."
        )
    if torch.any((value < 0.0) | (value > 1.0)):
        raise ValueError("pollution values must be in [0, 1].")
    return value


def _masked_scalar_stats(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask3 = mask.unsqueeze(-1).expand_as(value)
    weights = mask3.to(dtype=value.dtype)
    count = weights.sum(dim=(1, 2), keepdim=True)
    safe_count = count.clamp_min(1.0)
    mean = (value * weights).sum(dim=(1, 2), keepdim=True) / safe_count
    variance = (
        (value - mean).square() * weights
    ).sum(dim=(1, 2), keepdim=True) / safe_count
    std = variance.clamp_min(0.0).sqrt().clamp_min(eps)
    return mean, std, count > 0


def corrupt_condition(
    clean: torch.Tensor,
    *,
    pollution: Union[float, torch.Tensor],
    noise: torch.Tensor,
    noise_scale: float,
    rescale_mix: float,
    rescale: bool = True,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply CADS corruption and scalar-stat rescaling per sample.

    For the Original CADS baseline, ``mask`` is None, so every token and
    channel in each positive/null condition is included. Reduction never
    crosses the batch dimension. FP32 is used for all corruption statistics.
    """

    if clean.ndim != 3:
        raise ValueError("Expected condition shape [batch, tokens, dim].")
    if tuple(noise.shape) != tuple(clean.shape):
        raise ValueError("Noise and clean condition shapes must match.")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative.")
    if not 0.0 <= rescale_mix <= 1.0:
        raise ValueError("rescale_mix must be in [0, 1].")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    batch_size, sequence_length, _ = clean.shape
    q = _broadcast_pollution(
        pollution,
        batch_size=batch_size,
        device=clean.device,
    )
    if torch.count_nonzero(q).item() == 0:
        return clean

    if mask is None:
        condition_mask = torch.ones(
            (batch_size, sequence_length),
            dtype=torch.bool,
            device=clean.device,
        )
    else:
        condition_mask = torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=clean.device,
        )
        if condition_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                "Mask shape must equal [batch, sequence_length]."
            )

    clean32 = clean.float()
    noise32 = noise.to(device=clean.device, dtype=torch.float32)
    raw = (
        (1.0 - q).clamp_min(0.0).sqrt() * clean32
        + float(noise_scale) * q.clamp_min(0.0).sqrt() * noise32
    )
    mask3 = condition_mask.unsqueeze(-1)
    raw = torch.where(mask3, raw, clean32)
    if not rescale:
        return raw.to(dtype=clean.dtype)

    clean_mean, clean_std, has_values = _masked_scalar_stats(
        clean32,
        condition_mask,
        eps=eps,
    )
    raw_mean, raw_std, _ = _masked_scalar_stats(
        raw,
        condition_mask,
        eps=eps,
    )
    restored = (raw - raw_mean) / raw_std * clean_std + clean_mean
    mixed = (
        float(rescale_mix) * restored
        + (1.0 - float(rescale_mix)) * raw
    )
    mixed = torch.where(mask3, mixed, clean32)
    exact_clean = (q == 0.0) | ~has_values
    mixed = torch.where(exact_clean, clean32, mixed)
    return mixed.to(dtype=clean.dtype)


def corrupt_cfg_conditions(
    clean_positive: torch.Tensor,
    clean_negative: torch.Tensor,
    *,
    pollution: Union[float, torch.Tensor],
    positive_noise: torch.Tensor,
    negative_noise: torch.Tensor,
    config: PaperCADSConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt and independently rescale both CFG branches."""

    if clean_positive.shape != clean_negative.shape:
        raise ValueError("Positive and negative condition shapes must match.")
    positive = corrupt_condition(
        clean_positive,
        pollution=pollution,
        noise=positive_noise,
        noise_scale=config.noise_scale,
        rescale_mix=config.rescale_mix,
        rescale=config.rescale,
        mask=None,
        eps=config.eps,
    )
    negative = corrupt_condition(
        clean_negative,
        pollution=pollution,
        noise=negative_noise,
        noise_scale=config.noise_scale,
        rescale_mix=config.rescale_mix,
        rescale=config.rescale,
        mask=None,
        eps=config.eps,
    )
    return positive, negative
