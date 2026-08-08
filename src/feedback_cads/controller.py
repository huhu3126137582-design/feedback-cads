"""Frozen references and the Stage-B group proportional controller."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional, Union

import torch


@dataclass(frozen=True)
class FeedbackMVPConfig:
    """Frozen first-round settings for the Stage-B feedback MVP."""

    initial_rho: float = 0.55
    k_diversity: float = 0.08
    epsilon: float = 1e-6
    reference_margin_alpha: float = 0.0
    group_size: int = 8
    pool_size: int = 8

    def __post_init__(self) -> None:
        if not 0.0 <= self.initial_rho <= 1.0:
            raise ValueError("initial_rho must be in [0, 1].")
        if self.k_diversity < 0.0:
            raise ValueError("k_diversity must be non-negative.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.reference_margin_alpha < 0.0:
            raise ValueError("reference_margin_alpha must be non-negative.")
        if self.group_size < 2:
            raise ValueError("group_size must be at least two.")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive.")


@dataclass(frozen=True)
class FrozenDiversityReference:
    """Validated prompt-median reference curves from the frozen A* run."""

    step_count: int
    scheduler_timestep: tuple[float, ...]
    progress: tuple[float, ...]
    pollution_cap: tuple[float, ...]
    guidance_diversity: tuple[float, ...]
    predicted_clean_latent_diversity: tuple[float, ...]
    dz_control_eligible: tuple[bool, ...]
    dz_control_start_progress: float
    reference_margin_alpha: float
    rho_a_star: float
    sha256: str
    path: str

    def validate_step(
        self,
        step_index: int,
        *,
        scheduler_timestep: float,
        progress: float,
        pollution_cap: float,
        absolute_tolerance: float = 1e-9,
    ) -> None:
        """Reject a scheduler/schedule that does not match the reference."""

        if not 0 <= step_index < self.step_count:
            raise ValueError(f"Reference has no step {step_index}.")
        checks = {
            "scheduler_timestep": (
                scheduler_timestep,
                self.scheduler_timestep[step_index],
            ),
            "progress": (progress, self.progress[step_index]),
            "pollution_cap": (
                pollution_cap,
                self.pollution_cap[step_index],
            ),
        }
        for name, (actual, expected) in checks.items():
            if not math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=absolute_tolerance,
            ):
                raise ValueError(
                    f"Reference {name} mismatch at step {step_index}: "
                    f"expected {expected}, found {actual}."
                )


def _finite_float_tuple(
    data: dict[str, Any],
    field: str,
    *,
    expected_length: int,
    unit_interval: bool = False,
) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in data[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid reference field: {field}.") from error
    if len(result) != expected_length:
        raise ValueError(
            f"Reference field {field} must have {expected_length} entries."
        )
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"Reference field {field} must be finite.")
    if unit_interval and not all(0.0 <= value <= 1.0 for value in result):
        raise ValueError(f"Reference field {field} must lie in [0, 1].")
    return result


def load_frozen_diversity_reference(
    path: Union[str, Path],
    *,
    expected_sha256: Optional[str] = None,
    expected_steps: Optional[int] = 50,
    expected_reference_margin_alpha: float = 0.0,
) -> FrozenDiversityReference:
    """Load and strictly validate an A*-derived diversity reference."""

    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("Reference SHA-256 does not match the frozen value.")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Reference file is not valid JSON.") from error
    if not isinstance(data, dict):
        raise ValueError("Reference JSON must be an object.")
    step_count = int(data.get("step_count", 0))
    if step_count <= 0:
        raise ValueError("Reference step_count must be positive.")
    if expected_steps is not None and step_count != expected_steps:
        raise ValueError(
            f"Expected {expected_steps} reference steps, found {step_count}."
        )
    if data.get("formal_test_data_used") is not False:
        raise ValueError("Reference must not use formal test data.")

    step_index = tuple(int(value) for value in data.get("step_index", []))
    if step_index != tuple(range(step_count)):
        raise ValueError("Reference step_index must be contiguous from zero.")
    scheduler_timestep = _finite_float_tuple(
        data, "scheduler_timestep", expected_length=step_count
    )
    progress = _finite_float_tuple(
        data,
        "progress",
        expected_length=step_count,
        unit_interval=True,
    )
    pollution_cap = _finite_float_tuple(
        data,
        "pollution_cap",
        expected_length=step_count,
        unit_interval=True,
    )
    guidance = _finite_float_tuple(
        data,
        "guidance_diversity_reference",
        expected_length=step_count,
        unit_interval=True,
    )
    predicted_clean = _finite_float_tuple(
        data,
        "predicted_clean_latent_diversity_reference",
        expected_length=step_count,
        unit_interval=True,
    )
    if any(value <= 0.0 for value in guidance + predicted_clean):
        raise ValueError("Diversity references must be strictly positive.")

    eligibility_raw = data.get("dz_control_eligible")
    if not isinstance(eligibility_raw, list) or len(eligibility_raw) != step_count:
        raise ValueError("Invalid dz_control_eligible reference field.")
    if any(type(value) is not bool for value in eligibility_raw):
        raise ValueError("dz_control_eligible entries must be booleans.")
    eligibility = tuple(eligibility_raw)
    dz_start = float(data["dz_control_start_progress"])
    if not 0.0 <= dz_start <= 1.0:
        raise ValueError("dz_control_start_progress must lie in [0, 1].")
    expected_eligibility = tuple(value >= dz_start for value in progress)
    if eligibility != expected_eligibility:
        raise ValueError("Dz eligibility does not match the progress threshold.")

    alpha = float(data["reference_margin_alpha"])
    if not math.isclose(
        alpha,
        float(expected_reference_margin_alpha),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("Reference margin alpha does not match the expected value.")
    rho_a_star = float(data["rho_a_star"])
    if not math.isfinite(rho_a_star) or not 0.0 <= rho_a_star <= 1.0:
        raise ValueError("rho_a_star must lie in [0, 1].")

    return FrozenDiversityReference(
        step_count=step_count,
        scheduler_timestep=scheduler_timestep,
        progress=progress,
        pollution_cap=pollution_cap,
        guidance_diversity=guidance,
        predicted_clean_latent_diversity=predicted_clean,
        dz_control_eligible=eligibility,
        dz_control_start_progress=dz_start,
        reference_margin_alpha=alpha,
        rho_a_star=rho_a_star,
        sha256=actual_sha256,
        path=str(resolved),
    )


@dataclass(frozen=True)
class ProportionalControlUpdate:
    """One completed observation/update, with rho used kept separate."""

    rho_used: torch.Tensor
    rho_next: torch.Tensor
    guidance_deficit: torch.Tensor
    predicted_clean_latent_deficit: torch.Tensor
    diversity_deficit: torch.Tensor
    delta_rho: torch.Tensor
    guidance_target: float
    predicted_clean_latent_target: float
    dz_used: bool
    control_updated: bool


class GroupProportionalController:
    """Maintain one rho state per prompt group for Stage B."""

    def __init__(
        self,
        config: FeedbackMVPConfig,
        *,
        group_count: int,
        device: Union[str, torch.device],
    ) -> None:
        if group_count <= 0:
            raise ValueError("group_count must be positive.")
        self.config = config
        self.group_count = int(group_count)
        self._rho = torch.full(
            (self.group_count,),
            float(config.initial_rho),
            dtype=torch.float32,
            device=device,
        )

    @property
    def rho(self) -> torch.Tensor:
        """Return a copy so callers cannot mutate controller state."""

        return self._rho.clone()

    def update(
        self,
        *,
        guidance_diversity: torch.Tensor,
        predicted_clean_latent_diversity: torch.Tensor,
        guidance_reference: float,
        predicted_clean_latent_reference: float,
        dz_control_eligible: bool,
        pollution_cap: float,
    ) -> ProportionalControlUpdate:
        """Observe step i, then compute and store rho for step i+1."""

        guidance = torch.as_tensor(
            guidance_diversity, dtype=torch.float32, device=self._rho.device
        )
        predicted_clean = torch.as_tensor(
            predicted_clean_latent_diversity,
            dtype=torch.float32,
            device=self._rho.device,
        )
        expected_shape = (self.group_count,)
        if guidance.shape != expected_shape or predicted_clean.shape != expected_shape:
            raise ValueError(
                "Each diversity signal must contain one value per prompt group."
            )
        if not torch.isfinite(guidance).all() or not torch.isfinite(
            predicted_clean
        ).all():
            raise ValueError("Diversity observations must be finite.")
        if torch.any((guidance < 0.0) | (guidance > 1.0)) or torch.any(
            (predicted_clean < 0.0) | (predicted_clean > 1.0)
        ):
            raise ValueError("Diversity observations must lie in [0, 1].")
        if not 0.0 <= float(pollution_cap) <= 1.0:
            raise ValueError("pollution_cap must lie in [0, 1].")

        margin = 1.0 + float(self.config.reference_margin_alpha)
        guidance_target = float(guidance_reference) * margin
        predicted_clean_target = (
            float(predicted_clean_latent_reference) * margin
        )
        if (
            not math.isfinite(guidance_target)
            or not math.isfinite(predicted_clean_target)
            or guidance_target <= 0.0
            or predicted_clean_target <= 0.0
        ):
            raise ValueError("Diversity targets must be finite and positive.")

        epsilon = float(self.config.epsilon)
        guidance_deficit = (
            guidance_target - guidance
        ) / (guidance_target + epsilon)
        predicted_clean_deficit = (
            predicted_clean_target - predicted_clean
        ) / (predicted_clean_target + epsilon)
        diversity_deficit = (
            torch.maximum(guidance_deficit, predicted_clean_deficit)
            if dz_control_eligible
            else guidance_deficit
        )

        rho_used = self._rho.clone()
        control_updated = float(pollution_cap) > 0.0
        if control_updated:
            delta_rho = float(self.config.k_diversity) * diversity_deficit
            rho_next = (rho_used + delta_rho).clamp(0.0, 1.0)
        else:
            delta_rho = torch.zeros_like(rho_used)
            rho_next = rho_used.clone()
        self._rho = rho_next
        return ProportionalControlUpdate(
            rho_used=rho_used,
            rho_next=rho_next.clone(),
            guidance_deficit=guidance_deficit,
            predicted_clean_latent_deficit=predicted_clean_deficit,
            diversity_deficit=diversity_deficit,
            delta_rho=delta_rho,
            guidance_target=guidance_target,
            predicted_clean_latent_target=predicted_clean_target,
            dz_used=bool(dz_control_eligible and control_updated),
            control_updated=control_updated,
        )

