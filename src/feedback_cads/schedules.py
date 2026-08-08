"""CADS schedules with explicit paper and Diffusers time coordinates."""

from __future__ import annotations


def paper_gamma(
    normalized_timestep: float,
    *,
    tau1: float,
    tau2: float,
) -> float:
    """Return the CADS condition-cleanliness coefficient gamma(t).

    The paper uses reverse-diffusion time t=1 at the first/noisiest step and
    t=0 at the final/cleanest step.
    """

    if not 0.0 <= tau1 < tau2 <= 1.0:
        raise ValueError("Expected 0 <= tau1 < tau2 <= 1.")

    timestep = min(max(float(normalized_timestep), 0.0), 1.0)
    if timestep <= tau1:
        return 1.0
    if timestep >= tau2:
        return 0.0
    return (tau2 - timestep) / (tau2 - tau1)


def paper_pollution(
    scheduler_timestep: float,
    *,
    num_train_timesteps: int,
    tau1: float,
    tau2: float,
) -> float:
    """Return q=1-gamma from a Diffusers training timestep."""

    if num_train_timesteps <= 1:
        raise ValueError("num_train_timesteps must be greater than one.")
    normalized = float(scheduler_timestep) / float(
        num_train_timesteps - 1
    )
    return 1.0 - paper_gamma(
        normalized,
        tau1=tau1,
        tau2=tau2,
    )


def progress_pollution_cap(
    progress: float,
    *,
    hold_until: float,
    hard_off: float,
) -> float:
    """Return the A/B pollution cap from forward loop progress.

    ``progress=0`` is the first/noisiest reverse-diffusion step and
    ``progress=1`` is the final/cleanest step. This coordinate is deliberately
    separate from the scheduler-timestep coordinate used by Original CADS.
    """

    if not 0.0 <= hold_until < hard_off <= 1.0:
        raise ValueError("Expected 0 <= hold_until < hard_off <= 1.")

    value = min(max(float(progress), 0.0), 1.0)
    if value <= hold_until:
        return 1.0
    if value >= hard_off:
        return 0.0
    return (hard_off - value) / (hard_off - hold_until)
