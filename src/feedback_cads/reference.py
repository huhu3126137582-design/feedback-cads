"""Validation and prompt-median aggregation for reference curves."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def build_prompt_median_reference(
    records: Iterable[dict[str, Any]],
    *,
    expected_prompt_ids: Iterable[str],
    expected_steps: int,
    dz_control_start_progress: float,
) -> dict[str, Any]:
    """Validate a complete prompt-step grid and return stepwise medians."""

    rows = list(records)
    prompt_ids = list(expected_prompt_ids)
    if expected_steps <= 0 or not prompt_ids:
        raise ValueError("Expected prompts and steps must be non-empty.")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Expected prompt IDs must be unique.")
    expected_count = len(prompt_ids) * expected_steps
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} prompt-step records, found {len(rows)}."
        )

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    expected_prompt_set = set(prompt_ids)
    for row in rows:
        prompt_id = str(row["prompt_id"])
        step_index = int(row["step_index"])
        if prompt_id not in expected_prompt_set:
            raise ValueError(f"Unexpected prompt ID: {prompt_id}")
        if not 0 <= step_index < expected_steps:
            raise ValueError(f"Invalid step index: {step_index}")
        key = (prompt_id, step_index)
        if key in by_key:
            raise ValueError(f"Duplicate prompt-step record: {key}")
        by_key[key] = row
        by_step[step_index].append(row)

    result: dict[str, list[Any] | int | float] = {
        "step_count": expected_steps,
        "prompt_count": len(prompt_ids),
        "dz_control_start_progress": float(dz_control_start_progress),
        "step_index": [],
        "scheduler_timestep": [],
        "progress": [],
        "pollution_cap": [],
        "rho": [],
        "pollution": [],
        "guidance_diversity_reference": [],
        "predicted_clean_latent_diversity_reference": [],
        "dz_control_eligible": [],
    }
    metadata_fields = (
        "scheduler_timestep",
        "progress",
        "pollution_cap",
        "rho",
        "pollution",
    )
    for step_index in range(expected_steps):
        step_rows = by_step[step_index]
        if len(step_rows) != len(prompt_ids):
            raise ValueError(
                f"Step {step_index} has {len(step_rows)} prompts; "
                f"expected {len(prompt_ids)}."
            )
        for field in metadata_fields:
            values = np.asarray(
                [float(row[field]) for row in step_rows], dtype=np.float64
            )
            if not np.isfinite(values).all() or not np.allclose(
                values, values[0], atol=0.0, rtol=0.0
            ):
                raise ValueError(
                    f"Step metadata field {field} is inconsistent at step "
                    f"{step_index}."
                )
            result[field].append(float(values[0]))

        guidance = np.asarray(
            [float(row["guidance_diversity"]) for row in step_rows],
            dtype=np.float64,
        )
        predicted_clean = np.asarray(
            [
                float(row["predicted_clean_latent_diversity"])
                for row in step_rows
            ],
            dtype=np.float64,
        )
        if not np.isfinite(guidance).all() or not np.isfinite(
            predicted_clean
        ).all():
            raise ValueError("Reference signals must be finite.")
        if (
            (guidance < 0).any()
            or (guidance > 1).any()
            or (predicted_clean < 0).any()
            or (predicted_clean > 1).any()
        ):
            raise ValueError("Reference signals must lie in [0, 1].")

        progress = float(step_rows[0]["progress"])
        result["step_index"].append(step_index)
        result["guidance_diversity_reference"].append(
            float(np.median(guidance))
        )
        result["predicted_clean_latent_diversity_reference"].append(
            float(np.median(predicted_clean))
        )
        result["dz_control_eligible"].append(
            progress >= dz_control_start_progress
        )
    return result
