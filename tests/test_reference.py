"""Tests for reference-curve aggregation and validation."""

from __future__ import annotations

import copy

import pytest

from feedback_cads import build_prompt_median_reference


def _records() -> list[dict]:
    rows = []
    values = {"p0": [0.1, 0.2], "p1": [0.3, 0.4], "p2": [0.2, 0.6]}
    for prompt_id, signals in values.items():
        for step_index, guidance in enumerate(signals):
            progress = float(step_index)
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "step_index": step_index,
                    "scheduler_timestep": float(10 - step_index),
                    "progress": progress,
                    "pollution_cap": 1.0 - progress,
                    "rho": 0.55,
                    "pollution": 0.55 * (1.0 - progress),
                    "guidance_diversity": guidance,
                    "predicted_clean_latent_diversity": guidance / 2,
                }
            )
    return rows


def test_reference_is_stepwise_prompt_median() -> None:
    result = build_prompt_median_reference(
        _records(),
        expected_prompt_ids=["p0", "p1", "p2"],
        expected_steps=2,
        dz_control_start_progress=0.2,
    )
    assert result["guidance_diversity_reference"] == pytest.approx([0.2, 0.4])
    assert result["predicted_clean_latent_diversity_reference"] == pytest.approx(
        [0.1, 0.2]
    )
    assert result["dz_control_eligible"] == [False, True]


def test_reference_rejects_missing_duplicate_or_inconsistent_grid() -> None:
    with pytest.raises(ValueError, match="Expected 6"):
        build_prompt_median_reference(
            _records()[:-1],
            expected_prompt_ids=["p0", "p1", "p2"],
            expected_steps=2,
            dz_control_start_progress=0.2,
        )

    duplicate = _records()
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(ValueError, match="Duplicate"):
        build_prompt_median_reference(
            duplicate,
            expected_prompt_ids=["p0", "p1", "p2"],
            expected_steps=2,
            dz_control_start_progress=0.2,
        )

    inconsistent = _records()
    inconsistent[-1]["pollution"] = 0.25
    with pytest.raises(ValueError, match="inconsistent"):
        build_prompt_median_reference(
            inconsistent,
            expected_prompt_ids=["p0", "p1", "p2"],
            expected_steps=2,
            dz_control_start_progress=0.2,
        )
