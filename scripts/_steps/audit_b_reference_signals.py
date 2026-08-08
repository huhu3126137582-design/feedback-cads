#!/usr/bin/env python3
"""Audit Stage-B Dg/Dz reference signals from the frozen A* run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads import build_prompt_median_reference  # noqa: E402
from feedback_cads.experiments import read_jsonl, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/a_star_reference_coco_dev50",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/b_stage1_signal_audit",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)),
    }


def main() -> None:
    args = parse_args()
    reference_dir = args.reference_dir.resolve()
    per_step_path = reference_dir / "per_step.jsonl"
    curve_path = reference_dir / "reference_curves.json"
    validation_path = reference_dir / "validation_report.json"
    records = read_jsonl(per_step_path)
    curves: dict[str, Any] = json.loads(curve_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    prompt_ids = sorted({str(row["prompt_id"]) for row in records})
    recomputed = build_prompt_median_reference(
        records,
        expected_prompt_ids=prompt_ids,
        expected_steps=int(curves["step_count"]),
        dz_control_start_progress=float(
            curves["dz_control_start_progress"]
        ),
    )
    curve_fields = (
        "step_index",
        "scheduler_timestep",
        "progress",
        "pollution_cap",
        "rho",
        "pollution",
        "guidance_diversity_reference",
        "predicted_clean_latent_diversity_reference",
        "dz_control_eligible",
    )
    exact_median_reproduction = all(
        recomputed[field] == curves[field] for field in curve_fields
    )

    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_step[int(row["step_index"])].append(row)
    dg_step_sd = []
    dz_step_sd = []
    dg_step_iqr = []
    dz_step_iqr = []
    for step_index in range(int(curves["step_count"])):
        step_rows = by_step[step_index]
        dg = np.asarray(
            [row["guidance_diversity"] for row in step_rows],
            dtype=np.float64,
        )
        dz = np.asarray(
            [
                row["predicted_clean_latent_diversity"]
                for row in step_rows
            ],
            dtype=np.float64,
        )
        dg_step_sd.append(float(dg.std(ddof=1)))
        dz_step_sd.append(float(dz.std(ddof=1)))
        dg_step_iqr.append(float(np.quantile(dg, 0.75) - np.quantile(dg, 0.25)))
        dz_step_iqr.append(float(np.quantile(dz, 0.75) - np.quantile(dz, 0.25)))

    active_steps = sorted(
        {
            int(row["step_index"])
            for row in records
            if float(row["pollution_cap"]) > 0.0
        }
    )
    dz_active_steps = sorted(
        {
            int(row["step_index"])
            for row in records
            if float(row["pollution_cap"]) > 0.0
            and bool(row["dz_control_eligible"])
        }
    )
    post_hard_off_steps = sorted(set(range(50)) - set(active_steps))
    group_sizes = sorted({int(row["group_size"]) for row in records})
    pool_sizes = sorted({int(row["pool_size"]) for row in records})
    signal_values = [
        float(row[key])
        for row in records
        for key in (
            "guidance_diversity",
            "predicted_clean_latent_diversity",
        )
    ]
    signals_finite_and_bounded = bool(
        np.isfinite(signal_values).all()
        and min(signal_values) >= 0.0
        and max(signal_values) <= 1.0
    )
    prompt_variation_present = bool(
        min(dg_step_sd) > 0.0
        and min(dz_step_sd) > 0.0
        and min(dg_step_iqr) > 0.0
        and min(dz_step_iqr) > 0.0
    )
    reference_hashes_match = bool(
        sha256(per_step_path) == validation["per_step_sha256"]
        and sha256(curve_path) == validation["reference_curves_sha256"]
    )
    exact_hard_off = bool(
        post_hard_off_steps == list(range(30, 50))
        and all(
            float(row["pollution"]) == 0.0
            for row in records
            if int(row["step_index"]) in post_hard_off_steps
        )
    )
    correct_dz_window = dz_active_steps == list(range(10, 30))
    b_stage1_ready = bool(
        validation["passed"]
        and validation["all_frozen_images_match"]
        and validation["extra_unet_calls_for_signals"] == 0
        and exact_median_reproduction
        and reference_hashes_match
        and signals_finite_and_bounded
        and prompt_variation_present
        and exact_hard_off
        and correct_dz_window
        and group_sizes == [8]
        and pool_sizes == [8]
        and not curves["formal_test_data_used"]
    )
    report = {
        "stage": "B-step-1",
        "scope": "Dg and Dz implementation and audit only; no controller",
        "development_split": "COCO-Dev-50",
        "formal_test_data_used": False,
        "prompt_count": len(prompt_ids),
        "step_count": int(curves["step_count"]),
        "record_count": len(records),
        "group_sizes": group_sizes,
        "pool_sizes": pool_sizes,
        "signal_definition": "mean unique-pair (1-cosine)/2 after FP32 adaptive 8x8 pooling",
        "guidance_diversity": distribution(
            [float(row["guidance_diversity"]) for row in records]
        ),
        "predicted_clean_latent_diversity": distribution(
            [
                float(row["predicted_clean_latent_diversity"])
                for row in records
            ]
        ),
        "minimum_stepwise_prompt_sd": {
            "Dg": min(dg_step_sd),
            "Dz": min(dz_step_sd),
        },
        "minimum_stepwise_prompt_iqr": {
            "Dg": min(dg_step_iqr),
            "Dz": min(dz_step_iqr),
        },
        "active_control_steps": active_steps,
        "dz_active_control_steps": dz_active_steps,
        "post_hard_off_steps": post_hard_off_steps,
        "checks": {
            "signals_finite_and_bounded": signals_finite_and_bounded,
            "prompt_variation_present_at_every_step": prompt_variation_present,
            "groups_do_not_cross_prompts": group_sizes == [8],
            "pooling_is_frozen_8x8": pool_sizes == [8],
            "exact_prompt_median_reproduction": exact_median_reproduction,
            "reference_hashes_match": reference_hashes_match,
            "correct_dz_control_window": correct_dz_window,
            "exact_hard_off": exact_hard_off,
            "all_400_frozen_images_match": validation[
                "all_frozen_images_match"
            ],
            "extra_unet_calls_are_zero": (
                validation["extra_unet_calls_for_signals"] == 0
            ),
            "formal_test_data_unused": not curves["formal_test_data_used"],
        },
        "b_stage1_ready": b_stage1_ready,
        "next_step": (
            "Implement the group-level proportional controller with one-step delay."
            if b_stage1_ready
            else "Resolve failed signal checks before implementing the controller."
        ),
    }
    if not b_stage1_ready:
        raise RuntimeError(json.dumps(report["checks"], indent=2))
    write_json(args.output_dir.resolve() / "validation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
