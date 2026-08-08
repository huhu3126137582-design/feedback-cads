#!/usr/bin/env python3
"""Independently reproduce formal aggregation and acceptance decisions."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "_steps"))

from evaluate_formal_coco_test500 import (  # noqa: E402
    CONFIG_SHA256,
    _paired_effect,
    load_and_validate,
    sha256,
)
from feedback_cads.experiments import (  # noqa: E402
    mean_pairwise_cosine_distance,
    read_jsonl,
    write_json,
)


def _strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-standard JSON constant {value} in {path}")
        ),
    )


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def audit() -> dict[str, Any]:
    config = load_and_validate()
    root = (PROJECT_ROOT / config["runtime"]["final_directory"]).resolve()
    completion = _strict_json(root / "evaluation_complete.json")
    summary = _strict_json(root / "summary.json")
    paired = _strict_json(root / "paired_bootstrap.json")
    per_image = read_jsonl(root / "per_image.jsonl")
    per_prompt = read_jsonl(root / "per_prompt.jsonl")
    manifest = config["_records"]
    fingerprint = config["_fingerprint"]
    if (
        completion["evaluation_fingerprint"] != fingerprint
        or completion["formal_evaluation_config_sha256"] != CONFIG_SHA256
        or completion["image_count"] != 16000
        or completion["prompt_group_count"] != 2000
        or completion["all_metric_stages_complete_before_reporting"] is not True
    ):
        raise RuntimeError("Formal evaluation completion metadata is invalid.")
    if len(per_image) != 16000 or len(per_prompt) != 2000:
        raise RuntimeError("Formal metric row counts are invalid.")

    for metric_row, manifest_row in zip(per_image, manifest):
        identity = ("method", "prompt_source_index", "candidate_id", "image_sha256")
        if any(metric_row[key] != manifest_row[key] for key in identity):
            raise RuntimeError("Per-image metrics lost manifest ordering.")
        if not all(
            math.isfinite(float(metric_row[key]))
            for key in ("clip_alignment_cosine", "clipscore", "hpsv2")
        ):
            raise RuntimeError("A per-image metric is non-finite.")

    cache = root / "feature_cache"
    arrays = {}
    for name in (
        "dino_features",
        "clip_features",
        "clip_alignment",
        "clipscores",
        "hps_scores",
    ):
        path = cache / f"{name}.npy"
        marker = _strict_json(cache / f"{name}_complete.json")
        value = np.load(path, allow_pickle=False)
        if (
            marker["evaluation_fingerprint"] != fingerprint
            or marker["record_count"] != 16000
            or marker["array_sha256"] != sha256(path)
            or len(value) != 16000
            or not np.isfinite(value).all()
        ):
            raise RuntimeError(f"Feature cache validation failed: {name}")
        arrays[name] = value

    prompt_lookup = {
        (row["method"], int(row["prompt_source_index"])): row
        for row in per_prompt
    }
    if len(prompt_lookup) != 2000:
        raise RuntimeError("Duplicate per-prompt metric rows.")
    group_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(manifest):
        group_indices[(row["method"], int(row["prompt_source_index"]))].append(index)
    maximum_error = {
        "dino_diversity": 0.0,
        "clip_image_diversity": 0.0,
        "clip_alignment_cosine": 0.0,
        "clipscore": 0.0,
        "hpsv2": 0.0,
    }
    for key, indices in group_indices.items():
        indices.sort(key=lambda index: int(manifest[index]["candidate_id"]))
        row = prompt_lookup[key]
        recomputed = {
            "dino_diversity": mean_pairwise_cosine_distance(
                arrays["dino_features"][indices]
            ),
            "clip_image_diversity": mean_pairwise_cosine_distance(
                arrays["clip_features"][indices]
            ),
            "clip_alignment_cosine": float(arrays["clip_alignment"][indices].mean()),
            "clipscore": float(arrays["clipscores"][indices].mean()),
            "hpsv2": float(arrays["hps_scores"][indices].mean()),
        }
        for metric, value in recomputed.items():
            error = abs(float(row[metric]) - value)
            maximum_error[metric] = max(maximum_error[metric], error)
            if error > 1e-12:
                raise RuntimeError(f"Per-prompt recomputation failed: {key}/{metric}")

    method_order = config["generation"]["method_order"]
    summary_lookup = {row["method"]: row for row in summary}
    if list(summary_lookup) != method_order:
        raise RuntimeError("Summary method order changed.")
    core_metrics = config["statistics"]["core_metrics"]
    for method in method_order:
        rows = [row for row in per_prompt if row["method"] == method]
        if len(rows) != 500:
            raise RuntimeError(f"Expected 500 prompt rows for {method}.")
        for metric in (
            "dino_diversity",
            "clip_image_diversity",
            "clip_alignment_cosine",
            "clipscore",
            "hpsv2",
            "prompt_run_seconds",
            "unet_calls",
        ):
            mean = float(np.mean([float(row[metric]) for row in rows]))
            if not _close(mean, summary_lookup[method][metric]):
                raise RuntimeError(f"Summary recomputation failed: {method}/{metric}")

    comparison_specs = {
        "b_feedback_vs_a_star": ("b_feedback", "a_star"),
        "original_cads_vs_vanilla": ("original_cads", "vanilla"),
        "a_star_vs_vanilla": ("a_star", "vanilla"),
        "b_feedback_vs_vanilla": ("b_feedback", "vanilla"),
    }
    replicates = int(config["statistics"]["bootstrap_replicates"])
    seed = int(config["statistics"]["bootstrap_seed"])
    for name, (method, comparator) in comparison_specs.items():
        method_rows = sorted(
            (row for row in per_prompt if row["method"] == method),
            key=lambda row: int(row["prompt_source_index"]),
        )
        comparator_rows = sorted(
            (row for row in per_prompt if row["method"] == comparator),
            key=lambda row: int(row["prompt_source_index"]),
        )
        for metric in core_metrics:
            method_values = np.asarray([row[metric] for row in method_rows], dtype=np.float64)
            comparator_values = np.asarray(
                [row[metric] for row in comparator_rows], dtype=np.float64
            )
            effect = _paired_effect(
                method_values,
                comparator_values,
                replicates=replicates,
                seed=seed,
            )
            recorded = paired["comparisons"][name]["metrics"][metric]
            for field in (
                "mean_difference",
                "median_difference",
                "ci95_lower",
                "ci95_upper",
                "paired_difference_sd",
            ):
                if not _close(effect[field], recorded[field]):
                    raise RuntimeError(f"Bootstrap reproduction failed: {name}/{metric}/{field}")
            margin = 0.01 * float(comparator_values.mean())
            expected_flags = {
                "noninferior": effect["ci95_lower"] > -margin,
                "point_estimate_improved": effect["mean_difference"] > 0.0,
                "strong_improvement": effect["ci95_lower"] > 0.0,
            }
            if not _close(recorded["noninferiority_margin"], margin):
                raise RuntimeError(f"NI margin reproduction failed: {name}/{metric}")
            if any(recorded[field] is not value for field, value in expected_flags.items()):
                raise RuntimeError(f"Decision reproduction failed: {name}/{metric}")

    primary = paired["comparisons"]["b_feedback_vs_a_star"]["metrics"]
    all_ni = all(primary[metric]["noninferior"] for metric in core_metrics)
    point = [metric for metric in core_metrics if primary[metric]["point_estimate_improved"]]
    strong = [metric for metric in core_metrics if primary[metric]["strong_improvement"]]
    expected_acceptance = {
        "all_three_core_metrics_noninferior": all_ni,
        "point_estimate_improvements": point,
        "strong_improvements": strong,
        "assessment_success": all_ni and bool(point),
        "closed_loop_diversity_success": bool(
            primary["dino_diversity"]["point_estimate_improved"]
            and primary["clipscore"]["noninferior"]
            and primary["hpsv2"]["noninferior"]
        ),
        "strong_research_result": all_ni and bool(strong),
    }
    recorded_acceptance = paired["primary_acceptance"]
    if any(recorded_acceptance[key] != value for key, value in expected_acceptance.items()):
        raise RuntimeError("Primary acceptance logic reproduction failed.")

    artifact_names = (
        "per_image.jsonl",
        "per_prompt.jsonl",
        "summary.csv",
        "summary.json",
        "paired_bootstrap.json",
        "evaluation_complete.json",
    )
    report = {
        "passed": True,
        "evaluation_fingerprint": fingerprint,
        "formal_evaluation_config_sha256": CONFIG_SHA256,
        "independently_verified_images": len(per_image),
        "independently_verified_prompt_groups": len(per_prompt),
        "feature_cache_arrays_verified": 5,
        "maximum_per_prompt_recomputation_error": maximum_error,
        "bootstrap_replicates_reproduced": replicates,
        "all_four_comparisons_reproduced": True,
        "all_noninferiority_margins_reproduced": True,
        "primary_acceptance_reproduced": True,
        "primary_acceptance": recorded_acceptance,
        "artifact_sha256": {name: sha256(root / name) for name in artifact_names},
    }
    write_json(root / "evaluation_audit.json", report)
    return report


def main() -> None:
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
