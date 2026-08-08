#!/usr/bin/env python3
"""Strict prompt-paired evaluation of Stage B against frozen A*."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "_steps"))

from evaluate_a_strength_curve import (  # noqa: E402
    compute_hps_scores,
    extract_clip_metrics,
    extract_dino_features,
)
from feedback_cads.experiments import (  # noqa: E402
    mean_pairwise_cosine_distance,
    paired_bootstrap_mean_difference,
    read_jsonl,
    resolve_recorded_path,
    write_json,
    write_jsonl,
)
from feedback_cads.configuration import load_yaml_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/b_feedback_search_coco_dev50.yaml",
    )
    parser.add_argument(
        "--profile",
        choices=(
            "alpha_0p00",
            "alpha_0p05",
            "alpha_0p10",
            "alpha_0p15",
            "alpha_0p20",
        ),
        default="alpha_0p00",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_and_pair_manifests(
    config: dict[str, Any],
    experiment_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    b_complete = json.loads(
        (experiment_dir / "generation_complete.json").read_text(
            encoding="utf-8"
        )
    )
    if b_complete["formal_test_data_used"]:
        raise ValueError("B generation unexpectedly used formal test data.")
    if int(b_complete["prompt_count"]) != 50:
        raise ValueError("The paired comparison requires all 50 prompts.")
    if not b_complete["all_latent_and_condition_seeds_paired_with_a_star"]:
        raise ValueError("B generation did not certify seed pairing.")

    b_manifest_path = experiment_dir / "manifest.jsonl"
    b_sha = sha256(b_manifest_path)
    if b_sha != b_complete["manifest_sha256"]:
        raise ValueError("B manifest hash does not match completion metadata.")
    b_records = read_jsonl(b_manifest_path)
    if len(b_records) != 400:
        raise ValueError("Expected 400 B images.")

    a_manifest_path = PROJECT_ROOT / config["paired_a_star"]["manifest"]
    a_sha = sha256(a_manifest_path)
    if a_sha != config["paired_a_star"]["manifest_sha256"]:
        raise ValueError("Frozen A manifest SHA-256 changed.")
    a_records = [
        row
        for row in read_jsonl(a_manifest_path)
        if float(row["rho"]) == float(config["paired_a_star"]["rho"])
    ]
    if len(a_records) != 400:
        raise ValueError("Expected 400 frozen A* images.")

    a_by_pair = {
        (str(row["prompt_id"]), int(row["candidate_id"])): row
        for row in a_records
    }
    b_by_pair = {
        (str(row["prompt_id"]), int(row["candidate_id"])): row
        for row in b_records
    }
    if set(a_by_pair) != set(b_by_pair):
        raise ValueError("A and B do not contain identical prompt-candidate pairs.")
    for key in sorted(a_by_pair):
        a_row = a_by_pair[key]
        b_row = b_by_pair[key]
        if int(a_row["latent_seed"]) != int(b_row["latent_seed"]):
            raise ValueError(f"Latent seed mismatch for {key}.")
        if int(a_row["condition_seed"]) != int(b_row["condition_seed"]):
            raise ValueError(f"Condition seed mismatch for {key}.")
        if a_row["prompt"] != b_row["prompt"]:
            raise ValueError(f"Prompt mismatch for {key}.")
        for row in (a_row, b_row):
            image_path = resolve_recorded_path(
                row["image_path"], project_root=PROJECT_ROOT
            )
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if sha256(image_path) != row["image_sha256"]:
                raise ValueError(f"Image hash mismatch: {image_path}")

    a_records.sort(key=lambda row: (row["prompt_id"], row["candidate_id"]))
    b_records.sort(key=lambda row: (row["prompt_id"], row["candidate_id"]))
    return a_records, b_records, a_sha, b_sha


def paired_effect(
    method: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | None]:
    result = paired_bootstrap_mean_difference(
        method,
        baseline,
        replicates=replicates,
        seed=seed,
    )
    difference = np.asarray(method, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    standard_deviation = float(difference.std(ddof=1))
    result["paired_difference_sd"] = standard_deviation
    result["standardized_paired_effect"] = (
        None
        if standard_deviation == 0.0
        else float(difference.mean() / standard_deviation)
    )
    return result


def main() -> None:
    args = parse_args()
    config, _identity = load_yaml_profile(
        args.config,
        profile=args.profile,
    )
    experiment_dir = (
        args.experiment_dir
        if args.experiment_dir is not None
        else PROJECT_ROOT / "outputs" / config["experiment"]["name"]
    ).resolve()
    if config["experiment"]["formal_test_data_used"] is not False:
        raise ValueError("This evaluation must be development-only.")
    if config["dataset"]["split"] != "COCO-Dev-50":
        raise ValueError("This evaluation must use COCO-Dev-50.")
    a_records, b_records, a_manifest_sha, b_manifest_sha = (
        validate_and_pair_manifests(config, experiment_dir)
    )

    records = []
    for method, source in (("A_star", a_records), ("B_feedback", b_records)):
        for row in source:
            records.append({**row, "method": method})
    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metric_config_path = PROJECT_ROOT / config["evaluation"]["metric_config"]
    metric_protocol = yaml.safe_load(
        metric_config_path.read_text(encoding="utf-8")
    )
    fingerprint = hashlib.sha256(
        (
            a_manifest_sha
            + b_manifest_sha
            + sha256(metric_config_path)
        ).encode("ascii")
    ).hexdigest()
    cache_path = metrics_dir / "paired_feature_cache.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        if str(cached["fingerprint"].item()) != fingerprint:
            raise RuntimeError("Paired metric cache has a stale fingerprint.")
        dino_features = cached["dino_features"]
        clip_features = cached["clip_features"]
        clip_alignment = cached["clip_alignment"]
        clipscores = cached["clipscores"]
        hps_scores = cached["hps_scores"]
    else:
        cache_dir = PROJECT_ROOT / "models/huggingface/hub"
        dino_features = extract_dino_features(
            records,
            model_name=metric_protocol["dino_model"],
            batch_size=args.batch_size,
            cache_dir=cache_dir,
            revision=metric_protocol["dino_revision"],
        )
        clip_features, clip_alignment, clipscores = extract_clip_metrics(
            records,
            model_name=metric_protocol["clip_model"],
            batch_size=args.batch_size,
            cache_dir=cache_dir,
            clipscore_weight=float(metric_protocol["clipscore_weight"]),
            revision=metric_protocol["clip_revision"],
            subfolder=metric_protocol["clip_subfolder"],
        )
        hps_scores = compute_hps_scores(
            {},
            records,
            version=metric_protocol["hps_version"],
            batch_size=min(args.batch_size, 16),
        )
        np.savez_compressed(
            cache_path,
            fingerprint=np.asarray(fingerprint),
            dino_features=dino_features,
            clip_features=clip_features,
            clip_alignment=clip_alignment,
            clipscores=clipscores,
            hps_scores=hps_scores,
        )

    if not all(
        len(value) == len(records)
        for value in (
            dino_features,
            clip_features,
            clip_alignment,
            clipscores,
            hps_scores,
        )
    ):
        raise RuntimeError("Metric feature count does not match image count.")
    if not all(
        np.isfinite(value).all()
        for value in (
            dino_features,
            clip_features,
            clip_alignment,
            clipscores,
            hps_scores,
        )
    ):
        raise RuntimeError("A paired metric contains NaN or Inf.")

    per_image = []
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[(row["method"], row["prompt_id"])].append(index)
        per_image.append(
            {
                **row,
                "clip_alignment_cosine": float(clip_alignment[index]),
                "clipscore": float(clipscores[index]),
                "hpsv2": float(hps_scores[index]),
            }
        )
    write_jsonl(metrics_dir / "per_image.jsonl", per_image)

    controller_by_prompt = {
        row["prompt_id"]: row
        for row in read_jsonl(experiment_dir / "per_prompt_controller.jsonl")
    }
    per_prompt = []
    for (method, prompt_id), indices in sorted(groups.items()):
        indices.sort(key=lambda index: int(records[index]["candidate_id"]))
        first = records[indices[0]]
        if len(indices) != 8:
            raise RuntimeError(f"Group {(method, prompt_id)} does not have K=8.")
        if method == "B_feedback":
            control = controller_by_prompt[prompt_id]
            rho_mean = float(control["active_rho_mean"])
            rho_variance = float(control["active_rho_variance"])
            rho_boundary_rate = float(control["active_rho_boundary_rate"])
            unet_calls = 50
        else:
            rho_mean = 0.55
            rho_variance = 0.0
            rho_boundary_rate = 0.0
            unet_calls = 50
        per_prompt.append(
            {
                "method": method,
                "prompt_id": prompt_id,
                "prompt": first["prompt"],
                "candidate_count": len(indices),
                "dino_diversity": mean_pairwise_cosine_distance(
                    dino_features[indices]
                ),
                "clip_image_diversity": mean_pairwise_cosine_distance(
                    clip_features[indices]
                ),
                "clip_alignment_cosine": float(
                    clip_alignment[indices].mean()
                ),
                "clipscore": float(clipscores[indices].mean()),
                "hpsv2": float(hps_scores[indices].mean()),
                "prompt_run_seconds": float(first["prompt_run_seconds"]),
                "rho_mean_active_window": rho_mean,
                "rho_variance_active_window": rho_variance,
                "rho_boundary_rate_active_window": rho_boundary_rate,
                "unet_calls": unet_calls,
            }
        )
    write_jsonl(metrics_dir / "per_prompt.jsonl", per_prompt)
    frame = pd.DataFrame(per_prompt)
    metric_columns = [
        "dino_diversity",
        "clip_image_diversity",
        "clip_alignment_cosine",
        "clipscore",
        "hpsv2",
        "prompt_run_seconds",
        "rho_mean_active_window",
        "rho_variance_active_window",
        "rho_boundary_rate_active_window",
        "unet_calls",
    ]
    summary = frame.groupby("method", as_index=False)[metric_columns].mean()
    summary.to_csv(metrics_dir / "summary.csv", index=False)

    frozen_metrics_path = (
        PROJECT_ROOT / config["paired_a_star"]["per_prompt_metrics"]
    )
    if sha256(frozen_metrics_path) != config["paired_a_star"][
        "per_prompt_metrics_sha256"
    ]:
        raise ValueError("Frozen A* per-prompt metrics changed.")
    frozen_a = pd.DataFrame(
        [
            row
            for row in read_jsonl(frozen_metrics_path)
            if float(row["rho"]) == float(config["paired_a_star"]["rho"])
        ]
    ).sort_values("prompt_id")
    recomputed_a = frame[frame["method"] == "A_star"].sort_values(
        "prompt_id"
    )
    if frozen_a["prompt_id"].tolist() != recomputed_a["prompt_id"].tolist():
        raise RuntimeError("Frozen and recomputed A* prompts differ.")
    reproduction_errors = {
        metric: float(
            np.max(
                np.abs(
                    frozen_a[metric].to_numpy(dtype=np.float64)
                    - recomputed_a[metric].to_numpy(dtype=np.float64)
                )
            )
        )
        for metric in ("dino_diversity", "clipscore", "hpsv2")
    }
    frozen_a_reproduced = all(
        error <= 1e-6 for error in reproduction_errors.values()
    )
    if not frozen_a_reproduced:
        raise RuntimeError(
            f"Recomputed A* metrics changed: {reproduction_errors}"
        )

    a_frame = frame[frame["method"] == "A_star"].sort_values("prompt_id")
    b_frame = frame[frame["method"] == "B_feedback"].sort_values(
        "prompt_id"
    )
    if a_frame["prompt_id"].tolist() != b_frame["prompt_id"].tolist():
        raise RuntimeError("Prompt pairing was lost before bootstrap.")
    comparisons = {}
    replicates = int(config["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    for metric in ("dino_diversity", "clipscore", "hpsv2"):
        comparisons[metric] = paired_effect(
            b_frame[metric].to_numpy(dtype=np.float64),
            a_frame[metric].to_numpy(dtype=np.float64),
            replicates=replicates,
            seed=bootstrap_seed,
        )

    selection_path = PROJECT_ROOT / config["paired_a_star"]["selection"]
    if sha256(selection_path) != config["paired_a_star"]["selection_sha256"]:
        raise ValueError("Frozen A* selection SHA-256 changed.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    clip_margin = float(selection["clipscore_noninferiority_margin"])
    hps_margin = float(selection["hpsv2_noninferiority_margin"])
    clip_noninferior = (
        comparisons["clipscore"]["ci95_lower"] > -clip_margin
    )
    hps_noninferior = comparisons["hpsv2"]["ci95_lower"] > -hps_margin
    core_point_improvements = [
        metric
        for metric, result in comparisons.items()
        if result["mean_difference"] > 0.0
    ]
    core_strong_improvements = [
        metric
        for metric, result in comparisons.items()
        if result["ci95_lower"] > 0.0
    ]
    all_core_ci_cross_zero = all(
        result["ci95_lower"] <= 0.0 <= result["ci95_upper"]
        for result in comparisons.values()
    )
    b_rho = b_frame["rho_mean_active_window"].to_numpy(dtype=np.float64)
    b_boundary = b_frame[
        "rho_boundary_rate_active_window"
    ].to_numpy(dtype=np.float64)
    adaptive_trajectories = float(b_rho.std(ddof=1)) > 0.0
    no_long_saturation = float(b_boundary.mean()) < 0.25

    paired_report = {
        "comparison": "B_feedback minus A_star",
        "development_split": "COCO-Dev-50",
        "statistical_unit": "prompt",
        "prompt_count": 50,
        "candidates_per_prompt": 8,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": bootstrap_seed,
        "comparisons": comparisons,
        "quality_noninferiority": {
            "clipscore_margin": clip_margin,
            "clipscore_passed": bool(clip_noninferior),
            "hpsv2_margin": hps_margin,
            "hpsv2_passed": bool(hps_noninferior),
            "both_passed": bool(clip_noninferior and hps_noninferior),
        },
        "core_point_improvements": core_point_improvements,
        "core_strong_improvements": core_strong_improvements,
        "closed_loop_point_contribution_observed": bool(
            core_point_improvements
        ),
        "closed_loop_strong_contribution_observed": bool(
            core_strong_improvements
        ),
        "controller_behavior": {
            "mean_active_rho_across_prompts": float(b_rho.mean()),
            "sd_prompt_mean_active_rho": float(b_rho.std(ddof=1)),
            "mean_active_boundary_rate": float(b_boundary.mean()),
            "adaptive_prompt_trajectories_present": adaptive_trajectories,
            "no_long_saturation_under_25pct_diagnostic": no_long_saturation,
        },
        "alpha_margin_ablation_conditions": {
            "all_core_intervals_include_zero": all_core_ci_cross_zero,
            "quality_noninferiority_passed": bool(
                clip_noninferior and hps_noninferior
            ),
            "adaptive_prompt_trajectories_present": adaptive_trajectories,
            "no_long_saturation_under_25pct_diagnostic": no_long_saturation,
            "eligible_for_small_alpha_ablation": bool(
                all_core_ci_cross_zero
                and clip_noninferior
                and hps_noninferior
                and adaptive_trajectories
                and no_long_saturation
            ),
        },
        "frozen_a_metric_reproduction": {
            "passed": frozen_a_reproduced,
            "maximum_absolute_error": reproduction_errors,
        },
        "unet_calls_per_prompt": {"A_star": 50, "B_feedback": 50},
        "formal_test_data_used": False,
    }
    write_json(metrics_dir / "paired_bootstrap.json", paired_report)
    write_json(
        metrics_dir / "evaluation_complete.json",
        {
            "a_manifest_sha256": a_manifest_sha,
            "b_manifest_sha256": b_manifest_sha,
            "metric_protocol_sha256": sha256(metric_config_path),
            "paired_fingerprint": fingerprint,
            "image_count": len(records),
            "prompt_group_count": len(per_prompt),
            "methods": ["A_star", "B_feedback"],
            "dino_model": metric_protocol["dino_model"],
            "dino_revision": metric_protocol["dino_revision"],
            "clip_model": metric_protocol["clip_model"],
            "clip_revision": metric_protocol["clip_revision"],
            "hps_version": metric_protocol["hps_version"],
            "safe_weight_loading": True,
            "all_prompt_candidate_pairs_share_latents_and_condition_seed": True,
            "frozen_a_metric_reproduction_passed": frozen_a_reproduced,
            "formal_test_data_used": False,
        },
    )
    print(summary.to_string(index=False))
    print(json.dumps(paired_report, indent=2))


if __name__ == "__main__":
    main()
