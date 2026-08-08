#!/usr/bin/env python3
"""Evaluate all frozen formal methods before revealing any metric values."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
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
from validate_formal_test_protocol import validate_protocol  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/formal_artifacts_coco_test500.yaml"
CONFIG_PROFILE = "evaluation"
CONFIG_SHA256 = (
    "645db8fc998b0986fe818b4a3f46247518553af532859ab5594f6523e37d25a1"
)
FORMAL_PROTOCOL_SHA256 = (
    "9d19e02e284aefe0ed58fdc5a47dd0a0fec0966936a40dd9e281d2c97d0161ef"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _checked_path(path_value: str, expected_sha256: str) -> Path:
    path = (PROJECT_ROOT / path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != expected_sha256:
        raise ValueError(f"Frozen evaluation artifact changed: {path}")
    return path


def load_and_validate() -> dict[str, Any]:
    config, identity = load_yaml_profile(
        CONFIG_PATH,
        profile=CONFIG_PROFILE,
    )
    if identity["legacy_sha256"] != CONFIG_SHA256:
        raise ValueError("The frozen formal evaluation config SHA-256 changed.")
    evaluation = config["evaluation"]
    required_flags = {
        "revision": 1,
        "frozen_before_any_formal_metric_calculation": True,
        "formal_metrics_computed_at_freeze": False,
        "evaluate_all_methods_before_reporting": True,
        "no_subset_evaluation": True,
        "no_metric_skipping": True,
    }
    for key, expected in required_flags.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"Formal evaluation freeze flag changed: {key}")

    protocol_section = config["formal_protocol"]
    protocol_path = _checked_path(
        protocol_section["path"], protocol_section["sha256"]
    )
    if protocol_section["sha256"] != FORMAL_PROTOCOL_SHA256:
        raise ValueError("Formal generation protocol hash changed.")
    protocol_report = validate_protocol(protocol_path, require_unstarted=False)
    if protocol_report["expected_total_images"] != 16_000:
        raise ValueError("Formal protocol image count changed.")

    generation = config["generation"]
    manifest_path = _checked_path(
        generation["manifest"], generation["manifest_sha256"]
    )
    completion_path = _checked_path(
        generation["completion"], generation["completion_sha256"]
    )
    audit_path = _checked_path(
        generation["integrity_audit"], generation["integrity_audit_sha256"]
    )
    completion = _load_json(completion_path)
    audit = _load_json(audit_path)
    if (
        completion.get("image_count") != 16_000
        or completion.get("quality_metrics_computed") is not False
        or completion.get("manifest_sha256") != generation["manifest_sha256"]
        or audit.get("passed") is not True
        or audit.get("quality_metrics_computed") is not False
        or audit.get("formal_metric_files_seen") is not False
    ):
        raise ValueError("Formal generation was not cleanly completed.")

    records = read_jsonl(manifest_path)
    method_order = list(generation["method_order"])
    if method_order != ["vanilla", "original_cads", "a_star", "b_feedback"]:
        raise ValueError("Formal method order changed.")
    if len(records) != int(generation["expected_images"]):
        raise ValueError("Formal manifest image count changed.")
    if Counter(row["method"] for row in records) != Counter(
        {method: 4000 for method in method_order}
    ):
        raise ValueError("Formal manifest method counts changed.")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["method"]), int(row["prompt_source_index"]))].append(row)
    if len(groups) != 2000 or any(len(rows) != 8 for rows in groups.values()):
        raise ValueError("Formal prompt groups are incomplete.")
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["candidate_id"]))
        if [int(row["candidate_id"]) for row in rows] != list(range(8)):
            raise ValueError("A formal group has invalid candidate IDs.")

    metric_section = config["metric_protocol"]
    metric_path = _checked_path(metric_section["path"], metric_section["sha256"])
    metric = yaml.safe_load(metric_path.read_text(encoding="utf-8"))
    statistics = config["statistics"]
    if (
        int(metric["bootstrap_replicates"])
        != int(statistics["bootstrap_replicates"])
        or int(metric["bootstrap_seed"]) != int(statistics["bootstrap_seed"])
        or statistics["primary_comparison"] != "b_feedback_vs_a_star"
        or statistics["core_metrics"]
        != ["dino_diversity", "clipscore", "hpsv2"]
    ):
        raise ValueError("Formal metric/statistical protocol changed.")

    weight_hashes = {}
    for name, section in config["metric_weights"].items():
        path = _checked_path(section["path"], section["sha256"])
        weight_hashes[name] = {"path": str(path), "sha256": section["sha256"]}
    runtime = config["runtime"]
    if (
        runtime["device"] != "cuda"
        or runtime["offline_model_loading"] is not True
        or int(runtime["dino_batch_size"]) != 32
        or int(runtime["clip_batch_size"]) != 32
        or int(runtime["hpsv2_batch_size"]) != 16
        or runtime["do_not_print_metric_values_until_complete"] is not True
    ):
        raise ValueError("Formal metric runtime protocol changed.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal metric evaluation.")

    fingerprint = hashlib.sha256(
        (
            CONFIG_SHA256
            + generation["manifest_sha256"]
            + "".join(weight_hashes[name]["sha256"] for name in sorted(weight_hashes))
        ).encode("ascii")
    ).hexdigest()
    config["_records"] = records
    config["_groups"] = groups
    config["_metric"] = metric
    config["_fingerprint"] = fingerprint
    config["_manifest_path"] = str(manifest_path)
    config["_metric_path"] = str(metric_path)
    config["_weight_hashes"] = weight_hashes
    return config


def _save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _load_completed_array(
    cache_dir: Path,
    name: str,
    *,
    fingerprint: str,
    expected_count: int,
) -> np.ndarray | None:
    array_path = cache_dir / f"{name}.npy"
    marker_path = cache_dir / f"{name}_complete.json"
    if not array_path.exists() and not marker_path.exists():
        return None
    if not array_path.is_file() or not marker_path.is_file():
        raise RuntimeError(f"Incomplete metric cache stage: {name}")
    marker = _load_json(marker_path)
    if (
        marker.get("evaluation_fingerprint") != fingerprint
        or marker.get("record_count") != expected_count
        or marker.get("array_sha256") != sha256(array_path)
    ):
        raise RuntimeError(f"Stale metric cache stage: {name}")
    value = np.load(array_path, allow_pickle=False)
    if len(value) != expected_count or not np.isfinite(value).all():
        raise RuntimeError(f"Invalid metric cache array: {name}")
    return value


def _complete_array(
    cache_dir: Path,
    name: str,
    value: np.ndarray,
    *,
    fingerprint: str,
) -> None:
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise RuntimeError(f"Metric stage produced NaN/Inf: {name}")
    array_path = cache_dir / f"{name}.npy"
    _save_npy(array_path, value)
    write_json(
        cache_dir / f"{name}_complete.json",
        {
            "evaluation_fingerprint": fingerprint,
            "record_count": len(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "array_sha256": sha256(array_path),
            "metric_values_reported": False,
        },
    )


def _paired_effect(
    method: np.ndarray,
    comparator: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = paired_bootstrap_mean_difference(
        method,
        comparator,
        replicates=replicates,
        seed=seed,
    )
    differences = np.asarray(method, dtype=np.float64) - np.asarray(
        comparator, dtype=np.float64
    )
    difference_sd = float(differences.std(ddof=1))
    result["paired_difference_sd"] = difference_sd
    result["standardized_paired_effect"] = (
        None if difference_sd == 0.0 else float(differences.mean() / difference_sd)
    )
    return result


def _controller_summary(diagnostics_path: str) -> tuple[float, float, float]:
    diagnostics = _load_json(
        resolve_recorded_path(diagnostics_path, project_root=PROJECT_ROOT)
    )
    history = diagnostics["pipeline_stats"]["feedback_control_history"]
    if len(history) != 50:
        raise RuntimeError("B formal controller history is incomplete.")
    active = np.asarray(
        [row["rho_used_by_prompt"][0] for row in history[:30]],
        dtype=np.float64,
    )
    return (
        float(active.mean()),
        float(active.var()),
        float(np.mean((active == 0.0) | (active == 1.0))),
    )


def _build_outputs(
    config: dict[str, Any],
    staging: Path,
    *,
    dino_features: np.ndarray,
    clip_features: np.ndarray,
    clip_alignment: np.ndarray,
    clipscores: np.ndarray,
    hps_scores: np.ndarray,
) -> dict[str, Any]:
    records = config["_records"]
    method_order = config["generation"]["method_order"]
    per_image = [
        {
            **row,
            "clip_alignment_cosine": float(clip_alignment[index]),
            "clipscore": float(clipscores[index]),
            "hpsv2": float(hps_scores[index]),
        }
        for index, row in enumerate(records)
    ]
    write_jsonl(staging / "per_image.jsonl", per_image)

    indices_by_group: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        indices_by_group[(row["method"], int(row["prompt_source_index"]))].append(index)
    per_prompt = []
    for method in method_order:
        for prompt_index in range(500):
            indices = sorted(
                indices_by_group[(method, prompt_index)],
                key=lambda index: int(records[index]["candidate_id"]),
            )
            if len(indices) != 8:
                raise RuntimeError(f"Incomplete group: {(method, prompt_index)}")
            first = records[indices[0]]
            rho_mean: float | None = None
            rho_variance: float | None = None
            rho_boundary: float | None = None
            if method == "a_star":
                rho_mean, rho_variance, rho_boundary = 0.55, 0.0, 0.0
            elif method == "b_feedback":
                rho_mean, rho_variance, rho_boundary = _controller_summary(
                    first["diagnostics_path"]
                )
            per_prompt.append(
                {
                    "method": method,
                    "prompt_source_index": prompt_index,
                    "prompt_id": first["prompt_id"],
                    "prompt": first["prompt"],
                    "candidate_count": 8,
                    "dino_diversity": mean_pairwise_cosine_distance(
                        dino_features[indices]
                    ),
                    "clip_image_diversity": mean_pairwise_cosine_distance(
                        clip_features[indices]
                    ),
                    "clip_alignment_cosine": float(clip_alignment[indices].mean()),
                    "clipscore": float(clipscores[indices].mean()),
                    "hpsv2": float(hps_scores[indices].mean()),
                    "prompt_run_seconds": float(first["prompt_run_seconds"]),
                    "rho_mean_active_window": rho_mean,
                    "rho_variance_active_window": rho_variance,
                    "rho_boundary_rate_active_window": rho_boundary,
                    "unet_calls": 50,
                }
            )
    write_jsonl(staging / "per_prompt.jsonl", per_prompt)
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
    summary = (
        frame.groupby("method", as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .set_index("method")
        .loc[method_order]
        .reset_index()
    )
    summary.to_csv(staging / "summary.csv", index=False)
    summary_records = (
        summary.astype(object)
        .where(pd.notna(summary), None)
        .to_dict(orient="records")
    )
    write_json(staging / "summary.json", summary_records)

    statistics = config["statistics"]
    replicates = int(statistics["bootstrap_replicates"])
    seed = int(statistics["bootstrap_seed"])
    comparisons_spec = [
        ("b_feedback_vs_a_star", "b_feedback", "a_star", True),
        ("original_cads_vs_vanilla", "original_cads", "vanilla", False),
        ("a_star_vs_vanilla", "a_star", "vanilla", False),
        ("b_feedback_vs_vanilla", "b_feedback", "vanilla", False),
    ]
    core_metrics = list(statistics["core_metrics"])
    comparisons: dict[str, Any] = {}
    for name, method, comparator, primary in comparisons_spec:
        method_frame = frame[frame["method"] == method].sort_values(
            "prompt_source_index"
        )
        comparator_frame = frame[frame["method"] == comparator].sort_values(
            "prompt_source_index"
        )
        if method_frame["prompt_id"].tolist() != comparator_frame["prompt_id"].tolist():
            raise RuntimeError(f"Prompt pairing was lost for {name}.")
        metrics: dict[str, Any] = {}
        for metric in core_metrics:
            method_values = method_frame[metric].to_numpy(dtype=np.float64)
            comparator_values = comparator_frame[metric].to_numpy(dtype=np.float64)
            effect = _paired_effect(
                method_values,
                comparator_values,
                replicates=replicates,
                seed=seed,
            )
            comparator_mean = float(comparator_values.mean())
            if not math.isfinite(comparator_mean) or comparator_mean <= 0.0:
                raise RuntimeError(f"Comparator mean is invalid for {name}/{metric}.")
            margin = 0.01 * comparator_mean
            metrics[metric] = {
                "method_mean": float(method_values.mean()),
                "comparator_mean": comparator_mean,
                **effect,
                "noninferiority_margin": margin,
                "noninferior": bool(effect["ci95_lower"] > -margin),
                "point_estimate_improved": bool(effect["mean_difference"] > 0.0),
                "strong_improvement": bool(effect["ci95_lower"] > 0.0),
            }
        comparisons[name] = {
            "method": method,
            "comparator": comparator,
            "primary": primary,
            "metrics": metrics,
        }

    primary_metrics = comparisons["b_feedback_vs_a_star"]["metrics"]
    all_noninferior = all(
        primary_metrics[metric]["noninferior"] for metric in core_metrics
    )
    point_improvements = [
        metric
        for metric in core_metrics
        if primary_metrics[metric]["point_estimate_improved"]
    ]
    strong_improvements = [
        metric for metric in core_metrics if primary_metrics[metric]["strong_improvement"]
    ]
    assessment_success = all_noninferior and bool(point_improvements)
    closed_loop_diversity_success = bool(
        primary_metrics["dino_diversity"]["point_estimate_improved"]
        and primary_metrics["clipscore"]["noninferior"]
        and primary_metrics["hpsv2"]["noninferior"]
    )
    strong_research_result = all_noninferior and bool(strong_improvements)
    if strong_research_result:
        label = "statistically_significant_strong_research_result"
    elif assessment_success:
        label = "assessment_requirement_met_but_statistical_evidence_limited"
    else:
        label = "preregistered_assessment_requirement_not_met"
    paired_report = {
        "evaluation_fingerprint": config["_fingerprint"],
        "formal_protocol_sha256": FORMAL_PROTOCOL_SHA256,
        "statistical_unit": "prompt",
        "prompt_count": 500,
        "candidates_per_prompt": 8,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "confidence_interval": 0.95,
        "comparisons": comparisons,
        "primary_acceptance": {
            "comparison": "b_feedback_vs_a_star",
            "all_three_core_metrics_noninferior": all_noninferior,
            "point_estimate_improvements": point_improvements,
            "strong_improvements": strong_improvements,
            "assessment_success": assessment_success,
            "closed_loop_diversity_success": closed_loop_diversity_success,
            "strong_research_result": strong_research_result,
            "result_label": label,
        },
        "secondary_comparisons_are_exploratory": True,
    }
    write_json(staging / "paired_bootstrap.json", paired_report)
    return paired_report


def evaluate(config: dict[str, Any]) -> Path:
    runtime = config["runtime"]
    staging = (PROJECT_ROOT / runtime["staging_directory"]).resolve()
    final = (PROJECT_ROOT / runtime["final_directory"]).resolve()
    if final.exists():
        complete = _load_json(final / "evaluation_complete.json")
        if complete.get("evaluation_fingerprint") != config["_fingerprint"]:
            raise RuntimeError("Final metrics belong to another evaluation fingerprint.")
        return final
    lock = {
        "formal_evaluation_config_sha256": CONFIG_SHA256,
        "formal_protocol_sha256": FORMAL_PROTOCOL_SHA256,
        "evaluation_fingerprint": config["_fingerprint"],
        "manifest_sha256": config["generation"]["manifest_sha256"],
        "metric_protocol_sha256": config["metric_protocol"]["sha256"],
        "metric_weight_hashes": config["_weight_hashes"],
        "expected_images": 16000,
        "expected_prompt_groups": 2000,
        "metric_values_reported": False,
    }
    lock_path = staging / "evaluation_lock.json"
    if staging.exists():
        if not lock_path.is_file() or _load_json(lock_path) != lock:
            raise RuntimeError("Staging metrics belong to another evaluation.")
    else:
        staging.mkdir(parents=True)
        write_json(lock_path, lock)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    records = config["_records"]
    metric = config["_metric"]
    fingerprint = config["_fingerprint"]
    cache = staging / "feature_cache"
    cache.mkdir(exist_ok=True)
    count = len(records)

    dino = _load_completed_array(
        cache, "dino_features", fingerprint=fingerprint, expected_count=count
    )
    if dino is None:
        print("Computing complete DINO feature stage (values hidden)...", flush=True)
        dino = extract_dino_features(
            records,
            model_name=metric["dino_model"],
            batch_size=int(runtime["dino_batch_size"]),
            cache_dir=PROJECT_ROOT / "models/huggingface/hub",
            revision=metric["dino_revision"],
        )
        _complete_array(cache, "dino_features", dino, fingerprint=fingerprint)
        print("DINO feature stage complete; values remain hidden.", flush=True)

    clip = _load_completed_array(
        cache, "clip_features", fingerprint=fingerprint, expected_count=count
    )
    clip_alignment = _load_completed_array(
        cache, "clip_alignment", fingerprint=fingerprint, expected_count=count
    )
    clipscores = _load_completed_array(
        cache, "clipscores", fingerprint=fingerprint, expected_count=count
    )
    if any(value is None for value in (clip, clip_alignment, clipscores)):
        if any(value is not None for value in (clip, clip_alignment, clipscores)):
            raise RuntimeError("Partial CLIP cache is not allowed.")
        print("Computing complete CLIP feature/alignment stage (values hidden)...", flush=True)
        clip, clip_alignment, clipscores = extract_clip_metrics(
            records,
            model_name=metric["clip_model"],
            batch_size=int(runtime["clip_batch_size"]),
            cache_dir=PROJECT_ROOT / "models/huggingface/hub",
            clipscore_weight=float(metric["clipscore_weight"]),
            revision=metric["clip_revision"],
            subfolder=metric["clip_subfolder"],
        )
        _complete_array(cache, "clip_features", clip, fingerprint=fingerprint)
        _complete_array(cache, "clip_alignment", clip_alignment, fingerprint=fingerprint)
        _complete_array(cache, "clipscores", clipscores, fingerprint=fingerprint)
        print("CLIP stage complete; values remain hidden.", flush=True)

    hps = _load_completed_array(
        cache, "hps_scores", fingerprint=fingerprint, expected_count=count
    )
    if hps is None:
        print("Computing complete HPSv2.1 stage (values hidden)...", flush=True)
        hps = compute_hps_scores(
            {},
            records,
            version=metric["hps_version"],
            batch_size=int(runtime["hpsv2_batch_size"]),
        )
        _complete_array(cache, "hps_scores", hps, fingerprint=fingerprint)
        print("HPSv2.1 stage complete; values remain hidden.", flush=True)

    arrays = (dino, clip, clip_alignment, clipscores, hps)
    if any(value is None or len(value) != count for value in arrays):
        raise RuntimeError("Formal feature stages are not all complete.")
    print("All metric stages complete; aggregating preregistered results...", flush=True)
    paired_report = _build_outputs(
        config,
        staging,
        dino_features=dino,
        clip_features=clip,
        clip_alignment=clip_alignment,
        clipscores=clipscores,
        hps_scores=hps,
    )
    completion = {
        "evaluation_fingerprint": fingerprint,
        "formal_evaluation_config_sha256": CONFIG_SHA256,
        "formal_protocol_sha256": FORMAL_PROTOCOL_SHA256,
        "manifest_sha256": config["generation"]["manifest_sha256"],
        "metric_protocol_sha256": config["metric_protocol"]["sha256"],
        "image_count": 16000,
        "prompt_group_count": 2000,
        "method_count": 4,
        "all_metric_stages_complete_before_reporting": True,
        "formal_metric_values_reported": True,
        "primary_result_label": paired_report["primary_acceptance"]["result_label"],
    }
    write_json(staging / "evaluation_complete.json", completion)
    staging.replace(final)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--confirm-evaluation-sha256",
        default=None,
        help="Required to start or resume formal metric computation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "validated": True,
                    "formal_evaluation_config_sha256": CONFIG_SHA256,
                    "evaluation_fingerprint": config["_fingerprint"],
                    "image_count": len(config["_records"]),
                    "prompt_group_count": len(config["_groups"]),
                    "metrics": config["statistics"]["core_metrics"],
                    "final_metrics_exist": (
                        PROJECT_ROOT / config["runtime"]["final_directory"]
                    ).exists(),
                },
                indent=2,
            )
        )
        return
    if args.confirm_evaluation_sha256 != CONFIG_SHA256:
        raise ValueError(
            "Formal evaluation requires --confirm-evaluation-sha256 "
            f"{CONFIG_SHA256}."
        )
    final = evaluate(config)
    summary = json.loads((final / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, list):
        raise RuntimeError("Formal summary must be a JSON array.")
    paired = _load_json(final / "paired_bootstrap.json")
    print(json.dumps({"summary": summary, "primary": paired["primary_acceptance"]}, indent=2))


if __name__ == "__main__":
    main()
