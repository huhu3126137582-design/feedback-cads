#!/usr/bin/env python3
"""Validate the frozen COCO-Test-500 protocol without generating images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads.experiments import read_jsonl  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(
    section: dict[str, Any], path_key: str, hash_key: str
) -> Path:
    path = PROJECT_ROOT / section[path_key]
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != section[hash_key]:
        raise ValueError(f"Frozen artifact hash changed: {path}")
    return path


def _selection_value(data: dict[str, Any], key: str) -> float:
    try:
        return float(data[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing frozen selection value: {key}") from error


def validate_protocol(
    config_path: Path,
    *,
    require_unstarted: bool,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Formal protocol must be a YAML mapping.")

    experiment = config["experiment"]
    required_experiment = {
        "protocol_revision": 1,
        "protocol_frozen_before_any_formal_image_generation": True,
        "protocol_frozen_before_any_formal_metric_inspection": True,
        "formal_test_manifest_prepared": True,
        "formal_test_images_generated_at_freeze": False,
        "formal_test_metrics_seen_at_freeze": False,
        "one_shot_evaluation": True,
        "no_parameter_selection_or_tuning_on_test": True,
    }
    for key, expected in required_experiment.items():
        if experiment.get(key) != expected:
            raise ValueError(f"Formal-test disclosure changed: {key}")

    dataset = config["dataset"]
    dataset_path = checked_path(dataset, "path", "sha256")
    metadata_path = checked_path(
        dataset, "split_metadata_path", "split_metadata_sha256"
    )
    prompts = read_jsonl(dataset_path)
    expected_prompts = int(dataset["expected_prompts"])
    if len(prompts) != expected_prompts:
        raise ValueError("Formal prompt count changed.")
    if [int(row["split_index"]) for row in prompts] != list(
        range(expected_prompts)
    ):
        raise ValueError("Formal split indices are not contiguous.")
    if {row.get("split") for row in prompts} != {
        dataset["expected_split"]
    }:
        raise ValueError("Formal split label changed.")
    if len({row["prompt_id"] for row in prompts}) != expected_prompts:
        raise ValueError("Formal prompt IDs are not unique.")
    if len({int(row["image_id"]) for row in prompts}) != expected_prompts:
        raise ValueError("Formal image IDs are not unique.")
    if any(not str(row.get("prompt", "")).strip() for row in prompts):
        raise ValueError("A formal prompt is empty.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        int(metadata["dev_count"]) != 50
        or int(metadata["test_count"]) != expected_prompts
        or metadata["test_sha256"] != dataset["sha256"]
        or metadata["image_ids_disjoint"] is not True
    ):
        raise ValueError("COCO split metadata changed.")
    dev_path = PROJECT_ROOT / "data/coco/coco_dev50.jsonl"
    if sha256(dev_path) != metadata["dev_sha256"]:
        raise ValueError("Frozen COCO-Dev-50 changed.")
    dev = read_jsonl(dev_path)
    if {int(row["image_id"]) for row in dev} & {
        int(row["image_id"]) for row in prompts
    }:
        raise ValueError("Development and formal image IDs overlap.")

    model = config["model"]
    model_path = PROJECT_ROOT / model["snapshot"]
    if (
        model_path.name != str(model["revision"])
        or not model_path.joinpath("model_index.json").is_file()
    ):
        raise ValueError("Frozen SD1.5 snapshot is missing or mismatched.")
    if (model["resolution"], model["dtype"]) != (512, "float16"):
        raise ValueError("Formal model resolution or dtype changed.")

    sampling = config["sampling"]
    if sampling != {
        "scheduler": "DDIM",
        "num_inference_steps": 50,
        "eta": 0.0,
        "cfg_scale": 7.5,
        "candidates_per_prompt": 8,
        "prompt_batch_size": 1,
        "method_order": [
            "vanilla",
            "original_cads",
            "a_star",
            "b_feedback",
        ],
    }:
        raise ValueError("Formal sampling protocol changed.")

    methods = config["methods"]
    if list(methods) != sampling["method_order"]:
        raise ValueError("Formal method declaration order changed.")
    original_path = checked_path(
        methods["original_cads"], "config_path", "config_sha256"
    )
    a_path = checked_path(methods["a_star"], "config_path", "config_sha256")
    b_path = checked_path(
        methods["b_feedback"], "config_path", "config_sha256"
    )
    a_selection_path = checked_path(
        methods["a_star"], "selection_path", "selection_sha256"
    )
    b_selection_path = checked_path(
        methods["b_feedback"], "selection_path", "selection_sha256"
    )
    reference_path = checked_path(
        methods["b_feedback"], "reference_path", "reference_sha256"
    )
    original = yaml.safe_load(original_path.read_text(encoding="utf-8"))
    a_config = yaml.safe_load(a_path.read_text(encoding="utf-8"))
    b_config = yaml.safe_load(b_path.read_text(encoding="utf-8"))
    a_selection = json.loads(a_selection_path.read_text(encoding="utf-8"))
    b_selection = json.loads(b_selection_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if original["method"]["numerical_reproduction_of_paper_sd_results"]:
        raise ValueError("Original CADS claim was overstated.")
    if original["project_target"]["revision"] != model["revision"]:
        raise ValueError("Original CADS uses another SD1.5 revision.")
    a_rho = float(methods["a_star"]["fixed_rho"])
    if (
        a_rho != 0.55
        or float(a_config["cads"]["fixed_rho_a"]) != a_rho
        or _selection_value(a_selection, "selected_a_star_rho") != a_rho
    ):
        raise ValueError("Frozen A* rho changed.")
    b_alpha = float(methods["b_feedback"]["fixed_reference_margin_alpha"])
    if (
        b_alpha != 0.15
        or float(b_config["feedback"]["reference_margin_alpha"])
        != b_alpha
        or _selection_value(b_selection, "selected_alpha") != b_alpha
    ):
        raise ValueError("Frozen B alpha changed.")
    if (
        b_config["method"]["final_selected_configuration"] is not True
        or b_selection["no_further_alpha_extension_allowed"] is not True
        or reference["formal_test_data_used"] is not False
        or int(reference["step_count"]) != 50
        or float(reference["rho_a_star"]) != a_rho
        or float(reference["reference_margin_alpha"]) != 0.0
    ):
        raise ValueError("Frozen B/reference state changed.")

    randomness = config["randomness"]
    required_randomness_flags = (
        "latent_seed_range_disjoint_from_development",
        "condition_seed_range_disjoint_from_development",
        "identical_prompt_candidate_latents_across_all_methods",
        "identical_positive_condition_seed_across_cads_methods",
        "vanilla_does_not_consume_condition_noise",
        "global_cpu_cuda_rng_must_remain_unchanged",
    )
    if any(randomness.get(key) is not True for key in required_randomness_flags):
        raise ValueError("Formal randomness pairing flags changed.")
    candidate_count = int(sampling["candidates_per_prompt"])
    test_latent_seeds = {
        int(randomness["latent_seed_base"])
        + int(randomness["latent_prompt_stride"]) * prompt_index
        + candidate_index
        for prompt_index in range(expected_prompts)
        for candidate_index in range(candidate_count)
    }
    dev_latent_seeds = {
        320000 + 1000 * prompt_index + candidate_index
        for prompt_index in range(50)
        for candidate_index in range(candidate_count)
    }
    sample_stride = 1_000_003
    test_positive_seeds = {
        int(randomness["condition_seed_base"])
        + int(randomness["condition_prompt_stride"]) * prompt_index
        + sample_stride * candidate_index
        for prompt_index in range(expected_prompts)
        for candidate_index in range(candidate_count)
    }
    dev_positive_seeds = {
        2920000 + 100003 * prompt_index + sample_stride * candidate_index
        for prompt_index in range(50)
        for candidate_index in range(candidate_count)
    }
    modulus = 2**63 - 1
    test_negative_seeds = {
        (seed + 10_000_019) % modulus for seed in test_positive_seeds
    }
    if len(test_latent_seeds) != expected_prompts * candidate_count:
        raise ValueError("Formal latent seeds collide internally.")
    if len(test_positive_seeds) != expected_prompts * candidate_count:
        raise ValueError("Formal positive-condition seeds collide internally.")
    if test_latent_seeds & dev_latent_seeds:
        raise ValueError("Formal and development latent seeds overlap.")
    if test_positive_seeds & dev_positive_seeds:
        raise ValueError("Formal and development condition seeds overlap.")
    if test_negative_seeds & (test_positive_seeds | dev_positive_seeds):
        raise ValueError("Original negative-condition seeds overlap positives.")

    generation = config["generation"]
    method_count = len(sampling["method_order"])
    expected_per_method = expected_prompts * candidate_count
    if (
        int(generation["expected_images_per_method"]) != expected_per_method
        or int(generation["expected_total_images"])
        != expected_per_method * method_count
        or generation["do_not_compute_or_inspect_quality_metrics_until_all_methods_complete"]
        is not True
        or generation["do_not_change_prompts_seeds_or_parameters_after_partial_generation"]
        is not True
    ):
        raise ValueError("Formal generation count or no-peeking rule changed.")
    output_root = PROJECT_ROOT / generation["output_root"]
    if require_unstarted and output_root.exists():
        raise ValueError(
            f"Formal generation output already exists at freeze audit: {output_root}"
        )

    evaluation = config["evaluation"]
    metric_path = checked_path(
        evaluation, "metric_config_path", "metric_config_sha256"
    )
    metric_config = yaml.safe_load(metric_path.read_text(encoding="utf-8"))
    if (
        evaluation["statistical_unit"] != "prompt"
        or int(evaluation["bootstrap_replicates"])
        != int(metric_config["bootstrap_replicates"])
        or int(evaluation["bootstrap_seed"])
        != int(metric_config["bootstrap_seed"])
        or evaluation["primary_comparison"] != "b_feedback_vs_a_star"
        or evaluation["noninferiority_rule"]
        != "ci95_lower_strictly_greater_than_negative_margin"
        or evaluation["strong_improvement_rule"]
        != "ci95_lower_strictly_greater_than_zero"
    ):
        raise ValueError("Formal statistical protocol changed.")
    acceptance = config["acceptance"]
    if (
        acceptance["assessment_success"][
            "require_all_three_core_metrics_noninferior"
        ]
        is not True
        or acceptance["assessment_success"][
            "require_at_least_one_core_metric_point_estimate_improvement"
        ]
        is not True
        or acceptance["closed_loop_diversity_success"][
            "require_dino_point_estimate_improvement"
        ]
        is not True
        or acceptance["strong_research_result"][
            "require_at_least_one_core_metric_ci95_lower_above_zero"
        ]
        is not True
    ):
        raise ValueError("Formal acceptance rule changed.")

    return {
        "passed": True,
        "protocol_sha256": sha256(config_path),
        "dataset_sha256": sha256(dataset_path),
        "prompt_count": expected_prompts,
        "method_count": method_count,
        "candidate_count": candidate_count,
        "expected_total_images": int(generation["expected_total_images"]),
        "unique_test_latent_seeds": len(test_latent_seeds),
        "unique_test_positive_condition_seeds": len(test_positive_seeds),
        "seed_ranges_disjoint_from_development": True,
        "frozen_a_star_rho": a_rho,
        "frozen_b_reference_margin_alpha": b_alpha,
        "formal_output_absent": not output_root.exists(),
        "formal_metrics_seen_at_freeze": False,
        "formal_metrics_output_exists": output_root.joinpath(
            "metrics/evaluation_complete.json"
        ).is_file(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/formal_test_coco_test500.yaml",
    )
    parser.add_argument(
        "--allow-started",
        action="store_true",
        help="Validate a frozen protocol after its output directory exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_protocol(
        args.config,
        require_unstarted=not args.allow_started,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
