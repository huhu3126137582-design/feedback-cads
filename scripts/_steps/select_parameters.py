#!/usr/bin/env python3
"""Select frozen Stage-A or Stage-B parameters on COCO-Dev-50."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads.configuration import load_yaml_profile  # noqa: E402
from feedback_cads.experiments import read_jsonl, write_json  # noqa: E402


A_OUTPUTS = {
    "coarse": PROJECT_ROOT / "outputs/a_strength_curve_coco_dev50_coarse",
    "fine": PROJECT_ROOT / "outputs/a_strength_curve_coco_dev50_fine",
}
B_CONFIG_BUNDLE = PROJECT_ROOT / "configs/b_feedback_search_coco_dev50.yaml"
B_EXPERIMENTS = {
    0.0: ("alpha_0p00", PROJECT_ROOT / "outputs/b_feedback_coco_dev50"),
    0.05: (
        "alpha_0p05",
        PROJECT_ROOT / "outputs/b_feedback_alpha_0p05_coco_dev50",
    ),
    0.10: (
        "alpha_0p10",
        PROJECT_ROOT / "outputs/b_feedback_alpha_0p10_coco_dev50",
    ),
    0.15: (
        "alpha_0p15",
        PROJECT_ROOT / "outputs/b_feedback_alpha_0p15_coco_dev50",
    ),
    0.20: (
        "alpha_0p20",
        PROJECT_ROOT / "outputs/b_feedback_alpha_0p20_coco_dev50",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    a_parser = commands.add_parser("a-star", help="Select A* from coarse/fine runs.")
    a_parser.add_argument("--coarse-dir", type=Path, default=A_OUTPUTS["coarse"])
    a_parser.add_argument("--fine-dir", type=Path, default=A_OUTPUTS["fine"])
    a_parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/a_star_coco_dev50/A_STAR_SELECTION.json",
    )

    b_parser = commands.add_parser(
        "b-margin", help="Select the Stage-B reference margin."
    )
    b_parser.add_argument("--extended", action="store_true")
    b_parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _protocol_without_grid(config: dict[str, Any]) -> dict[str, Any]:
    sampling = dict(config["sampling"])
    sampling.pop("rho_grid")
    return {
        "dataset": config["dataset"],
        "model": config["model"],
        "sampling": sampling,
        "cads": config["cads"],
        "randomness": config["randomness"],
    }


def _baseline_hashes(experiment_dir: Path) -> dict[tuple[str, int], str]:
    return {
        (row["prompt_id"], int(row["candidate_id"])): row["image_sha256"]
        for row in read_jsonl(experiment_dir / "manifest.jsonl")
        if float(row["rho"]) == 0.0
    }


def select_a_star(args: argparse.Namespace) -> dict[str, Any]:
    coarse_dir = args.coarse_dir.resolve()
    fine_dir = args.fine_dir.resolve()
    coarse_config = read_json(coarse_dir / "run_config.json")
    fine_config = read_json(fine_dir / "run_config.json")
    if coarse_config["experiment"]["purpose"] != "a_star_coarse_coco_dev50":
        raise ValueError("The coarse experiment has the wrong purpose.")
    if fine_config["experiment"]["purpose"] != "a_star_fine_coco_dev50":
        raise ValueError("The fine experiment has the wrong purpose.")
    if _protocol_without_grid(coarse_config) != _protocol_without_grid(fine_config):
        raise ValueError("Coarse and fine experiments do not share one protocol.")
    if _baseline_hashes(coarse_dir) != _baseline_hashes(fine_dir):
        raise ValueError("Coarse and fine runs do not reuse the same rho=0 images.")

    coarse_audit = read_json(coarse_dir / "metrics/curve_audit.json")
    fine_audit = read_json(fine_dir / "metrics/curve_audit.json")
    center = float(coarse_audit["selected_coarse_center_rho"])
    if center != float(fine_config["a_star_search"]["coarse_center"]):
        raise ValueError("Fine-grid center does not match the frozen coarse audit.")

    coarse_summary = pd.read_csv(coarse_dir / "metrics/summary.csv")
    fine_summary = pd.read_csv(fine_dir / "metrics/summary.csv")
    baseline_coarse = coarse_summary.loc[coarse_summary["rho"] == 0.0].iloc[0]
    baseline_fine = fine_summary.loc[fine_summary["rho"] == 0.0].iloc[0]
    for metric in ("dino_diversity", "clipscore", "hpsv2"):
        if abs(float(baseline_coarse[metric]) - float(baseline_fine[metric])) > 1e-12:
            raise ValueError(f"Baseline metric mismatch for {metric}.")

    decisions: dict[float, dict[str, Any]] = {}
    dino_by_rho = {0.0: float(baseline_coarse["dino_diversity"])}
    for audit, summary, phase in (
        (coarse_audit, coarse_summary, "coarse"),
        (fine_audit, fine_summary, "fine"),
    ):
        lookup = {
            float(row.rho): float(row.dino_diversity)
            for row in summary.itertuples(index=False)
        }
        for decision in audit["decisions"]:
            rho = float(decision["rho"])
            if rho in decisions:
                raise ValueError(f"rho={rho} was evaluated in both phases.")
            decisions[rho] = {**decision, "search_phase": phase}
            dino_by_rho[rho] = lookup[rho]

    eligible = [
        item
        for item in decisions.values()
        if item["passes_all_quality_constraints"] and item["dino_point_improved"]
    ]
    selected = (
        max(eligible, key=lambda item: dino_by_rho[float(item["rho"])])
        if eligible
        else None
    )
    selected_rho = None if selected is None else float(selected["rho"])
    result = {
        "development_split": "COCO-Dev-50",
        "formal_test_data_used": False,
        "selection_protocol": "frozen coarse grid then 0.05 fine grid within ±0.20",
        "coarse_grid": [float(value) for value in coarse_config["sampling"]["rho_grid"]],
        "coarse_center_rho": center,
        "new_fine_grid_points": [
            float(value)
            for value in fine_config["sampling"]["rho_grid"]
            if float(value) != 0.0
        ],
        "clipscore_noninferiority_margin": coarse_audit[
            "clipscore_noninferiority_margin"
        ],
        "hpsv2_noninferiority_margin": coarse_audit[
            "hpsv2_noninferiority_margin"
        ],
        "selected_a_star_rho": selected_rho,
        "selection_blocker": (
            None
            if selected is not None
            else "No nonzero configuration both improves DINO diversity and passes both quality constraints."
        ),
        "selected_metrics": (
            None
            if selected is None
            else {
                "dino_diversity": dino_by_rho[selected_rho],
                "dino_mean_difference": selected["dino_mean_difference"],
                "clipscore_mean_difference": selected["clipscore_mean_difference"],
                "hpsv2_mean_difference": selected["hpsv2_mean_difference"],
                "search_phase": selected["search_phase"],
            }
        ),
        "decisions": sorted(decisions.values(), key=lambda item: item["rho"]),
        "all_rho_dino_diversity": {
            str(rho): dino_by_rho[rho] for rho in sorted(dino_by_rho)
        },
        # Historical wording is retained byte-for-byte because this selection
        # artifact was frozen before the formal split expanded to Test-500.
        "test_data_policy": "Freeze A* here; do not tune on COCO-Test-300.",
    }
    write_json(args.output.resolve(), result)
    return result


def _common_b_protocol(config: dict[str, Any]) -> dict[str, Any]:
    feedback = dict(config["feedback"])
    feedback.pop("reference_margin_alpha")
    return {
        "dataset": config["dataset"],
        "model": config["model"],
        "sampling": config["sampling"],
        "cads": config["cads"],
        "feedback_without_alpha": feedback,
        "reference": config["reference"],
        "paired_a_star": config["paired_a_star"],
        "randomness": config["randomness"],
        "evaluation": config["evaluation"],
    }


def select_b_margin(args: argparse.Namespace) -> dict[str, Any]:
    experiments = B_EXPERIMENTS if args.extended else {
        alpha: B_EXPERIMENTS[alpha] for alpha in (0.0, 0.05, 0.10)
    }
    output_path = args.output
    if output_path is None:
        output_path = PROJECT_ROOT / (
            "outputs/b_feedback_margin_selection_extended_coco_dev50/ALPHA_SELECTION_EXTENDED.json"
            if args.extended
            else "outputs/b_feedback_margin_selection_coco_dev50/ALPHA_SELECTION.json"
        )
    loaded = {
        alpha: load_yaml_profile(B_CONFIG_BUNDLE, profile=profile)
        for alpha, (profile, _directory) in experiments.items()
    }
    configs = {alpha: item[0] for alpha, item in loaded.items()}
    identities = {alpha: item[1] for alpha, item in loaded.items()}
    baseline_protocol = _common_b_protocol(configs[0.0])
    for alpha, config in configs.items():
        if config["experiment"]["formal_test_data_used"] is not False:
            raise ValueError(f"alpha={alpha} used formal test data.")
        if _common_b_protocol(config) != baseline_protocol:
            raise ValueError(f"alpha={alpha} does not share the frozen protocol.")
        if float(config["feedback"]["reference_margin_alpha"]) != alpha:
            raise ValueError(f"alpha={alpha} configuration is mislabeled.")

    rule_key = "extension_selection_rule" if args.extended else "selection_rule"
    rule_alphas = (0.15, 0.20) if args.extended else (0.05, 0.10)
    selection_rules = [configs[alpha][rule_key] for alpha in rule_alphas]
    if selection_rules[0] != selection_rules[1]:
        raise ValueError("Margin configurations use different selection rules.")
    rule = selection_rules[0]
    if [float(value) for value in rule["alpha_grid"]] != sorted(experiments):
        raise ValueError("The frozen alpha grid changed.")
    if rule["maximize"] != "dino_diversity" or rule["tie_breaker"] != "smaller_alpha":
        raise ValueError("The frozen B selection objective changed.")
    saturation_limit = float(rule["maximum_mean_active_boundary_rate"])
    expected_stage1_sha = None
    if args.extended:
        if [float(value) for value in rule["added_alpha_grid"]] != [0.15, 0.20]:
            raise ValueError("The adaptive alpha extension changed.")
        if float(rule["absolute_stop_alpha"]) != 0.20 or rule["no_further_extension"] is not True:
            raise ValueError("The extended search stop rule changed.")
        stage1_path = PROJECT_ROOT / "outputs/b_feedback_margin_selection_coco_dev50/ALPHA_SELECTION.json"
        expected_stage1_sha = configs[0.15]["prerequisite_stage1_selection"]["sha256"]
        if sha256(stage1_path) != expected_stage1_sha:
            raise ValueError("Stage-1 alpha selection SHA-256 changed.")
        stage1 = read_json(stage1_path)
        if stage1["formal_test_data_used"] or stage1["selected_alpha"] != 0.1:
            raise ValueError("Invalid stage-1 boundary prerequisite.")

    decisions = []
    a_metrics_by_alpha = {}
    artifact_hashes = {}
    for alpha, (_profile, experiment_dir) in experiments.items():
        metrics_dir = experiment_dir / "metrics"
        complete_path = metrics_dir / "evaluation_complete.json"
        paired_path = metrics_dir / "paired_bootstrap.json"
        summary_path = metrics_dir / "summary.csv"
        generation_path = experiment_dir / "generation_complete.json"
        for path in (complete_path, paired_path, summary_path, generation_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        complete = read_json(complete_path)
        paired = read_json(paired_path)
        generation = read_json(generation_path)
        if complete["formal_test_data_used"] or paired["formal_test_data_used"] or generation["formal_test_data_used"]:
            raise ValueError(f"alpha={alpha} used formal test data.")
        if not complete["frozen_a_metric_reproduction_passed"]:
            raise ValueError(f"alpha={alpha} did not reproduce frozen A metrics.")
        if not complete["all_prompt_candidate_pairs_share_latents_and_condition_seed"]:
            raise ValueError(f"alpha={alpha} lost seed pairing.")

        summary = pd.read_csv(summary_path).set_index("method")
        a_row = summary.loc["A_star"]
        b_row = summary.loc["B_feedback"]
        a_metrics_by_alpha[alpha] = {
            metric: float(a_row[metric])
            for metric in ("dino_diversity", "clipscore", "hpsv2")
        }
        noninferiority = paired["quality_noninferiority"]
        boundary_rate = float(paired["controller_behavior"]["mean_active_boundary_rate"])
        quality_passed = bool(noninferiority["both_passed"])
        saturation_passed = boundary_rate < saturation_limit
        decisions.append({
            "alpha": alpha,
            "dino_diversity": float(b_row["dino_diversity"]),
            "clipscore": float(b_row["clipscore"]),
            "hpsv2": float(b_row["hpsv2"]),
            "mean_active_rho": float(paired["controller_behavior"]["mean_active_rho_across_prompts"]),
            "mean_active_boundary_rate": boundary_rate,
            "clipscore_noninferior": bool(noninferiority["clipscore_passed"]),
            "hpsv2_noninferior": bool(noninferiority["hpsv2_passed"]),
            "saturation_constraint_passed": saturation_passed,
            "eligible": quality_passed and saturation_passed,
            "paired_differences_vs_a": paired["comparisons"],
        })
        artifact_hashes[str(alpha)] = {
            "config_sha256": identities[alpha]["legacy_sha256"],
            "generation_complete_sha256": sha256(generation_path),
            "summary_sha256": sha256(summary_path),
            "paired_report_sha256": sha256(paired_path),
            "evaluation_complete_sha256": sha256(complete_path),
        }

    frozen_a = a_metrics_by_alpha[0.0]
    for alpha, metrics in a_metrics_by_alpha.items():
        for metric, value in metrics.items():
            if abs(value - frozen_a[metric]) > 1e-12:
                raise ValueError(f"Recomputed A {metric} differs for alpha={alpha}.")
    eligible = [decision for decision in decisions if decision["eligible"]]
    selected = (
        sorted(eligible, key=lambda item: (-item["dino_diversity"], item["alpha"]))[0]
        if eligible
        else None
    )
    selected_alpha = None if selected is None else selected["alpha"]
    selected_improves_a_dino = bool(
        selected is not None and selected["dino_diversity"] > frozen_a["dino_diversity"]
    )
    selected_dino_strong = bool(
        selected is not None
        and selected["paired_differences_vs_a"]["dino_diversity"]["ci95_lower"] > 0.0
    )
    result = {
        "development_split": "COCO-Dev-50",
        "formal_test_data_used": False,
        "selection_rule_frozen_before_margin_generation": True,
        "selection_rule": rule,
        "frozen_a_star_metrics": frozen_a,
        "decisions": sorted(decisions, key=lambda item: item["alpha"]),
        "selected_alpha": selected_alpha,
        "selection_blocker": None if selected is not None else "No alpha passed quality and saturation constraints.",
        "selected_dino_point_improves_a_star": selected_improves_a_dino,
        "selected_dino_ci_lower_above_zero": selected_dino_strong,
        "closed_loop_development_point_improvement_observed": selected_improves_a_dino,
        "statistically_significant_closed_loop_development_evidence": selected_dino_strong,
        "test_data_policy": "Freeze the selected alpha here; do not retune on COCO-Test-500.",
        "artifact_hashes": artifact_hashes,
    }
    if args.extended:
        result["adaptive_extension_after_stage1_results"] = True
        result["stage1_selection_sha256"] = expected_stage1_sha
        result["absolute_stop_alpha"] = 0.20
        result["no_further_alpha_extension_allowed"] = True
    write_json(output_path.resolve(), result)
    return result


def main() -> None:
    args = parse_args()
    result = select_a_star(args) if args.command == "a-star" else select_b_margin(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
