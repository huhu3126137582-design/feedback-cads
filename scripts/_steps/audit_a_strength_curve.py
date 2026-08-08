#!/usr/bin/env python3
"""Apply the preregistered stage-A point/CI rules to a completed curve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads.experiments import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/a_strength_curve_dev20",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = args.experiment_dir.resolve() / "metrics"
    summary = pd.read_csv(metrics_dir / "summary.csv")
    bootstrap = json.loads(
        (metrics_dir / "paired_bootstrap.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (metrics_dir / "evaluation_complete.json").read_text(encoding="utf-8")
    )
    generation = json.loads(
        (args.experiment_dir.resolve() / "generation_complete.json").read_text(
            encoding="utf-8"
        )
    )
    run_config = json.loads(
        (args.experiment_dir.resolve() / "run_config.json").read_text(
            encoding="utf-8"
        )
    )
    purpose = run_config["experiment"]["purpose"]
    expected_rhos = tuple(float(value) for value in evaluation["rho_grid"])
    if tuple(summary["rho"].tolist()) != expected_rhos:
        raise ValueError("Summary does not contain the frozen rho grid.")
    if summary.isna().any().any():
        raise ValueError("Completed curve must not contain missing metrics.")

    baseline = summary.loc[summary["rho"] == 0.0].iloc[0]
    margins = {
        "clipscore": 0.01 * float(baseline["clipscore"]),
        "hpsv2": 0.01 * float(baseline["hpsv2"]),
    }
    comparison_lookup = {
        (float(item["rho"]), item["metric"]): item
        for item in bootstrap["comparisons"]
    }
    decisions = []
    for row in summary.itertuples(index=False):
        rho = float(row.rho)
        if rho == 0.0:
            continue
        dino = comparison_lookup[(rho, "dino_diversity")]
        clip = comparison_lookup[(rho, "clipscore")]
        hps = comparison_lookup[(rho, "hpsv2")]
        decisions.append(
            {
                "rho": rho,
                "dino_point_improved": dino["mean_difference"] > 0.0,
                "dino_strong_improvement": dino["ci95_lower"] > 0.0,
                "clipscore_noninferior_1pct": (
                    clip["ci95_lower"] > -margins["clipscore"]
                ),
                "hpsv2_noninferior_1pct": (
                    hps["ci95_lower"] > -margins["hpsv2"]
                ),
                "passes_available_quality_constraints": (
                    clip["ci95_lower"] > -margins["clipscore"]
                    and hps["ci95_lower"] > -margins["hpsv2"]
                ),
                "passes_all_quality_constraints": (
                    clip["ci95_lower"] > -margins["clipscore"]
                    and hps["ci95_lower"] > -margins["hpsv2"]
                ),
                "dino_mean_difference": dino["mean_difference"],
                "dino_ci95": [dino["ci95_lower"], dino["ci95_upper"]],
                "clipscore_mean_difference": clip["mean_difference"],
                "clipscore_ci95": [clip["ci95_lower"], clip["ci95_upper"]],
                "hpsv2_mean_difference": hps["mean_difference"],
                "hpsv2_ci95": [hps["ci95_lower"], hps["ci95_upper"]],
            }
        )

    eligible = [
        item
        for item in decisions
        if item["passes_all_quality_constraints"]
        and item["dino_point_improved"]
    ]
    quality_safe_rhos = [0.0] + [
        item["rho"]
        for item in decisions
        if item["passes_all_quality_constraints"]
    ]
    selected_coarse_center = None
    if purpose == "a_star_coarse_coco_dev50":
        selected_coarse_center = float(
            max(
                quality_safe_rhos,
                key=lambda rho: float(
                    summary.loc[
                        summary["rho"] == rho, "dino_diversity"
                    ].iloc[0]
                ),
            )
        )
    selected_a_star = None
    if eligible and purpose != "a_star_coarse_coco_dev50":
        best = max(
            eligible,
            key=lambda item: float(
                summary.loc[
                    summary["rho"] == item["rho"], "dino_diversity"
                ].iloc[0]
            ),
        )
        selected_a_star = best["rho"]
    output = {
        "curve_generation_complete": True,
        "curve_evaluation_complete": True,
        "development_only": True,
        "formal_test_data_used": False,
        "prompt_count": int(generation["prompt_count"]),
        "candidates_per_prompt": int(generation["candidates_per_prompt"]),
        "image_count": int(generation["image_count"]),
        "clipscore_noninferiority_margin": margins["clipscore"],
        "hpsv2_noninferiority_margin": margins["hpsv2"],
        "eligible_for_a_star_selection": True,
        "search_phase": (
            "coarse" if purpose == "a_star_coarse_coco_dev50" else "fine_or_full"
        ),
        "selected_coarse_center_rho": selected_coarse_center,
        "selected_a_star_rho": selected_a_star,
        "selection_blocker": (
            "Fine-grid evaluation is required before A* can be frozen."
            if purpose == "a_star_coarse_coco_dev50"
            else (
                "No nonzero rho both improves DINO diversity and passes the "
                "CLIPScore/HPSv2 quality constraints."
                if selected_a_star is None
                else None
            )
        ),
        "decisions": decisions,
        "interpretation": {
            "highest_dino_point_rho": float(
                summary.loc[summary["dino_diversity"].idxmax(), "rho"]
            ),
            "rho_with_both_available_quality_constraints": [
                item["rho"]
                for item in decisions
                if item["passes_available_quality_constraints"]
            ],
            "rho_with_strong_dino_improvement": [
                item["rho"]
                for item in decisions
                if item["dino_strong_improvement"]
            ],
            "rho_passing_all_quality_constraints": [
                item["rho"]
                for item in decisions
                if item["passes_all_quality_constraints"]
            ],
            "selected_a_star_has_higher_dino_than_vanilla": (
                None
                if selected_a_star is None
                else bool(
                    summary.loc[
                        summary["rho"] == selected_a_star,
                        "dino_diversity",
                    ].iloc[0]
                    > baseline["dino_diversity"]
                )
            ),
            "claim": (
                "A* is selected from nonzero settings with a positive DINO "
                "diversity point difference that pass the preregistered 1% "
                "CLIPScore and HPSv2 non-inferiority checks."
            ),
        },
    }
    write_json(metrics_dir / "curve_audit.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
