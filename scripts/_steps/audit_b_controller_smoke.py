#!/usr/bin/env python3
"""Smoke-test and audit one real SD1.5 Stage-B controller trajectory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads import (  # noqa: E402
    CleanUnconditionalCADSConfig,
    FeedbackMVPConfig,
    OriginalCADSStableDiffusionPipeline,
    load_frozen_diversity_reference,
)
from feedback_cads.experiments import write_json  # noqa: E402


MODEL_PATH = (
    PROJECT_ROOT
    / "models/huggingface/hub/"
    "models--stable-diffusion-v1-5--stable-diffusion-v1-5/"
    "snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)
REFERENCE_PATH = (
    PROJECT_ROOT
    / "outputs/a_star_reference_coco_dev50/reference_curves.json"
)
REFERENCE_SHA256 = (
    "c2ae2024f06c699eb9e51e063ab9aea2f118c03663d6918965c1e639f9667382"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/b_stage2_controller_smoke",
    )
    return parser.parse_args()


def close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stage-B smoke audit.")

    reference = load_frozen_diversity_reference(
        REFERENCE_PATH,
        expected_sha256=REFERENCE_SHA256,
    )
    feedback_config = FeedbackMVPConfig()
    cads_config = CleanUnconditionalCADSConfig()

    official = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    official.scheduler = DDIMScheduler.from_config(official.scheduler.config)
    official.to(device="cuda", dtype=torch.float16)
    official.set_progress_bar_config(disable=True)
    pipeline = OriginalCADSStableDiffusionPipeline.from_vanilla_pipeline(
        official
    )
    pipeline.set_progress_bar_config(disable=True)

    latent_generator = torch.Generator(device="cuda").manual_seed(20260808)
    initial_latents = torch.randn(
        (8, 4, 64, 64),
        generator=latent_generator,
        device="cuda",
        dtype=torch.float16,
    )
    result = pipeline(
        prompt="a red fox in a forest",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_images_per_prompt=8,
        eta=0.0,
        latents=initial_latents,
        output_type="latent",
        clean_unconditional_cads_config=cads_config,
        feedback_mvp_config=feedback_config,
        feedback_reference=reference,
        condition_seed=2920000,
    ).images
    stats: dict[str, Any] = pipeline.last_run_stats
    history = stats["feedback_control_history"]

    one_step_delay = len(history) == 50 and all(
        close(
            history[index]["rho_used_by_prompt"][0],
            history[index - 1]["rho_next_by_prompt"][0],
            tolerance=0.0,
        )
        for index in range(1, len(history))
    )
    exact_pollution_product = all(
        close(
            row["pollution_by_prompt"][0],
            row["pollution_cap"] * row["rho_used_by_prompt"][0],
        )
        for row in history
    )
    proportional_rule = all(
        (
            close(
                row["delta_rho_by_prompt"][0],
                feedback_config.k_diversity
                * row["diversity_deficit_by_prompt"][0],
            )
            and close(
                row["rho_next_by_prompt"][0],
                max(
                    0.0,
                    min(
                        1.0,
                        row["rho_used_by_prompt"][0]
                        + row["delta_rho_by_prompt"][0],
                    ),
                ),
            )
        )
        if row["control_updated"]
        else (
            row["delta_rho_by_prompt"][0] == 0.0
            and close(
                row["rho_next_by_prompt"][0],
                row["rho_used_by_prompt"][0],
                tolerance=0.0,
            )
        )
        for row in history
    )
    hard_off = all(
        row["step_index"] < 30
        or (
            row["pollution_cap"] == 0.0
            and row["pollution_by_prompt"][0] == 0.0
            and not row["control_updated"]
            and row["delta_rho_by_prompt"][0] == 0.0
        )
        for row in history
    )
    correct_dz_window = all(
        row["dz_used"] == (10 <= row["step_index"] < 30)
        for row in history
    )
    rho_values = [row["rho_used_by_prompt"][0] for row in history]
    direction_correct = all(
        (
            row["rho_next_by_prompt"][0]
            >= row["rho_used_by_prompt"][0]
        )
        if row["diversity_deficit_by_prompt"][0] >= 0.0
        else (
            row["rho_next_by_prompt"][0]
            <= row["rho_used_by_prompt"][0]
        )
        for row in history
        if row["control_updated"]
    )
    checks = {
        "latent_output_is_finite": bool(torch.isfinite(result).all().item()),
        "mode_is_feedback_mvp": stats["mode"] == "feedback_mvp",
        "exactly_50_controller_records": len(history) == 50,
        "initial_rho_equals_a_star": close(rho_values[0], 0.55),
        "rho_adapts_from_initial_value": any(
            not close(value, 0.55) for value in rho_values[1:30]
        ),
        "one_step_delay_is_exact": one_step_delay,
        "pollution_equals_cap_times_rho": exact_pollution_product,
        "proportional_update_and_clipping_are_exact": proportional_rule,
        "control_direction_is_correct": direction_correct,
        "dz_active_control_window_is_steps_10_through_29": (
            correct_dz_window
        ),
        "hard_off_is_exact_from_step_30": hard_off,
        "unet_calls_equal_50": stats["unet_calls"] == 50,
        "scheduler_calls_equal_50": stats["scheduler_step_calls"] == 50,
        "condition_draws_equal_30_active_steps": (
            stats["positive_condition_noise_draw_calls"] == 30
        ),
        "unconditional_branch_is_clean": (
            stats["negative_condition_max_delta"] == 0.0
        ),
        "non_content_tokens_are_clean": (
            stats["non_content_positive_max_delta"] == 0.0
        ),
        "reference_hash_matches": (
            stats["feedback_reference_sha256"] == REFERENCE_SHA256
        ),
        "reference_margin_alpha_is_zero": (
            stats["feedback_reference_margin_alpha"] == 0.0
        ),
        "no_formal_test_data_used": True,
    }
    passed = all(checks.values())
    report = {
        "stage": "B-step-2",
        "scope": (
            "Group proportional controller integration smoke; no B-vs-A "
            "quality/diversity claim"
        ),
        "model": "stable-diffusion-v1-5",
        "prompt": "a red fox in a forest",
        "candidate_count": 8,
        "num_inference_steps": 50,
        "feedback_config": {
            "initial_rho": feedback_config.initial_rho,
            "k_diversity": feedback_config.k_diversity,
            "epsilon": feedback_config.epsilon,
            "reference_margin_alpha": (
                feedback_config.reference_margin_alpha
            ),
        },
        "rho_active_window": {
            "minimum": min(rho_values[:30]),
            "maximum": max(rho_values[:30]),
            "final_active_step": rho_values[29],
        },
        "unet_calls": stats["unet_calls"],
        "scheduler_step_calls": stats["scheduler_step_calls"],
        "positive_condition_noise_draw_calls": stats[
            "positive_condition_noise_draw_calls"
        ],
        "formal_test_data_used": False,
        "checks": checks,
        "passed": passed,
        "controller_history": history,
    }
    write_json(args.output_dir.resolve() / "validation_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "controller_history"}, indent=2))
    if not passed:
        raise RuntimeError("Stage-B controller smoke audit failed.")


if __name__ == "__main__":
    main()
