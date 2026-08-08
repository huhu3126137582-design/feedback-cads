#!/usr/bin/env python3
"""Collect and freeze COCO-Dev-50 A* intermediate diversity references."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads import (  # noqa: E402
    CleanUnconditionalCADSConfig,
    OriginalCADSStableDiffusionPipeline,
    build_prompt_median_reference,
)
from feedback_cads.experiments import (  # noqa: E402
    condition_seed_for_prompt,
    latent_seeds_for_prompt,
    load_curve_config,
    make_initial_latents,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/a_strength_search_coco_dev50.yaml",
    )
    parser.add_argument("--profile", default="reference", choices=("reference",))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/a_star_reference_coco_dev50",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Smoke-test only; omitted for the frozen reference.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def main() -> None:
    args = parse_args()
    config = load_curve_config(
        args.config,
        project_root=PROJECT_ROOT,
        profile=args.profile,
    )
    prompts = config["_prompts"]
    if args.limit_prompts is not None:
        if args.limit_prompts <= 0:
            raise ValueError("--limit-prompts must be positive.")
        prompts = prompts[: args.limit_prompts]

    reference_config = config["reference"]
    selection_path = PROJECT_ROOT / reference_config["a_star_selection"]
    if sha256(selection_path) != reference_config["a_star_selection_sha256"]:
        raise RuntimeError("The frozen A* selection artifact hash changed.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rho = float(config["sampling"]["rho_grid"][0])
    if float(selection["selected_a_star_rho"]) != rho:
        raise RuntimeError("Reference rho does not match the frozen A* selection.")
    if selection["formal_test_data_used"]:
        raise RuntimeError("A* selection unexpectedly used formal test data.")

    regression_root = PROJECT_ROOT / reference_config["regression_images"]
    if not regression_root.is_dir():
        raise FileNotFoundError(regression_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_config = {key: value for key, value in config.items() if key != "_prompts"}
    run_config_path = output_dir / "run_config.json"
    if run_config_path.exists():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing["_config_sha256"] != config["_config_sha256"]:
            raise RuntimeError("Output directory contains a different config.")
    else:
        write_json(run_config_path, frozen_config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen SD1.5 reference run.")
    device = torch.device("cuda")
    model_path = Path(config["_resolved"]["model_path"])
    vanilla = StableDiffusionPipeline.from_pretrained(
        model_path,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    vanilla.scheduler = DDIMScheduler.from_config(vanilla.scheduler.config)
    vanilla.to(device=device, dtype=torch.float16)
    vanilla.set_progress_bar_config(disable=True)
    pipeline = OriginalCADSStableDiffusionPipeline.from_vanilla_pipeline(vanilla)
    pipeline.set_progress_bar_config(disable=True)

    sampling = config["sampling"]
    cads = config["cads"]
    stage_a_config = CleanUnconditionalCADSConfig(
        noise_scale=float(cads["noise_scale"]),
        rescale_mix=float(cads["rescale_mix"]),
        fixed_rho_a=rho,
        hold_until_progress=float(cads["hold_until_progress"]),
        hard_off_progress=float(cads["hard_off_progress"]),
        content_tokens_only=bool(cads["content_tokens_only"]),
        clean_unconditional=bool(cads["clean_unconditional"]),
        rescale=bool(cads["rescale"]),
    )
    candidates = int(sampling["candidates_per_prompt"])
    steps = int(sampling["num_inference_steps"])
    resolution = int(config["model"]["resolution"])
    pool_size = int(reference_config["pool_size"])
    artifact_dir = output_dir / "prompt_artifacts"
    all_records: list[dict[str, Any]] = []
    all_artifacts: list[dict[str, Any]] = []

    for prompt_index, prompt_record in enumerate(prompts):
        prompt_id = prompt_record["prompt_id"]
        artifact_path = artifact_dir / f"{prompt_id}.json"
        if artifact_path.exists():
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if artifact["config_sha256"] != config["_config_sha256"]:
                raise RuntimeError(f"Stale prompt artifact: {artifact_path}")
            resumed = True
        else:
            latent_seeds = latent_seeds_for_prompt(config, prompt_index)
            condition_seed = condition_seed_for_prompt(config, prompt_index)
            latents = make_initial_latents(
                latent_seeds,
                device=device,
                dtype=torch.float16,
                height=resolution,
                width=resolution,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = pipeline(
                prompt=prompt_record["prompt"],
                height=resolution,
                width=resolution,
                num_inference_steps=steps,
                guidance_scale=float(sampling["cfg_scale"]),
                num_images_per_prompt=candidates,
                eta=float(sampling["eta"]),
                latents=latents,
                output_type="pil",
                clean_unconditional_cads_config=stage_a_config,
                condition_seed=condition_seed,
                collect_group_diversity=True,
                diversity_group_size=candidates,
                diversity_pool_size=pool_size,
            )
            torch.cuda.synchronize()
            elapsed_seconds = time.perf_counter() - started
            stats = dict(pipeline.last_run_stats)
            history = stats["diversity_signal_history"]
            if len(history) != steps:
                raise RuntimeError("Diversity history has the wrong length.")
            if stats["unet_calls"] != steps or stats["scheduler_step_calls"] != steps:
                raise RuntimeError("Reference collection added a model/scheduler call.")
            if stats["negative_condition_max_delta"] != 0.0:
                raise RuntimeError("Reference run changed the unconditional branch.")
            if stats["non_content_positive_max_delta"] != 0.0:
                raise RuntimeError("Reference run changed a masked-out token.")
            if any(value != 0.0 for value in stats["pollution_history"][-20:]):
                raise RuntimeError("The final 40% was not exactly clean.")
            if len(result.images) != candidates:
                raise RuntimeError("Pipeline returned the wrong image count.")

            image_matches = []
            for candidate_index, image in enumerate(result.images):
                expected_path = (
                    regression_root
                    / prompt_id
                    / f"candidate_{candidate_index:02d}.png"
                )
                if not expected_path.is_file():
                    raise FileNotFoundError(expected_path)
                match = np.array_equal(
                    np.asarray(image.convert("RGB")), image_pixels(expected_path)
                )
                image_matches.append(bool(match))
            if not all(image_matches):
                raise RuntimeError(
                    "Signal instrumentation changed a frozen A* output image."
                )

            records = []
            for item in history:
                guidance = item["guidance_diversity_by_prompt"]
                predicted_clean = item[
                    "predicted_clean_latent_diversity_by_prompt"
                ]
                if len(guidance) != 1 or len(predicted_clean) != 1:
                    raise RuntimeError("Expected exactly one prompt group per run.")
                records.append(
                    {
                        "experiment": config["experiment"]["name"],
                        "config_sha256": config["_config_sha256"],
                        "selection_sha256": reference_config[
                            "a_star_selection_sha256"
                        ],
                        "dataset_sha256": config["dataset"]["sha256"],
                        "prompt_index": prompt_index,
                        "prompt_id": prompt_id,
                        "prompt": prompt_record["prompt"],
                        "step_index": int(item["step_index"]),
                        "scheduler_timestep": float(item["scheduler_timestep"]),
                        "progress": float(item["progress"]),
                        "pollution_cap": float(item["pollution_cap"]),
                        "rho": rho,
                        "pollution": float(item["pollution"]),
                        "guidance_diversity": float(guidance[0]),
                        "predicted_clean_latent_diversity": float(
                            predicted_clean[0]
                        ),
                        "dz_control_eligible": (
                            float(item["progress"])
                            >= float(
                                reference_config[
                                    "dz_control_start_progress"
                                ]
                            )
                        ),
                        "group_size": candidates,
                        "pool_size": pool_size,
                        "condition_seed": condition_seed,
                        "latent_seeds": latent_seeds,
                    }
                )
            artifact = {
                "config_sha256": config["_config_sha256"],
                "prompt_index": prompt_index,
                "prompt_id": prompt_id,
                "condition_seed": condition_seed,
                "latent_seeds": latent_seeds,
                "elapsed_seconds": elapsed_seconds,
                "unet_calls": stats["unet_calls"],
                "scheduler_step_calls": stats["scheduler_step_calls"],
                "positive_condition_noise_draw_calls": stats[
                    "positive_condition_noise_draw_calls"
                ],
                "final_image_match_count": sum(image_matches),
                "records": records,
            }
            write_json(artifact_path, artifact)
            resumed = False

        all_artifacts.append(artifact)
        all_records.extend(artifact["records"])
        print(
            json.dumps(
                {
                    "completed": prompt_index + 1,
                    "total": len(prompts),
                    "prompt_id": prompt_id,
                    "resumed": resumed,
                }
            ),
            flush=True,
        )

    all_records.sort(key=lambda row: (row["prompt_index"], row["step_index"]))
    per_step_path = output_dir / "per_step.jsonl"
    write_jsonl(per_step_path, all_records)
    curves = build_prompt_median_reference(
        all_records,
        expected_prompt_ids=[record["prompt_id"] for record in prompts],
        expected_steps=steps,
        dz_control_start_progress=float(
            reference_config["dz_control_start_progress"]
        ),
    )
    curves.update(
        {
            "reference_margin_alpha": float(
                reference_config["reference_margin_alpha"]
            ),
            "rho_a_star": rho,
            "aggregation": "prompt-level median at each actual sampling step",
            "guidance_definition": (
                "mean unique-pair (1-cosine)/2 after adaptive 8x8 pooling"
            ),
            "predicted_clean_latent_definition": (
                "mean unique-pair (1-cosine)/2 on DDIM pred_original_sample "
                "after adaptive 8x8 pooling"
            ),
            "config_sha256": config["_config_sha256"],
            "selection_sha256": reference_config["a_star_selection_sha256"],
            "dataset_sha256": config["dataset"]["sha256"],
            "formal_test_data_used": False,
        }
    )
    curve_path = output_dir / "reference_curves.json"
    write_json(curve_path, curves)

    expected_records = len(prompts) * steps
    final_image_match_count = sum(
        int(artifact["final_image_match_count"]) for artifact in all_artifacts
    )
    validation = {
        "config_sha256": config["_config_sha256"],
        "per_step_sha256": sha256(per_step_path),
        "reference_curves_sha256": sha256(curve_path),
        "prompt_count": len(prompts),
        "step_count": steps,
        "record_count": len(all_records),
        "expected_record_count": expected_records,
        "candidate_count": candidates,
        "final_image_match_count": final_image_match_count,
        "expected_final_image_match_count": len(prompts) * candidates,
        "all_frozen_images_match": (
            final_image_match_count == len(prompts) * candidates
        ),
        "unet_calls_per_prompt": sorted(
            {int(artifact["unet_calls"]) for artifact in all_artifacts}
        ),
        "scheduler_step_calls_per_prompt": sorted(
            {
                int(artifact["scheduler_step_calls"])
                for artifact in all_artifacts
            }
        ),
        "extra_unet_calls_for_signals": 0,
        "rho_a_star": rho,
        "reference_margin_alpha": float(
            reference_config["reference_margin_alpha"]
        ),
        "development_split": "COCO-Dev-50",
        "formal_test_data_used": False,
        "passed": (
            len(all_records) == expected_records
            and final_image_match_count == len(prompts) * candidates
            and {int(artifact["unet_calls"]) for artifact in all_artifacts}
            == {steps}
            and {
                int(artifact["scheduler_step_calls"])
                for artifact in all_artifacts
            }
            == {steps}
        ),
    }
    if not validation["passed"]:
        raise RuntimeError("Frozen reference validation failed.")
    write_json(output_dir / "validation_report.json", validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
