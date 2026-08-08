#!/usr/bin/env python3
"""Generate the frozen, paired stage-A fixed-rho development curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads import (  # noqa: E402
    CleanUnconditionalCADSConfig,
    OriginalCADSStableDiffusionPipeline,
)
from feedback_cads.experiments import (  # noqa: E402
    condition_seed_for_prompt,
    latent_seeds_for_prompt,
    load_curve_config,
    make_initial_latents,
    rho_slug,
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
    parser.add_argument(
        "--profile",
        choices=("dev20", "dev50_confirmation", "coarse", "fine"),
        default="dev20",
        help="Named profile when --config points to a consolidated bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Smoke-test only; omitted for the frozen curve.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = load_curve_config(
        args.config,
        project_root=PROJECT_ROOT,
        profile=args.profile,
    )
    default_output = config["_resolved"]["default_output_dir"]
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT / str(default_output)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = output_dir / "run_config.json"
    frozen_config = {
        key: value for key, value in config.items() if key != "_prompts"
    }
    if run_config_path.exists() and not args.overwrite:
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing["_config_sha256"] != config["_config_sha256"]:
            raise RuntimeError(
                "Output directory contains a different frozen configuration."
            )
    else:
        write_json(run_config_path, frozen_config)

    prompts = config["_prompts"]
    if args.limit_prompts is not None:
        if args.limit_prompts <= 0:
            raise ValueError("--limit-prompts must be positive.")
        prompts = prompts[: args.limit_prompts]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen SD1.5 curve.")
    device = torch.device("cuda")
    dtype = torch.float16
    model_path = Path(config["_resolved"]["model_path"])
    vanilla = StableDiffusionPipeline.from_pretrained(
        model_path,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    vanilla.scheduler = DDIMScheduler.from_config(vanilla.scheduler.config)
    vanilla.to(device=device, dtype=dtype)
    vanilla.set_progress_bar_config(disable=True)
    pipeline = OriginalCADSStableDiffusionPipeline.from_vanilla_pipeline(
        vanilla
    )
    pipeline.set_progress_bar_config(disable=True)

    sampling = config["sampling"]
    cads = config["cads"]
    height = int(config["model"]["resolution"])
    width = height
    candidates = int(sampling["candidates_per_prompt"])
    expected_draw_calls = sum(
        1
        for index in range(int(sampling["num_inference_steps"]))
        if index / (int(sampling["num_inference_steps"]) - 1)
        < float(cads["hard_off_progress"])
    )
    records = []
    total_runs = len(sampling["rho_grid"]) * len(prompts)
    completed_runs = 0

    for rho in (float(value) for value in sampling["rho_grid"]):
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
        rho_dir = output_dir / "images" / rho_slug(rho)
        for prompt_index, prompt_record in enumerate(prompts):
            prompt_id = prompt_record["prompt_id"]
            prompt_dir = rho_dir / prompt_id
            prompt_dir.mkdir(parents=True, exist_ok=True)
            image_paths = [
                prompt_dir / f"candidate_{index:02d}.png"
                for index in range(candidates)
            ]
            diagnostics_path = prompt_dir / "diagnostics.json"
            complete = diagnostics_path.exists() and all(
                path.exists() and path.stat().st_size > 0
                for path in image_paths
            )

            latent_seeds = latent_seeds_for_prompt(config, prompt_index)
            condition_seed = condition_seed_for_prompt(config, prompt_index)
            if not complete or args.overwrite:
                latents = make_initial_latents(
                    latent_seeds,
                    device=device,
                    dtype=dtype,
                    height=height,
                    width=width,
                )
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = pipeline(
                    prompt=prompt_record["prompt"],
                    height=height,
                    width=width,
                    num_inference_steps=int(sampling["num_inference_steps"]),
                    guidance_scale=float(sampling["cfg_scale"]),
                    num_images_per_prompt=candidates,
                    eta=float(sampling["eta"]),
                    latents=latents,
                    output_type="pil",
                    clean_unconditional_cads_config=stage_a_config,
                    condition_seed=condition_seed,
                )
                torch.cuda.synchronize()
                elapsed_seconds = time.perf_counter() - started
                stats = dict(pipeline.last_run_stats)

                if stats["actual_timesteps"] != int(
                    sampling["num_inference_steps"]
                ):
                    raise RuntimeError("Unexpected number of sampling steps.")
                if stats["negative_condition_max_delta"] != 0.0:
                    raise RuntimeError("A changed the unconditional branch.")
                if stats["non_content_positive_max_delta"] != 0.0:
                    raise RuntimeError("A changed a masked-out token.")
                expected_draws = 0 if rho == 0.0 else expected_draw_calls
                if stats["positive_condition_noise_draw_calls"] != expected_draws:
                    raise RuntimeError("Unexpected condition-noise draw count.")
                if any(value != 0.0 for value in stats["pollution_history"][-20:]):
                    raise RuntimeError("The final 40% was not exactly clean.")
                if len(result.images) != candidates:
                    raise RuntimeError("Pipeline returned the wrong batch size.")

                for image, final_path in zip(result.images, image_paths):
                    temporary = final_path.with_suffix(".png.tmp")
                    image.save(temporary, format="PNG")
                    temporary.replace(final_path)
                write_json(
                    diagnostics_path,
                    {
                        "prompt_id": prompt_id,
                        "rho": rho,
                        "latent_seeds": latent_seeds,
                        "condition_seed": condition_seed,
                        "elapsed_seconds": elapsed_seconds,
                        "pipeline_stats": stats,
                    },
                )

            diagnostics = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )
            for candidate_index, (latent_seed, image_path) in enumerate(
                zip(latent_seeds, image_paths)
            ):
                records.append(
                    {
                        "experiment": config["experiment"]["name"],
                        "config_sha256": config["_config_sha256"],
                        "rho": rho,
                        "prompt_index": prompt_index,
                        "prompt_id": prompt_id,
                        "category": prompt_record["category"],
                        "prompt": prompt_record["prompt"],
                        "candidate_id": candidate_index,
                        "latent_seed": latent_seed,
                        "condition_seed": condition_seed,
                        "image_path": str(image_path.resolve()),
                        "image_sha256": file_sha256(image_path),
                        "prompt_run_seconds": diagnostics["elapsed_seconds"],
                    }
                )
            completed_runs += 1
            print(
                json.dumps(
                    {
                        "completed": completed_runs,
                        "total": total_runs,
                        "rho": rho,
                        "prompt_id": prompt_id,
                        "resumed": complete and not args.overwrite,
                    }
                ),
                flush=True,
            )

    records.sort(
        key=lambda row: (
            float(row["rho"]),
            int(row["prompt_index"]),
            int(row["candidate_id"]),
        )
    )
    write_jsonl(output_dir / "manifest.jsonl", records)
    expected_records = len(prompts) * len(sampling["rho_grid"]) * candidates
    if len(records) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} manifest rows, found {len(records)}."
        )
    write_json(
        output_dir / "generation_complete.json",
        {
            "config_sha256": config["_config_sha256"],
            "prompt_count": len(prompts),
            "rho_count": len(sampling["rho_grid"]),
            "candidates_per_prompt": candidates,
            "image_count": len(records),
            "formal_test_data_used": False,
        },
    )
    print(f"Wrote {len(records)} paired images to {output_dir}")


if __name__ == "__main__":
    main()
