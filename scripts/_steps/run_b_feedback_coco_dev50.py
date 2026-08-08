#!/usr/bin/env python3
"""Generate the paired COCO-Dev-50 Stage-B feedback experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
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
from feedback_cads.experiments import (  # noqa: E402
    condition_seed_for_prompt,
    latent_seeds_for_prompt,
    make_initial_latents,
    read_jsonl,
    resolve_recorded_path,
    write_json,
    write_jsonl,
)
from feedback_cads.configuration import load_yaml_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def load_and_validate_config(
    path: Path,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    config, identity = load_yaml_profile(path, profile=profile)
    if config["experiment"]["formal_test_data_used"] is not False:
        raise ValueError("Stage B must use development data only.")
    if config["dataset"]["split"] != "COCO-Dev-50":
        raise ValueError("Stage-B tuning must use COCO-Dev-50.")
    if config["sampling"]["scheduler"] != "DDIM":
        raise ValueError("The frozen protocol requires DDIM.")
    if int(config["sampling"]["num_inference_steps"]) != 50:
        raise ValueError("The frozen protocol requires 50 steps.")
    if int(config["sampling"]["candidates_per_prompt"]) != 8:
        raise ValueError("The frozen protocol requires K=8.")
    if int(config["feedback"]["group_size"]) != 8:
        raise ValueError("Feedback groups must contain K=8 candidates.")
    alpha = float(config["feedback"]["reference_margin_alpha"])
    if alpha not in (0.0, 0.05, 0.10, 0.15, 0.20):
        raise ValueError(
            "Feedback alpha must be one of 0, 0.05, 0.10, 0.15, or 0.20."
        )
    if float(config["feedback"]["initial_rho"]) != float(
        config["paired_a_star"]["rho"]
    ):
        raise ValueError("B must initialize at the frozen A* rho.")

    dataset_path = PROJECT_ROOT / config["dataset"]["path"]
    if sha256(dataset_path) != config["dataset"]["sha256"]:
        raise ValueError("COCO-Dev-50 SHA-256 changed.")
    prompts = read_jsonl(dataset_path)
    if len(prompts) != int(config["dataset"]["expected_prompts"]):
        raise ValueError("COCO-Dev-50 prompt count changed.")
    if len({row["prompt_id"] for row in prompts}) != len(prompts):
        raise ValueError("COCO-Dev-50 prompt IDs must be unique.")
    if any(row.get("split") != "COCO-Dev-50" for row in prompts):
        raise ValueError("Dataset contains a non-development prompt.")

    model_path = PROJECT_ROOT / config["model"]["snapshot"]
    if not model_path.joinpath("model_index.json").is_file():
        raise FileNotFoundError(model_path)
    for section, path_key, hash_key in (
        ("reference", "path", "sha256"),
        ("paired_a_star", "manifest", "manifest_sha256"),
        ("paired_a_star", "per_prompt_metrics", "per_prompt_metrics_sha256"),
        ("paired_a_star", "selection", "selection_sha256"),
    ):
        artifact = PROJECT_ROOT / config[section][path_key]
        if sha256(artifact) != config[section][hash_key]:
            raise ValueError(f"Frozen artifact changed: {artifact}")

    selection = json.loads(
        (PROJECT_ROOT / config["paired_a_star"]["selection"]).read_text(
            encoding="utf-8"
        )
    )
    if selection["formal_test_data_used"]:
        raise ValueError("A* selection unexpectedly used formal test data.")
    if float(selection["selected_a_star_rho"]) != float(
        config["paired_a_star"]["rho"]
    ):
        raise ValueError("Configured A* rho does not match its selection file.")
    if alpha > 0.0:
        prerequisite = config.get("prerequisite_alpha0")
        if not isinstance(prerequisite, dict):
            raise ValueError("Margin ablation requires the alpha=0 audit.")
        prerequisite_path = PROJECT_ROOT / prerequisite["paired_report"]
        if sha256(prerequisite_path) != prerequisite["sha256"]:
            raise ValueError("The alpha=0 paired audit SHA-256 changed.")
        alpha0_report = json.loads(
            prerequisite_path.read_text(encoding="utf-8")
        )
        if alpha0_report["formal_test_data_used"]:
            raise ValueError("The alpha=0 prerequisite used formal test data.")
        conditions = alpha0_report["alpha_margin_ablation_conditions"]
        if not conditions["eligible_for_small_alpha_ablation"]:
            raise ValueError("The alpha=0 result does not permit margin ablation.")
    if alpha > 0.10:
        prerequisite = config.get("prerequisite_stage1_selection")
        if not isinstance(prerequisite, dict):
            raise ValueError("Extended margin search requires stage-1 selection.")
        stage1_path = PROJECT_ROOT / prerequisite["path"]
        if sha256(stage1_path) != prerequisite["sha256"]:
            raise ValueError("The stage-1 alpha selection SHA-256 changed.")
        stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
        if stage1["formal_test_data_used"]:
            raise ValueError("Stage-1 alpha selection used formal test data.")
        if float(stage1["selected_alpha"]) != 0.10:
            raise ValueError("Extended search requires a boundary alpha=0.10.")
        if not stage1["selected_dino_point_improves_a_star"]:
            raise ValueError("Stage-1 boundary did not improve DINO point estimate.")
        extension_rule = config.get("extension_selection_rule")
        if not isinstance(extension_rule, dict):
            raise ValueError("Extended search requires a frozen selection rule.")
        if float(extension_rule["absolute_stop_alpha"]) != 0.20:
            raise ValueError("Extended alpha search must stop at 0.20.")

    config["_prompts"] = prompts
    config["_resolved"] = {
        "config_path": str(path.resolve()),
        "config_profile": identity["profile"],
        "legacy_config_sha256": identity["legacy_sha256"],
        "dataset_path": str(dataset_path.resolve()),
        "model_path": str(model_path.resolve()),
    }
    config["_config_sha256"] = canonical_sha256(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )
    return config


def validate_a_star_pairs(
    config: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    manifest_path = PROJECT_ROOT / config["paired_a_star"]["manifest"]
    rho = float(config["paired_a_star"]["rho"])
    records = [
        row
        for row in read_jsonl(manifest_path)
        if float(row["rho"]) == rho
    ]
    expected = (
        int(config["dataset"]["expected_prompts"])
        * int(config["sampling"]["candidates_per_prompt"])
    )
    if len(records) != expected:
        raise ValueError(f"Expected {expected} frozen A* images.")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in records:
        key = (str(row["prompt_id"]), int(row["candidate_id"]))
        if key in result:
            raise ValueError(f"Duplicate A* pair: {key}")
        image_path = resolve_recorded_path(
            row["image_path"], project_root=PROJECT_ROOT
        )
        if not image_path.is_file() or sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"Frozen A* image changed: {image_path}")
        result[key] = row
    return result


def audit_history(
    history: list[dict[str, Any]],
    *,
    feedback_config: FeedbackMVPConfig,
) -> dict[str, Any]:
    if len(history) != 50:
        raise RuntimeError("Feedback history must contain 50 steps.")
    one_step_delay = all(
        history[index]["rho_used_by_prompt"][0]
        == history[index - 1]["rho_next_by_prompt"][0]
        for index in range(1, len(history))
    )
    pollution_product = all(
        close(
            row["pollution_by_prompt"][0],
            row["pollution_cap"] * row["rho_used_by_prompt"][0],
        )
        for row in history
    )
    hard_off = all(
        row["step_index"] < 30
        or (
            row["pollution_cap"] == 0.0
            and row["pollution_by_prompt"][0] == 0.0
            and row["delta_rho_by_prompt"][0] == 0.0
            and not row["control_updated"]
        )
        for row in history
    )
    dz_window = all(
        row["dz_used"] == (10 <= int(row["step_index"]) < 30)
        for row in history
    )
    proportional = all(
        close(
            row["delta_rho_by_prompt"][0],
            feedback_config.k_diversity
            * row["diversity_deficit_by_prompt"][0],
        )
        and close(
            row["rho_next_by_prompt"][0],
            min(
                1.0,
                max(
                    0.0,
                    row["rho_used_by_prompt"][0]
                    + row["delta_rho_by_prompt"][0],
                ),
            ),
        )
        for row in history[:30]
    )
    checks = {
        "one_step_delay": one_step_delay,
        "pollution_equals_cap_times_rho": pollution_product,
        "proportional_rule_and_clipping": proportional,
        "dz_active_only_on_steps_10_to_29": dz_window,
        "hard_off_exact_from_step_30": hard_off,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Controller history audit failed: {checks}")
    active_rho = np.asarray(
        [row["rho_used_by_prompt"][0] for row in history[:30]],
        dtype=np.float64,
    )
    active_pollution = np.asarray(
        [row["pollution_by_prompt"][0] for row in history[:30]],
        dtype=np.float64,
    )
    return {
        "checks": checks,
        "active_rho_mean": float(active_rho.mean()),
        "active_rho_variance": float(active_rho.var()),
        "active_rho_minimum": float(active_rho.min()),
        "active_rho_maximum": float(active_rho.max()),
        "active_rho_boundary_rate": float(
            np.mean((active_rho == 0.0) | (active_rho == 1.0))
        ),
        "active_pollution_mean": float(active_pollution.mean()),
    }


def main() -> None:
    args = parse_args()
    config = load_and_validate_config(
        args.config.resolve(),
        profile=args.profile,
    )
    prompts = config["_prompts"]
    if args.limit_prompts is not None:
        if args.limit_prompts <= 0:
            raise ValueError("--limit-prompts must be positive.")
        prompts = prompts[: args.limit_prompts]
    a_star_pairs = validate_a_star_pairs(config)

    reference = load_frozen_diversity_reference(
        PROJECT_ROOT / config["reference"]["path"],
        expected_sha256=config["reference"]["sha256"],
    )
    feedback_values = config["feedback"]
    feedback_config = FeedbackMVPConfig(
        initial_rho=float(feedback_values["initial_rho"]),
        k_diversity=float(feedback_values["k_diversity"]),
        epsilon=float(feedback_values["epsilon"]),
        reference_margin_alpha=float(
            feedback_values["reference_margin_alpha"]
        ),
        group_size=int(feedback_values["group_size"]),
        pool_size=int(feedback_values["pool_size"]),
    )
    cads_values = config["cads"]
    cads_config = CleanUnconditionalCADSConfig(
        noise_scale=float(cads_values["noise_scale"]),
        rescale_mix=float(cads_values["rescale_mix"]),
        fixed_rho_a=float(config["paired_a_star"]["rho"]),
        hold_until_progress=float(cads_values["hold_until_progress"]),
        hard_off_progress=float(cads_values["hard_off_progress"]),
        content_tokens_only=bool(cads_values["content_tokens_only"]),
        clean_unconditional=bool(cads_values["clean_unconditional"]),
        rescale=bool(cads_values["rescale"]),
    )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / config["experiment"]["name"]
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    frozen_config = {
        key: value for key, value in config.items() if key != "_prompts"
    }
    if run_config_path.exists() and not args.overwrite:
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing["_config_sha256"] != config["_config_sha256"]:
            raise RuntimeError("Output directory has another frozen config.")
    else:
        write_json(run_config_path, frozen_config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage-B generation.")
    device = torch.device("cuda")
    dtype = torch.float16
    official = StableDiffusionPipeline.from_pretrained(
        Path(config["_resolved"]["model_path"]),
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    official.scheduler = DDIMScheduler.from_config(official.scheduler.config)
    official.to(device=device, dtype=dtype)
    official.set_progress_bar_config(disable=True)
    pipeline = OriginalCADSStableDiffusionPipeline.from_vanilla_pipeline(
        official
    )
    pipeline.set_progress_bar_config(disable=True)

    sampling = config["sampling"]
    resolution = int(config["model"]["resolution"])
    candidates = int(sampling["candidates_per_prompt"])
    alpha = float(feedback_config.reference_margin_alpha)
    alpha_slug = str(alpha).replace(".", "p")
    method_name = (
        "feedback_mvp"
        if alpha == 0.0
        else f"feedback_alpha_{alpha_slug}"
    )
    image_root = output_dir / "images" / "feedback_mvp"
    manifest_records = []
    prompt_summaries = []
    completed = 0
    for prompt_index, prompt_record in enumerate(prompts):
        prompt_id = prompt_record["prompt_id"]
        latent_seeds = latent_seeds_for_prompt(config, prompt_index)
        condition_seed = condition_seed_for_prompt(config, prompt_index)
        for candidate_id, latent_seed in enumerate(latent_seeds):
            a_row = a_star_pairs[(prompt_id, candidate_id)]
            if int(a_row["latent_seed"]) != latent_seed:
                raise RuntimeError("B latent seed is not paired with A*.")
            if int(a_row["condition_seed"]) != condition_seed:
                raise RuntimeError("B condition seed is not paired with A*.")
            if a_row["prompt"] != prompt_record["prompt"]:
                raise RuntimeError("B prompt text is not paired with A*.")

        prompt_dir = image_root / prompt_id
        prompt_dir.mkdir(parents=True, exist_ok=True)
        image_paths = [
            prompt_dir / f"candidate_{index:02d}.png"
            for index in range(candidates)
        ]
        diagnostics_path = prompt_dir / "diagnostics.json"
        complete = diagnostics_path.is_file() and all(
            path.is_file() and path.stat().st_size > 0 for path in image_paths
        )
        if not complete or args.overwrite:
            latents = make_initial_latents(
                latent_seeds,
                device=device,
                dtype=dtype,
                height=resolution,
                width=resolution,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = pipeline(
                prompt=prompt_record["prompt"],
                height=resolution,
                width=resolution,
                num_inference_steps=int(sampling["num_inference_steps"]),
                guidance_scale=float(sampling["cfg_scale"]),
                num_images_per_prompt=candidates,
                eta=float(sampling["eta"]),
                latents=latents,
                output_type="pil",
                clean_unconditional_cads_config=cads_config,
                feedback_mvp_config=feedback_config,
                feedback_reference=reference,
                condition_seed=condition_seed,
            )
            torch.cuda.synchronize()
            elapsed_seconds = time.perf_counter() - started
            stats = dict(pipeline.last_run_stats)
            history = stats["feedback_control_history"]
            controller_summary = audit_history(
                history,
                feedback_config=feedback_config,
            )
            if stats["unet_calls"] != 50 or stats["scheduler_step_calls"] != 50:
                raise RuntimeError("B used an unexpected model call count.")
            if stats["positive_condition_noise_draw_calls"] != 30:
                raise RuntimeError("B condition-noise bank is misaligned.")
            if stats["negative_condition_max_delta"] != 0.0:
                raise RuntimeError("B changed the unconditional branch.")
            if stats["non_content_positive_max_delta"] != 0.0:
                raise RuntimeError("B changed a non-content token.")
            if len(result.images) != candidates:
                raise RuntimeError("B returned an unexpected image count.")
            for image, image_path in zip(result.images, image_paths):
                temporary = image_path.with_suffix(".png.tmp")
                image.save(temporary, format="PNG")
                temporary.replace(image_path)
            write_json(
                diagnostics_path,
                {
                    "config_sha256": config["_config_sha256"],
                    "reference_sha256": reference.sha256,
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_id,
                    "prompt": prompt_record["prompt"],
                    "latent_seeds": latent_seeds,
                    "condition_seed": condition_seed,
                    "elapsed_seconds": elapsed_seconds,
                    "unet_calls": stats["unet_calls"],
                    "scheduler_step_calls": stats["scheduler_step_calls"],
                    "positive_condition_noise_draw_calls": stats[
                        "positive_condition_noise_draw_calls"
                    ],
                    "negative_condition_max_delta": stats[
                        "negative_condition_max_delta"
                    ],
                    "non_content_positive_max_delta": stats[
                        "non_content_positive_max_delta"
                    ],
                    "controller_summary": controller_summary,
                    "controller_history": history,
                },
            )

        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics["config_sha256"] != config["_config_sha256"]:
            raise RuntimeError(f"Stale diagnostics: {diagnostics_path}")
        if diagnostics["reference_sha256"] != reference.sha256:
            raise RuntimeError(f"Stale reference: {diagnostics_path}")
        if diagnostics["latent_seeds"] != latent_seeds:
            raise RuntimeError(f"Stale latent seeds: {diagnostics_path}")
        if int(diagnostics["condition_seed"]) != condition_seed:
            raise RuntimeError(f"Stale condition seed: {diagnostics_path}")
        audit_history(
            diagnostics["controller_history"],
            feedback_config=feedback_config,
        )
        prompt_summaries.append(
            {
                "prompt_index": prompt_index,
                "prompt_id": prompt_id,
                "elapsed_seconds": diagnostics["elapsed_seconds"],
                **diagnostics["controller_summary"],
            }
        )
        for candidate_id, (latent_seed, image_path) in enumerate(
            zip(latent_seeds, image_paths)
        ):
            manifest_records.append(
                {
                    "experiment": config["experiment"]["name"],
                    "method": method_name,
                    "config_sha256": config["_config_sha256"],
                    "reference_sha256": reference.sha256,
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_id,
                    "category": prompt_record["category"],
                    "prompt": prompt_record["prompt"],
                    "candidate_id": candidate_id,
                    "latent_seed": latent_seed,
                    "condition_seed": condition_seed,
                    "image_path": str(image_path.resolve()),
                    "image_sha256": sha256(image_path),
                    "prompt_run_seconds": diagnostics["elapsed_seconds"],
                    "diagnostics_path": str(diagnostics_path.resolve()),
                }
            )
        completed += 1
        print(
            json.dumps(
                {
                    "completed": completed,
                    "total": len(prompts),
                    "prompt_id": prompt_id,
                    "resumed": complete and not args.overwrite,
                }
            ),
            flush=True,
        )

    manifest_records.sort(
        key=lambda row: (int(row["prompt_index"]), int(row["candidate_id"]))
    )
    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_records)
    write_jsonl(output_dir / "per_prompt_controller.jsonl", prompt_summaries)
    expected_images = len(prompts) * candidates
    if len(manifest_records) != expected_images:
        raise RuntimeError("Stage-B manifest is incomplete.")
    write_json(
        output_dir / "generation_complete.json",
        {
            "config_sha256": config["_config_sha256"],
            "manifest_sha256": sha256(manifest_path),
            "reference_sha256": reference.sha256,
            "reference_margin_alpha": alpha,
            "development_split": "COCO-Dev-50",
            "prompt_count": len(prompts),
            "candidates_per_prompt": candidates,
            "image_count": len(manifest_records),
            "a_star_pair_count": len(manifest_records),
            "all_latent_and_condition_seeds_paired_with_a_star": True,
            "unet_calls_per_prompt": 50,
            "formal_test_data_used": False,
        },
    )
    print(f"Wrote {len(manifest_records)} paired B images to {output_dir}")


if __name__ == "__main__":
    main()
