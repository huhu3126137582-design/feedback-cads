#!/usr/bin/env python3
"""Run the audited four-method generator in Dev smoke or formal mode."""

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
import yaml
from diffusers import DDIMScheduler, StableDiffusionPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "_steps"))

from feedback_cads import (  # noqa: E402
    CleanUnconditionalCADSConfig,
    FeedbackMVPConfig,
    OriginalCADSStableDiffusionPipeline,
    PaperCADSConfig,
    load_frozen_diversity_reference,
)
from feedback_cads.condition_noise import (  # noqa: E402
    negative_seed_from_positive,
)
from feedback_cads.experiments import (  # noqa: E402
    condition_seed_for_prompt,
    latent_seeds_for_prompt,
    make_initial_latents,
    read_jsonl,
    write_json,
    write_jsonl,
)
from validate_formal_test_protocol import validate_protocol  # noqa: E402


FROZEN_FORMAL_PROTOCOL_SHA256 = (
    "9d19e02e284aefe0ed58fdc5a47dd0a0fec0966936a40dd9e281d2c97d0161ef"
)
FORMAL_PROTOCOL_PATH = PROJECT_ROOT / "configs/formal_test_coco_test500.yaml"
SMOKE_CONFIG_PATH = (
    PROJECT_ROOT / "configs/formal_generation_smoke_coco_dev50.yaml"
)
RUNNER_SCHEMA = "feedback-cads-four-method-generation-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _checked_file(path_value: str, expected_sha256: str) -> Path:
    path = (PROJECT_ROOT / path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != expected_sha256:
        raise ValueError(f"Frozen artifact hash changed: {path}")
    return path


def _base_protocol() -> dict[str, Any]:
    if sha256(FORMAL_PROTOCOL_PATH) != FROZEN_FORMAL_PROTOCOL_SHA256:
        raise ValueError("The frozen formal protocol SHA-256 changed.")
    validate_protocol(FORMAL_PROTOCOL_PATH, require_unstarted=False)
    return _load_yaml(FORMAL_PROTOCOL_PATH)


def load_run_spec(
    mode: str,
    *,
    smoke_config_path: Path = SMOKE_CONFIG_PATH,
    confirm_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Create an immutable runtime spec without loading the diffusion model."""

    protocol = _base_protocol()
    if mode == "formal":
        if confirm_protocol_sha256 != FROZEN_FORMAL_PROTOCOL_SHA256:
            raise ValueError(
                "Formal generation requires --confirm-protocol-sha256 "
                f"{FROZEN_FORMAL_PROTOCOL_SHA256}."
            )
        dataset_path = _checked_file(
            protocol["dataset"]["path"], protocol["dataset"]["sha256"]
        )
        prompts = read_jsonl(dataset_path)
        selected_indices = list(range(int(protocol["dataset"]["expected_prompts"])))
        randomness = dict(protocol["randomness"])
        output_root = (PROJECT_ROOT / protocol["generation"]["output_root"]).resolve()
        expected_total = int(protocol["generation"]["expected_total_images"])
        expected_per_method = int(
            protocol["generation"]["expected_images_per_method"]
        )
        source_config_sha = FROZEN_FORMAL_PROTOCOL_SHA256
        formal_test_data_used = True
    elif mode == "smoke":
        smoke_config_path = smoke_config_path.resolve()
        smoke = _load_yaml(smoke_config_path)
        formal_link = smoke["formal_protocol"]
        if (
            (PROJECT_ROOT / formal_link["path"]).resolve()
            != FORMAL_PROTOCOL_PATH.resolve()
            or formal_link["sha256"] != FROZEN_FORMAL_PROTOCOL_SHA256
        ):
            raise ValueError("Smoke run is not linked to the frozen protocol.")
        dataset = smoke["dataset"]
        dataset_path = _checked_file(dataset["path"], dataset["sha256"])
        source_prompts = read_jsonl(dataset_path)
        if len(source_prompts) != int(dataset["expected_source_prompts"]):
            raise ValueError("The Dev-50 source prompt count changed.")
        if {row.get("split") for row in source_prompts} != {
            dataset["expected_split"]
        }:
            raise ValueError("Smoke data must come only from COCO-Dev-50.")
        selected_indices = [int(value) for value in dataset["selected_prompt_indices"]]
        if (
            not selected_indices
            or len(set(selected_indices)) != len(selected_indices)
            or min(selected_indices) < 0
            or max(selected_indices) >= len(source_prompts)
        ):
            raise ValueError("Invalid smoke prompt indices.")
        prompts = source_prompts
        randomness = dict(smoke["randomness"])
        output_root = (PROJECT_ROOT / smoke["generation"]["output_root"]).resolve()
        formal_output = (
            PROJECT_ROOT / protocol["generation"]["output_root"]
        ).resolve()
        if output_root == formal_output:
            raise ValueError(
                "Smoke validation must not write into the formal output."
            )
        if smoke["generation"].get("formal_test_data_used") is not False:
            raise ValueError("Smoke run must explicitly exclude formal data.")
        expected_total = int(smoke["generation"]["expected_total_images"])
        expected_per_method = int(
            smoke["generation"]["expected_images_per_method"]
        )
        source_config_sha = sha256(smoke_config_path)
        formal_test_data_used = False
    else:
        raise ValueError("mode must be 'smoke' or 'formal'.")

    selected_prompts = [prompts[index] for index in selected_indices]
    sampling = dict(protocol["sampling"])
    method_order = list(sampling["method_order"])
    candidate_count = int(sampling["candidates_per_prompt"])
    if len(selected_prompts) * candidate_count != expected_per_method:
        raise ValueError("Per-method image count does not match selected prompts.")
    if expected_per_method * len(method_order) != expected_total:
        raise ValueError("Total image count does not match the four methods.")

    method_artifacts: dict[str, dict[str, str]] = {}
    for method in ("original_cads", "a_star", "b_feedback"):
        section = protocol["methods"][method]
        path = _checked_file(section["config_path"], section["config_sha256"])
        method_artifacts[method] = {
            "path": str(path),
            "sha256": section["config_sha256"],
        }
    b_section = protocol["methods"]["b_feedback"]
    reference_path = _checked_file(
        b_section["reference_path"], b_section["reference_sha256"]
    )
    selection_artifacts = {
        "a_star": {
            "path": str(
                _checked_file(
                    protocol["methods"]["a_star"]["selection_path"],
                    protocol["methods"]["a_star"]["selection_sha256"],
                )
            ),
            "sha256": protocol["methods"]["a_star"]["selection_sha256"],
        },
        "b_feedback": {
            "path": str(
                _checked_file(
                    b_section["selection_path"],
                    b_section["selection_sha256"],
                )
            ),
            "sha256": b_section["selection_sha256"],
        },
        "reference": {
            "path": str(reference_path),
            "sha256": b_section["reference_sha256"],
        },
    }
    model_path = (PROJECT_ROOT / protocol["model"]["snapshot"]).resolve()
    if not model_path.joinpath("model_index.json").is_file():
        raise FileNotFoundError(model_path)

    seed_config = {
        "sampling": sampling,
        "randomness": randomness,
    }
    spec = {
        "runner_schema": RUNNER_SCHEMA,
        "mode": mode,
        "formal_test_data_used": formal_test_data_used,
        "formal_protocol_path": str(FORMAL_PROTOCOL_PATH.resolve()),
        "formal_protocol_sha256": FROZEN_FORMAL_PROTOCOL_SHA256,
        "source_config_sha256": source_config_sha,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "selected_prompt_indices": selected_indices,
        "prompts": selected_prompts,
        "model_path": str(model_path),
        "sampling": sampling,
        "randomness": randomness,
        "seed_config": seed_config,
        "method_order": method_order,
        "method_artifacts": method_artifacts,
        "selection_artifacts": selection_artifacts,
        "output_root": str(output_root),
        "expected_images_per_method": expected_per_method,
        "expected_total_images": expected_total,
    }
    serializable = {key: value for key, value in spec.items() if key != "prompts"}
    serializable["prompt_identity"] = [
        {
            "prompt_index": index,
            "prompt_id": prompt["prompt_id"],
            "prompt": prompt["prompt"],
        }
        for index, prompt in zip(selected_indices, selected_prompts)
    ]
    spec["run_spec_sha256"] = canonical_sha256(serializable)
    return spec


def _paper_config(spec: dict[str, Any]) -> PaperCADSConfig:
    values = _load_yaml(Path(spec["method_artifacts"]["original_cads"]["path"]))
    cads = values["cads"]
    conventions = values["implementation_conventions"]
    return PaperCADSConfig(
        tau1=float(cads["tau1"]),
        tau2=float(cads["tau2"]),
        noise_scale=float(cads["noise_scale"]),
        rescale_mix=float(cads["rescale_mix_psi"]),
        rescale=bool(cads["rescale"]),
        eps=float(conventions["denominator_floor_epsilon"]),
    )


def _a_config(spec: dict[str, Any]) -> CleanUnconditionalCADSConfig:
    values = _load_yaml(Path(spec["method_artifacts"]["a_star"]["path"]))["cads"]
    return CleanUnconditionalCADSConfig(
        noise_scale=float(values["noise_scale"]),
        rescale_mix=float(values["rescale_mix"]),
        fixed_rho_a=float(values["fixed_rho_a"]),
        hold_until_progress=float(values["hold_until_progress"]),
        hard_off_progress=float(values["hard_off_progress"]),
        content_tokens_only=bool(values["content_tokens_only"]),
        clean_unconditional=bool(values["clean_unconditional"]),
        rescale=bool(values["rescale"]),
    )


def _b_config(spec: dict[str, Any]) -> tuple[CleanUnconditionalCADSConfig, FeedbackMVPConfig]:
    values = _load_yaml(Path(spec["method_artifacts"]["b_feedback"]["path"]))
    cads = values["cads"]
    feedback = values["feedback"]
    a_config = CleanUnconditionalCADSConfig(
        noise_scale=float(cads["noise_scale"]),
        rescale_mix=float(cads["rescale_mix"]),
        fixed_rho_a=float(feedback["initial_rho"]),
        hold_until_progress=float(cads["hold_until_progress"]),
        hard_off_progress=float(cads["hard_off_progress"]),
        content_tokens_only=bool(cads["content_tokens_only"]),
        clean_unconditional=bool(cads["clean_unconditional"]),
        rescale=bool(cads["rescale"]),
    )
    feedback_config = FeedbackMVPConfig(
        initial_rho=float(feedback["initial_rho"]),
        k_diversity=float(feedback["k_diversity"]),
        epsilon=float(feedback["epsilon"]),
        reference_margin_alpha=float(feedback["reference_margin_alpha"]),
        group_size=int(feedback["group_size"]),
        pool_size=int(feedback["pool_size"]),
    )
    return a_config, feedback_config


def _close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def audit_feedback_history(
    history: list[dict[str, Any]],
    config: FeedbackMVPConfig,
) -> dict[str, Any]:
    if len(history) != 50:
        raise RuntimeError("B feedback history must contain exactly 50 steps.")
    checks = {
        "one_step_delay": all(
            history[index]["rho_used_by_prompt"][0]
            == history[index - 1]["rho_next_by_prompt"][0]
            for index in range(1, 50)
        ),
        "pollution_equals_cap_times_rho": all(
            _close(
                row["pollution_by_prompt"][0],
                row["pollution_cap"] * row["rho_used_by_prompt"][0],
            )
            for row in history
        ),
        "proportional_rule_and_clipping": all(
            _close(
                row["delta_rho_by_prompt"][0],
                config.k_diversity * row["diversity_deficit_by_prompt"][0],
            )
            and _close(
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
        ),
        "dz_active_only_on_steps_10_to_29": all(
            row["dz_used"] == (10 <= int(row["step_index"]) < 30)
            for row in history
        ),
        "hard_off_exact_from_step_30": all(
            row["step_index"] < 30
            or (
                row["pollution_cap"] == 0.0
                and row["pollution_by_prompt"][0] == 0.0
                and row["delta_rho_by_prompt"][0] == 0.0
                and not row["control_updated"]
            )
            for row in history
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"B controller audit failed: {checks}")
    active_rho = np.asarray(
        [row["rho_used_by_prompt"][0] for row in history[:30]],
        dtype=np.float64,
    )
    return {
        "checks": checks,
        "active_rho_mean": float(active_rho.mean()),
        "active_rho_variance": float(active_rho.var()),
        "active_rho_boundary_rate": float(
            np.mean((active_rho == 0.0) | (active_rho == 1.0))
        ),
    }


def audit_method_stats(
    method: str,
    stats: dict[str, Any],
    *,
    a_config: CleanUnconditionalCADSConfig,
    feedback_config: FeedbackMVPConfig,
) -> dict[str, Any]:
    if (
        stats["actual_timesteps"] != 50
        or stats["unet_calls"] != 50
        or stats["main_unet_calls"] != 50
        or stats["scheduler_step_calls"] != 50
    ):
        raise RuntimeError(f"{method} did not use exactly 50 denoising calls.")
    if stats["grad_enabled_during_loop"]:
        raise RuntimeError(f"{method} unexpectedly enabled gradients.")

    if method == "vanilla":
        checks = {
            "mode": stats["mode"] == "vanilla",
            "no_condition_noise": stats["positive_condition_noise_draw_calls"] == 0
            and stats["negative_condition_noise_draw_calls"] == 0,
            "no_condition_seed_consumed": stats["condition_seed"] is None,
            "no_feedback": not stats["feedback_control_history"],
        }
    elif method == "original_cads":
        active_steps = sum(float(value) > 0.0 for value in stats["pollution_history"])
        checks = {
            "mode": stats["mode"] == "original_cads",
            "positive_and_negative_independent": stats["condition_seed"]
            != stats["negative_condition_seed"],
            "both_branches_perturbed": stats["positive_condition_max_delta"] > 0.0
            and stats["negative_condition_max_delta"] > 0.0,
            "draw_count": stats["positive_condition_noise_draw_calls"]
            == active_steps
            and stats["negative_condition_noise_draw_calls"] == active_steps,
            "correct_time_direction": stats["pollution_history"][0] == 1.0
            and stats["pollution_history"][-1] == 0.0
            and all(
                left >= right
                for left, right in zip(
                    stats["pollution_history"], stats["pollution_history"][1:]
                )
            ),
        }
    elif method in ("a_star", "b_feedback"):
        def pollution_is_zero(value: Any) -> bool:
            if isinstance(value, list):
                return all(float(item) == 0.0 for item in value)
            return float(value) == 0.0

        checks = {
            "mode": stats["mode"]
            == ("clean_unconditional_cads" if method == "a_star" else "feedback_mvp"),
            "clean_unconditional": stats["negative_condition_max_delta"] == 0.0
            and stats["negative_condition_noise_draw_calls"] == 0,
            "content_tokens_only": stats["non_content_positive_max_delta"] == 0.0,
            "draw_count": stats["positive_condition_noise_draw_calls"] == 30,
            "hard_off": all(
                pollution_is_zero(value)
                for value in stats["pollution_history"][-20:]
            ),
        }
        if method == "a_star":
            checks["frozen_rho"] = _close(stats["fixed_rho_a"], a_config.fixed_rho_a)
            checks["no_feedback"] = not stats["feedback_control_history"]
        else:
            checks["frozen_alpha"] = _close(
                stats["feedback_reference_margin_alpha"],
                feedback_config.reference_margin_alpha,
            )
            audit_feedback_history(stats["feedback_control_history"], feedback_config)
    else:
        raise ValueError(f"Unknown method: {method}")
    if not all(checks.values()):
        raise RuntimeError(f"{method} audit failed: {checks}")
    return checks


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.random.get_rng_state().clone(), [
        state.clone() for state in torch.cuda.get_rng_state_all()
    ]


def _rng_unchanged(
    before: tuple[torch.Tensor, list[torch.Tensor]],
    after: tuple[torch.Tensor, list[torch.Tensor]],
) -> bool:
    return torch.equal(before[0], after[0]) and len(before[1]) == len(after[1]) and all(
        torch.equal(left, right) for left, right in zip(before[1], after[1])
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _runtime_lock(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "runner_schema": RUNNER_SCHEMA,
        "mode": spec["mode"],
        "formal_test_data_used": spec["formal_test_data_used"],
        "formal_protocol_sha256": spec["formal_protocol_sha256"],
        "source_config_sha256": spec["source_config_sha256"],
        "run_spec_sha256": spec["run_spec_sha256"],
        "dataset_sha256": spec["dataset_sha256"],
        "selected_prompt_indices": spec["selected_prompt_indices"],
        "method_order": spec["method_order"],
        "method_artifacts": spec["method_artifacts"],
        "selection_artifacts": spec["selection_artifacts"],
        "sampling": spec["sampling"],
        "randomness": spec["randomness"],
        "expected_images_per_method": spec["expected_images_per_method"],
        "expected_total_images": spec["expected_total_images"],
        "quality_metrics_computed": False,
    }


def _prepare_output(spec: dict[str, Any]) -> Path:
    output_root = Path(spec["output_root"])
    lock = _runtime_lock(spec)
    lock_path = output_root / "run_lock.json"
    if output_root.exists():
        if not lock_path.is_file():
            raise RuntimeError("Output exists without an immutable run lock.")
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing != lock:
            raise RuntimeError("Output belongs to a different run specification.")
    else:
        output_root.mkdir(parents=True)
        write_json(lock_path, lock)
    return output_root


def _method_kwargs(
    method: str,
    *,
    paper_config: PaperCADSConfig,
    a_config: CleanUnconditionalCADSConfig,
    b_cads_config: CleanUnconditionalCADSConfig,
    feedback_config: FeedbackMVPConfig,
    reference: Any,
    condition_seed: int,
) -> dict[str, Any]:
    if method == "vanilla":
        return {}
    if method == "original_cads":
        return {
            "original_cads_config": paper_config,
            "condition_seed": condition_seed,
            "negative_condition_seed": negative_seed_from_positive(condition_seed),
        }
    if method == "a_star":
        return {
            "clean_unconditional_cads_config": a_config,
            "condition_seed": condition_seed,
        }
    if method == "b_feedback":
        return {
            "clean_unconditional_cads_config": b_cads_config,
            "feedback_mvp_config": feedback_config,
            "feedback_reference": reference,
            "condition_seed": condition_seed,
        }
    raise ValueError(f"Unknown method: {method}")


def run_generation(spec: dict[str, Any]) -> dict[str, Any]:
    """Generate every declared image and validate diagnostics before completion."""

    output_root = _prepare_output(spec)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SD1.5 generation.")
    device = torch.device("cuda")
    dtype = torch.float16
    official = StableDiffusionPipeline.from_pretrained(
        Path(spec["model_path"]),
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    official.scheduler = DDIMScheduler.from_config(official.scheduler.config)
    official.to(device=device, dtype=dtype)
    official.set_progress_bar_config(disable=True)
    pipeline = OriginalCADSStableDiffusionPipeline.from_vanilla_pipeline(official)
    pipeline.set_progress_bar_config(disable=True)

    paper_config = _paper_config(spec)
    a_config = _a_config(spec)
    b_cads_config, feedback_config = _b_config(spec)
    reference = load_frozen_diversity_reference(
        Path(spec["selection_artifacts"]["reference"]["path"]),
        expected_sha256=spec["selection_artifacts"]["reference"]["sha256"],
    )
    if not _close(a_config.fixed_rho_a, 0.55):
        raise RuntimeError("A* rho drifted from 0.55.")
    if not _close(feedback_config.reference_margin_alpha, 0.15):
        raise RuntimeError("B alpha drifted from 0.15.")

    sampling = spec["sampling"]
    candidates = int(sampling["candidates_per_prompt"])
    resolution = 512
    records: list[dict[str, Any]] = []
    prompt_audits: list[dict[str, Any]] = []
    total_runs = len(spec["prompts"]) * len(spec["method_order"])
    completed_runs = 0
    latent_hashes: dict[tuple[int, str], str] = {}
    condition_seeds: dict[tuple[int, str], int | None] = {}

    for method in spec["method_order"]:
        method_root = output_root / "images" / method
        for local_index, (source_index, prompt_record) in enumerate(
            zip(spec["selected_prompt_indices"], spec["prompts"])
        ):
            del local_index
            prompt_id = str(prompt_record["prompt_id"])
            latent_seeds = latent_seeds_for_prompt(spec["seed_config"], source_index)
            condition_seed = condition_seed_for_prompt(
                spec["seed_config"], source_index
            )
            prompt_dir = method_root / prompt_id
            image_paths = [
                prompt_dir / f"candidate_{candidate_id:02d}.png"
                for candidate_id in range(candidates)
            ]
            diagnostics_path = prompt_dir / "diagnostics.json"
            complete = diagnostics_path.is_file() and all(
                path.is_file() and path.stat().st_size > 0 for path in image_paths
            )
            if not complete:
                prompt_dir.mkdir(parents=True, exist_ok=True)
                latents = make_initial_latents(
                    latent_seeds,
                    device=device,
                    dtype=dtype,
                    height=resolution,
                    width=resolution,
                )
                initial_latents_sha256 = _tensor_sha256(latents)
                rng_before = _rng_state()
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
                    **_method_kwargs(
                        method,
                        paper_config=paper_config,
                        a_config=a_config,
                        b_cads_config=b_cads_config,
                        feedback_config=feedback_config,
                        reference=reference,
                        condition_seed=condition_seed,
                    ),
                )
                torch.cuda.synchronize()
                elapsed_seconds = time.perf_counter() - started
                rng_after = _rng_state()
                if not _rng_unchanged(rng_before, rng_after):
                    raise RuntimeError(f"{method} mutated a global Torch RNG state.")
                stats = dict(pipeline.last_run_stats)
                checks = audit_method_stats(
                    method,
                    stats,
                    a_config=a_config,
                    feedback_config=feedback_config,
                )
                if len(result.images) != candidates:
                    raise RuntimeError(f"{method} returned the wrong image count.")
                for image, image_path in zip(result.images, image_paths):
                    temporary = image_path.with_suffix(".png.tmp")
                    image.save(temporary, format="PNG")
                    temporary.replace(image_path)
                diagnostics = {
                    "runner_schema": RUNNER_SCHEMA,
                    "run_spec_sha256": spec["run_spec_sha256"],
                    "formal_protocol_sha256": spec["formal_protocol_sha256"],
                    "dataset_sha256": spec["dataset_sha256"],
                    "formal_test_data_used": spec["formal_test_data_used"],
                    "method": method,
                    "prompt_source_index": source_index,
                    "prompt_id": prompt_id,
                    "prompt": prompt_record["prompt"],
                    "latent_seeds": latent_seeds,
                    "initial_latents_sha256": initial_latents_sha256,
                    "condition_seed": None if method == "vanilla" else condition_seed,
                    "negative_condition_seed": (
                        negative_seed_from_positive(condition_seed)
                        if method == "original_cads"
                        else None
                    ),
                    "elapsed_seconds": elapsed_seconds,
                    "global_torch_rng_unchanged": True,
                    "audit_checks": checks,
                    "pipeline_stats": stats,
                }
                write_json(diagnostics_path, diagnostics)
            else:
                diagnostics = json.loads(
                    diagnostics_path.read_text(encoding="utf-8")
                )

            expected_condition_seed = None if method == "vanilla" else condition_seed
            required_diagnostics = {
                "runner_schema": RUNNER_SCHEMA,
                "run_spec_sha256": spec["run_spec_sha256"],
                "formal_protocol_sha256": spec["formal_protocol_sha256"],
                "dataset_sha256": spec["dataset_sha256"],
                "formal_test_data_used": spec["formal_test_data_used"],
                "method": method,
                "prompt_source_index": source_index,
                "prompt_id": prompt_id,
                "prompt": prompt_record["prompt"],
                "latent_seeds": latent_seeds,
                "condition_seed": expected_condition_seed,
                "global_torch_rng_unchanged": True,
            }
            for key, expected in required_diagnostics.items():
                if diagnostics.get(key) != expected:
                    raise RuntimeError(f"Stale diagnostics field {key}: {diagnostics_path}")
            audit_method_stats(
                method,
                diagnostics["pipeline_stats"],
                a_config=a_config,
                feedback_config=feedback_config,
            )
            latent_hashes[(source_index, method)] = diagnostics[
                "initial_latents_sha256"
            ]
            condition_seeds[(source_index, method)] = diagnostics["condition_seed"]
            prompt_audits.append(
                {
                    "method": method,
                    "prompt_source_index": source_index,
                    "prompt_id": prompt_id,
                    "initial_latents_sha256": diagnostics[
                        "initial_latents_sha256"
                    ],
                    "condition_seed": diagnostics["condition_seed"],
                    "negative_condition_seed": diagnostics[
                        "negative_condition_seed"
                    ],
                    "global_torch_rng_unchanged": True,
                    "unet_calls": diagnostics["pipeline_stats"]["unet_calls"],
                    "elapsed_seconds": diagnostics["elapsed_seconds"],
                }
            )
            for candidate_id, (latent_seed, image_path) in enumerate(
                zip(latent_seeds, image_paths)
            ):
                records.append(
                    {
                        "runner_schema": RUNNER_SCHEMA,
                        "run_spec_sha256": spec["run_spec_sha256"],
                        "formal_protocol_sha256": spec["formal_protocol_sha256"],
                        "formal_test_data_used": spec["formal_test_data_used"],
                        "method": method,
                        "prompt_source_index": source_index,
                        "prompt_id": prompt_id,
                        "category": prompt_record["category"],
                        "prompt": prompt_record["prompt"],
                        "candidate_id": candidate_id,
                        "latent_seed": latent_seed,
                        "condition_seed": expected_condition_seed,
                        "image_path": str(image_path.resolve()),
                        "image_sha256": sha256(image_path),
                        "prompt_run_seconds": diagnostics["elapsed_seconds"],
                        "diagnostics_path": str(diagnostics_path.resolve()),
                    }
                )
            completed_runs += 1
            print(
                json.dumps(
                    {
                        "completed_method_prompts": completed_runs,
                        "total_method_prompts": total_runs,
                        "method": method,
                        "prompt_id": prompt_id,
                        "resumed": complete,
                    }
                ),
                flush=True,
            )

    paired_latents = all(
        len({latent_hashes[(index, method)] for method in spec["method_order"]}) == 1
        for index in spec["selected_prompt_indices"]
    )
    paired_positive_conditions = all(
        len(
            {
                condition_seeds[(index, method)]
                for method in ("original_cads", "a_star", "b_feedback")
            }
        )
        == 1
        and condition_seeds[(index, "vanilla")] is None
        for index in spec["selected_prompt_indices"]
    )
    if not paired_latents or not paired_positive_conditions:
        raise RuntimeError("Four-method seed pairing audit failed.")
    if len(records) != spec["expected_total_images"]:
        raise RuntimeError("The completed manifest has the wrong image count.")
    if len({row["image_path"] for row in records}) != len(records):
        raise RuntimeError("Manifest image paths are not unique.")

    records.sort(
        key=lambda row: (
            spec["method_order"].index(row["method"]),
            int(row["prompt_source_index"]),
            int(row["candidate_id"]),
        )
    )
    manifest_path = output_root / "manifest.jsonl"
    write_jsonl(manifest_path, records)
    write_jsonl(output_root / "per_prompt_generation_audit.jsonl", prompt_audits)
    completion = {
        "runner_schema": RUNNER_SCHEMA,
        "run_spec_sha256": spec["run_spec_sha256"],
        "formal_protocol_sha256": spec["formal_protocol_sha256"],
        "mode": spec["mode"],
        "formal_test_data_used": spec["formal_test_data_used"],
        "quality_metrics_computed": False,
        "method_order": spec["method_order"],
        "prompt_count": len(spec["prompts"]),
        "candidates_per_prompt": candidates,
        "image_count": len(records),
        "images_per_method": spec["expected_images_per_method"],
        "manifest_sha256": sha256(manifest_path),
        "all_methods_share_prompt_candidate_latents": paired_latents,
        "all_cads_methods_share_positive_condition_seeds": paired_positive_conditions,
        "vanilla_consumed_no_condition_seed": True,
        "all_global_torch_rng_checks_passed": True,
        "all_runs_used_exactly_50_unet_calls": all(
            row["unet_calls"] == 50 for row in prompt_audits
        ),
        "formal_output_created": spec["mode"] == "formal",
    }
    write_json(output_root / "generation_complete.json", completion)
    return completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument(
        "--smoke-config",
        type=Path,
        default=SMOKE_CONFIG_PATH,
        help="Used only in smoke mode; formal mode has no dataset override.",
    )
    parser.add_argument(
        "--confirm-protocol-sha256",
        default=None,
        help="Required in formal mode to prevent accidental Test500 startup.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the run spec without creating an output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and args.smoke_config != SMOKE_CONFIG_PATH:
        raise ValueError("Formal mode does not accept --smoke-config.")
    spec = load_run_spec(
        args.mode,
        smoke_config_path=args.smoke_config,
        confirm_protocol_sha256=args.confirm_protocol_sha256,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "validated": True,
                    "mode": spec["mode"],
                    "run_spec_sha256": spec["run_spec_sha256"],
                    "formal_protocol_sha256": spec[
                        "formal_protocol_sha256"
                    ],
                    "prompt_count": len(spec["prompts"]),
                    "expected_total_images": spec["expected_total_images"],
                    "output_root": spec["output_root"],
                },
                indent=2,
            )
        )
        return
    result = run_generation(spec)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
