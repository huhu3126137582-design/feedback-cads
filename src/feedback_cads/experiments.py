"""Frozen experiment helpers for paired Feedback-CADS evaluations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .configuration import load_yaml_profile


STAGE_A_RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PROJECT_TOP_LEVELS = ("outputs", "data", "configs", "models", "results")


def resolve_recorded_path(
    value: str | Path,
    *,
    project_root: Path,
) -> Path:
    """Resolve a path stored by an earlier checkout of this project.

    Frozen manifests remain byte-for-byte immutable and therefore retain the
    absolute paths recorded on the generation machine. If such a path no
    longer exists, rebase its project-relative suffix onto the current
    checkout without modifying the archived record.
    """

    path = Path(value).expanduser()
    root = project_root.resolve()
    if not path.is_absolute():
        return (root / path).resolve()
    if path.exists():
        return path.resolve()
    parts = path.parts
    for marker in PROJECT_TOP_LEVELS:
        try:
            marker_index = parts.index(marker)
        except ValueError:
            continue
        candidate = root.joinpath(*parts[marker_index:]).resolve()
        if candidate.exists():
            return candidate
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}."
                ) from error
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_curve_config(
    path: Path,
    *,
    project_root: Path,
    profile: str | None = None,
) -> dict[str, Any]:
    config, identity = load_yaml_profile(path, profile=profile)

    dataset_path = project_root / config["dataset"]["path"]
    model_path = project_root / config["model"]["snapshot"]
    expected_dataset_sha = config["dataset"].get("sha256")
    if expected_dataset_sha is not None:
        actual_dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if actual_dataset_sha != expected_dataset_sha:
            raise ValueError(
                "Dataset SHA-256 does not match the frozen configuration."
            )
    prompts = read_jsonl(dataset_path)
    expected_prompts = int(config["dataset"]["expected_prompts"])
    expected_categories = int(config["dataset"]["expected_categories"])
    if len(prompts) != expected_prompts:
        raise ValueError(
            f"Expected {expected_prompts} prompts, found {len(prompts)}."
        )
    prompt_ids = [record["prompt_id"] for record in prompts]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt_id values must be unique.")
    if any(not record.get("prompt") for record in prompts):
        raise ValueError("Every prompt record needs non-empty prompt text.")
    categories = {record["category"] for record in prompts}
    if len(categories) != expected_categories:
        raise ValueError(
            f"Expected {expected_categories} categories, found {len(categories)}."
        )
    if "semantic_proxy" in config:
        expected_checks = int(config["semantic_proxy"]["checks_per_prompt"])
        for record in prompts:
            checks = record.get("checks")
            if not isinstance(checks, list) or len(checks) != expected_checks:
                raise ValueError(
                    f"{record['prompt_id']} must have {expected_checks} checks."
                )
            answers = {str(check.get("answer", "")).lower() for check in checks}
            if answers != {"yes", "no"}:
                raise ValueError(
                    f"{record['prompt_id']} needs one yes and one no check."
                )
            if any(not check.get("question") for check in checks):
                raise ValueError("Semantic-check questions must be non-empty.")

    rho_grid = tuple(float(value) for value in config["sampling"]["rho_grid"])
    purpose = config["experiment"]["purpose"]
    if purpose == "a_star_confirmation_development":
        if rho_grid != (0.0, 0.2, 0.4):
            raise ValueError(
                "The frozen A* confirmation grid must be (0.0, 0.2, 0.4)."
            )
    elif purpose == "a_star_coarse_coco_dev50":
        if rho_grid != STAGE_A_RHO_GRID:
            raise ValueError(
                f"The frozen A* coarse grid must be {STAGE_A_RHO_GRID}."
            )
    elif purpose == "a_star_fine_coco_dev50":
        search = config.get("a_star_search", {})
        center = float(search["coarse_center"])
        radius = float(search.get("fine_radius", 0.20))
        step = float(search.get("fine_step", 0.05))
        coarse = set(STAGE_A_RHO_GRID)
        fine = {
            round(index * step, 10)
            for index in range(round(1.0 / step) + 1)
            if abs(index * step - center) <= radius + 1e-9
        }
        expected = tuple(
            [0.0] + sorted(value for value in fine if value not in coarse)
        )
        if rho_grid != expected:
            raise ValueError(
                "The fine run must contain rho=0 plus only the new points in "
                f"the frozen neighborhood; expected {expected}."
            )
    elif purpose == "a_star_reference_coco_dev50":
        if rho_grid != (0.55,):
            raise ValueError("The frozen A* reference run requires rho=0.55.")
    elif rho_grid != STAGE_A_RHO_GRID:
        raise ValueError(f"Stage A rho grid must be {STAGE_A_RHO_GRID}.")
    if int(config["sampling"]["candidates_per_prompt"]) != 8:
        raise ValueError("The frozen development curve requires K=8.")
    if int(config["sampling"]["num_inference_steps"]) != 50:
        raise ValueError("The frozen SD1.5 protocol requires 50 steps.")
    if config["sampling"]["scheduler"] != "DDIM":
        raise ValueError("The frozen SD1.5 protocol requires DDIM.")
    if not model_path.joinpath("model_index.json").exists():
        raise FileNotFoundError(f"SD1.5 snapshot is incomplete: {model_path}")

    config["_resolved"] = {
        "dataset_path": str(dataset_path.resolve()),
        "model_path": str(model_path.resolve()),
        "config_path": str(path.resolve()),
        "config_profile": identity["profile"],
        "legacy_config_sha256": identity["legacy_sha256"],
        "default_output_dir": identity["output_dir"],
    }
    config["_prompts"] = prompts
    canonical = json.dumps(
        {key: value for key, value in config.items() if not key.startswith("_")},
        sort_keys=True,
        separators=(",", ":"),
    )
    config["_config_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return config


def rho_slug(rho: float) -> str:
    value = f"{float(rho):.2f}".rstrip("0")
    if value.endswith("."):
        value += "0"
    return f"rho_{value}".replace(".", "p")


def latent_seeds_for_prompt(
    config: dict[str, Any],
    prompt_index: int,
) -> list[int]:
    randomness = config["randomness"]
    first = (
        int(randomness["latent_seed_base"])
        + int(randomness["latent_prompt_stride"]) * prompt_index
    )
    count = int(config["sampling"]["candidates_per_prompt"])
    return [first + candidate_index for candidate_index in range(count)]


def condition_seed_for_prompt(
    config: dict[str, Any],
    prompt_index: int,
) -> int:
    randomness = config["randomness"]
    return (
        int(randomness["condition_seed_base"])
        + int(randomness["condition_prompt_stride"]) * prompt_index
    )


def make_initial_latents(
    seeds: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
) -> torch.Tensor:
    shape = (1, 4, height // 8, width // 8)
    samples = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        samples.append(
            torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
    return torch.cat(samples, dim=0)


def mean_pairwise_cosine_distance(features: np.ndarray) -> float:
    """Compute mean (1-cosine)/2 over unique pairs in one prompt group."""

    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError("Expected at least two [K, feature_dim] features.")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Feature vectors must have non-zero norm.")
    normalized = value / norms
    similarity = normalized @ normalized.T
    upper = np.triu_indices(value.shape[0], k=1)
    return float(((1.0 - similarity[upper]) / 2.0).mean())


def paired_bootstrap_mean_difference(
    method: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Prompt-level paired bootstrap interval for a mean difference."""

    method_value = np.asarray(method, dtype=np.float64)
    baseline_value = np.asarray(baseline, dtype=np.float64)
    if method_value.shape != baseline_value.shape or method_value.ndim != 1:
        raise ValueError("Method and baseline must be matching 1-D arrays.")
    if method_value.size == 0 or replicates <= 0:
        raise ValueError("Bootstrap inputs and replicate count must be positive.")
    differences = method_value - baseline_value
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        differences.size,
        size=(replicates, differences.size),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }
