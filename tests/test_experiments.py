"""Tests for the frozen paired stage-A experiment protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from feedback_cads.experiments import (
    STAGE_A_RHO_GRID,
    condition_seed_for_prompt,
    latent_seeds_for_prompt,
    load_curve_config,
    make_initial_latents,
    mean_pairwise_cosine_distance,
    paired_bootstrap_mean_difference,
    resolve_recorded_path,
    rho_slug,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return load_curve_config(
        PROJECT_ROOT / "configs/a_strength_search_coco_dev50.yaml",
        project_root=PROJECT_ROOT,
        profile="dev20",
    )


@pytest.mark.artifact
def test_frozen_curve_config_and_stratified_dataset() -> None:
    config = _config()
    assert tuple(config["sampling"]["rho_grid"]) == STAGE_A_RHO_GRID
    assert config["sampling"]["candidates_per_prompt"] == 8
    assert len(config["_prompts"]) == 20
    categories = [record["category"] for record in config["_prompts"]]
    assert len(set(categories)) == 5
    assert all(categories.count(category) == 4 for category in set(categories))
    assert len(config["_config_sha256"]) == 64


@pytest.mark.artifact
def test_seeds_are_paired_across_rho_and_distinct_across_prompts() -> None:
    config = _config()
    prompt_zero = latent_seeds_for_prompt(config, 0)
    prompt_one = latent_seeds_for_prompt(config, 1)
    assert prompt_zero == list(range(120000, 120008))
    assert set(prompt_zero).isdisjoint(prompt_one)
    assert condition_seed_for_prompt(config, 0) == 920000
    assert condition_seed_for_prompt(config, 0) != condition_seed_for_prompt(
        config, 1
    )
    for _rho in STAGE_A_RHO_GRID:
        assert latent_seeds_for_prompt(config, 0) == prompt_zero


def test_initial_latents_are_reproducible_per_candidate() -> None:
    seeds = [101, 102, 103]
    first = make_initial_latents(
        seeds,
        device=torch.device("cpu"),
        dtype=torch.float32,
        height=64,
        width=64,
    )
    replay = make_initial_latents(
        seeds,
        device=torch.device("cpu"),
        dtype=torch.float32,
        height=64,
        width=64,
    )
    assert torch.equal(first, replay)
    assert not torch.equal(first[0], first[1])


def test_pairwise_cosine_distance_uses_unique_pairs_and_half_scale() -> None:
    features = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    # Pair distances are 0.5, 1.0, and 0.5.
    assert mean_pairwise_cosine_distance(features) == pytest.approx(2 / 3)
    identical = np.repeat(np.array([[2.0, -3.0]]), repeats=8, axis=0)
    assert mean_pairwise_cosine_distance(identical) == pytest.approx(0.0)


def test_paired_bootstrap_uses_prompt_differences_and_is_reproducible() -> None:
    baseline = np.array([1.0, 2.0, 3.0, 4.0])
    method = baseline + np.array([0.1, 0.2, 0.3, 0.4])
    first = paired_bootstrap_mean_difference(
        method,
        baseline,
        replicates=1000,
        seed=55,
    )
    replay = paired_bootstrap_mean_difference(
        method,
        baseline,
        replicates=1000,
        seed=55,
    )
    assert first == replay
    assert first["mean_difference"] == pytest.approx(0.25)
    assert first["ci95_lower"] > 0.0


def test_rho_slug_is_stable() -> None:
    assert [rho_slug(value) for value in STAGE_A_RHO_GRID] == [
        "rho_0p0",
        "rho_0p2",
        "rho_0p4",
        "rho_0p6",
        "rho_0p8",
        "rho_1p0",
    ]
    assert rho_slug(0.05) == "rho_0p05"
    assert rho_slug(0.15) == "rho_0p15"


def test_recorded_absolute_path_rebases_to_current_checkout(
    tmp_path: Path,
) -> None:
    current = tmp_path / "outputs/formal/images/example.png"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"frozen-image")
    recorded = "/old/machine/project/outputs/formal/images/example.png"
    assert resolve_recorded_path(
        recorded, project_root=tmp_path
    ) == current.resolve()


def test_recorded_relative_and_unresolved_paths_are_stable(
    tmp_path: Path,
) -> None:
    relative = resolve_recorded_path(
        "data/coco/test.jsonl", project_root=tmp_path
    )
    assert relative == (tmp_path / "data/coco/test.jsonl").resolve()
    missing = Path("/external/archive/not-in-this-project.bin")
    assert resolve_recorded_path(
        missing, project_root=tmp_path
    ) == missing


@pytest.mark.artifact
def test_sd15_dev50_prompts_are_short_stratified_and_annotated() -> None:
    config = load_curve_config(
        PROJECT_ROOT / "configs/a_strength_search_coco_dev50.yaml",
        project_root=PROJECT_ROOT,
        profile="dev50_confirmation",
    )
    prompts = config["_prompts"]
    assert len(prompts) == 50
    categories = [record["category"] for record in prompts]
    assert all(categories.count(category) == 10 for category in set(categories))
    assert max(len(record["prompt"].split()) for record in prompts) <= 10
    for record in prompts:
        assert len(record["checks"]) == 2
        assert {check["answer"] for check in record["checks"]} == {"yes", "no"}
