"""Regression tests for the consolidated development configuration profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from feedback_cads.configuration import load_yaml_profile
from feedback_cads.experiments import load_curve_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "_steps"))
A_BUNDLE = PROJECT_ROOT / "configs/a_strength_search_coco_dev50.yaml"
B_BUNDLE = PROJECT_ROOT / "configs/b_feedback_search_coco_dev50.yaml"
pytestmark = pytest.mark.artifact


def test_a_profiles_retain_frozen_run_identity() -> None:
    expected = {
        "coarse": "outputs/a_strength_curve_coco_dev50_coarse",
        "fine": "outputs/a_strength_curve_coco_dev50_fine",
    }
    for profile, output in expected.items():
        config = load_curve_config(
            A_BUNDLE,
            project_root=PROJECT_ROOT,
            profile=profile,
        )
        frozen = json.loads(
            (PROJECT_ROOT / output / "run_config.json").read_text(
                encoding="utf-8"
            )
        )
        assert config["_config_sha256"] == frozen["_config_sha256"]
        assert config["_resolved"]["config_profile"] == profile


def test_b_profiles_retain_frozen_alpha_and_legacy_hashes() -> None:
    selection = json.loads(
        (
            PROJECT_ROOT
            / "outputs/b_feedback_margin_selection_extended_coco_dev50/"
            "ALPHA_SELECTION_EXTENDED.json"
        ).read_text(encoding="utf-8")
    )
    expected = {
        "alpha_0p00": 0.0,
        "alpha_0p05": 0.05,
        "alpha_0p10": 0.10,
        "alpha_0p15": 0.15,
        "alpha_0p20": 0.20,
    }
    for profile, alpha in expected.items():
        config, identity = load_yaml_profile(B_BUNDLE, profile=profile)
        assert float(config["feedback"]["reference_margin_alpha"]) == alpha
        assert identity["legacy_sha256"] == selection["artifact_hashes"][
            str(alpha)
        ]["config_sha256"]


def test_unified_parameter_selector_replays_frozen_outputs(tmp_path: Path) -> None:
    import select_parameters

    a_path = tmp_path / "a.json"
    a_args = type(
        "Args",
        (),
        {
            "coarse_dir": select_parameters.A_OUTPUTS["coarse"],
            "fine_dir": select_parameters.A_OUTPUTS["fine"],
            "output": a_path,
        },
    )()
    select_parameters.select_a_star(a_args)
    frozen_a = PROJECT_ROOT / "outputs/a_star_coco_dev50/A_STAR_SELECTION.json"
    assert hashlib.sha256(a_path.read_bytes()).hexdigest() == hashlib.sha256(
        frozen_a.read_bytes()
    ).hexdigest()

    b_path = tmp_path / "b.json"
    b_args = type("Args", (), {"extended": True, "output": b_path})()
    select_parameters.select_b_margin(b_args)
    frozen_b = (
        PROJECT_ROOT
        / "outputs/b_feedback_margin_selection_extended_coco_dev50/"
        "ALPHA_SELECTION_EXTENDED.json"
    )
    assert hashlib.sha256(b_path.read_bytes()).hexdigest() == hashlib.sha256(
        frozen_b.read_bytes()
    ).hexdigest()
