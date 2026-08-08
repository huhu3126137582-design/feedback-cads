"""Regression tests for the frozen all-method formal evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "_steps"))

from evaluate_formal_coco_test500 import (  # noqa: E402
    CONFIG_PATH,
    CONFIG_PROFILE,
    CONFIG_SHA256,
    FORMAL_PROTOCOL_SHA256,
    load_and_validate,
)
from feedback_cads.configuration import load_yaml_profile  # noqa: E402


pytestmark = pytest.mark.artifact


def test_frozen_formal_evaluation_config_and_inputs_are_exact() -> None:
    _profile, identity = load_yaml_profile(CONFIG_PATH, profile=CONFIG_PROFILE)
    assert identity["legacy_sha256"] == CONFIG_SHA256
    config = load_and_validate()
    assert config["formal_protocol"]["sha256"] == FORMAL_PROTOCOL_SHA256
    assert len(config["_records"]) == 16000
    assert len(config["_groups"]) == 2000
    assert config["generation"]["method_order"] == [
        "vanilla",
        "original_cads",
        "a_star",
        "b_feedback",
    ]
    assert config["statistics"]["primary_comparison"] == (
        "b_feedback_vs_a_star"
    )
    assert config["statistics"]["bootstrap_replicates"] == 10000
    assert config["statistics"]["bootstrap_seed"] == 20260808
    assert config["statistics"]["core_metrics"] == [
        "dino_diversity",
        "clipscore",
        "hpsv2",
    ]
    assert config["_fingerprint"] == (
        "a920a97f3f5ec20d8de731647fa960a03eb3b11cc16e9315cb1d184e70566db7"
    )


def test_formal_evaluation_has_no_subset_or_skip_controls() -> None:
    source = (PROJECT_ROOT / "scripts/_steps/evaluate_formal_coco_test500.py").read_text(
        encoding="utf-8"
    )
    assert "--limit" not in source
    assert "--skip" not in source
    assert "--batch-size" not in source


def test_completed_formal_evaluation_has_an_independent_audit() -> None:
    audit_path = (
        PROJECT_ROOT
        / "outputs/formal_test_coco_test500/metrics/evaluation_audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["independently_verified_images"] == 16000
    assert audit["independently_verified_prompt_groups"] == 2000
    assert set(audit["maximum_per_prompt_recomputation_error"].values()) == {
        0.0
    }
    assert audit["all_four_comparisons_reproduced"] is True
    assert audit["all_noninferiority_margins_reproduced"] is True
    assert audit["primary_acceptance_reproduced"] is True
    acceptance = audit["primary_acceptance"]
    assert acceptance["assessment_success"] is True
    assert acceptance["closed_loop_diversity_success"] is True
    assert acceptance["strong_research_result"] is True
