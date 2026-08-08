"""Regression tests for frozen formal tables and low-score case selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "_steps"))

from build_formal_report import (  # noqa: E402
    CONFIG_PATH,
    CONFIG_PROFILE,
    CONFIG_SHA256,
    load_and_validate,
)
from feedback_cads.configuration import load_yaml_profile  # noqa: E402


pytestmark = pytest.mark.artifact


def test_formal_reporting_inputs_and_selection_rule_are_frozen() -> None:
    _profile, identity = load_yaml_profile(CONFIG_PATH, profile=CONFIG_PROFILE)
    assert identity["legacy_sha256"] == CONFIG_SHA256
    config = load_and_validate()
    cases = config["low_score_cases"]
    assert cases["criteria"] == ["clipscore", "hpsv2"]
    assert cases["selection_unit"] == "prompt_group_mean"
    assert cases["direction"] == "ascending"
    assert cases["cases_per_method_per_criterion"] == 5
    assert cases["stable_tie_break"] == "prompt_source_index_ascending"
    assert cases["equal_slots_required_for_every_method_and_criterion"] is True
    assert cases["manual_failure_confirmation_required"] is True
    assert len(config["_per_prompt"]) == 2000
    assert len(config["_per_image"]) == 16000


def test_reporting_cannot_relabel_automatic_cases_as_confirmed_failures() -> None:
    config = load_and_validate()
    assert config["low_score_cases"]["automatic_label"] == (
        "low_metric_candidate_not_human_confirmed_failure"
    )
    assert config["reporting"]["cannot_change_formal_acceptance"] is True


def test_completed_formal_reporting_is_audited_and_balanced() -> None:
    path = (
        PROJECT_ROOT
        / "outputs/formal_test_coco_test500/reporting/reporting_audit.json"
    )
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["main_table_rows_verified"] == 4
    assert audit["balanced_low_score_slots_verified"] == 40
    assert audit["source_images_per_slot_verified"] == 8
    assert audit["contact_sheets_verified"] == 8
    assert audit["result_figures_verified"] == 2
    assert audit["manual_annotation_rows_blank"] == 40
    assert audit["formal_acceptance_unchanged"] is True
    assert audit["required_claim_caveats_present"] is True
