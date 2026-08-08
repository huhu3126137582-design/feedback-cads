"""Regression tests for the frozen one-shot COCO-Test-500 protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts" / "_steps"
sys.path.insert(0, str(SCRIPTS_ROOT))

from validate_formal_test_protocol import validate_protocol  # noqa: E402


PROTOCOL_PATH = PROJECT_ROOT / "configs/formal_test_coco_test500.yaml"
PROTOCOL_SHA256 = (
    "9d19e02e284aefe0ed58fdc5a47dd0a0fec0966936a40dd9e281d2c97d0161ef"
)
pytestmark = pytest.mark.artifact


def test_frozen_formal_protocol_hash_and_dependencies_are_exact() -> None:
    assert hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() == (
        PROTOCOL_SHA256
    )
    report = validate_protocol(PROTOCOL_PATH, require_unstarted=False)
    assert report == {
        "passed": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset_sha256": (
            "ed2251c29003aea93bf2c623861d16187bbddb093382d3d029aa5787e5f361d5"
        ),
        "prompt_count": 500,
        "method_count": 4,
        "candidate_count": 8,
        "expected_total_images": 16000,
        "unique_test_latent_seeds": 4000,
        "unique_test_positive_condition_seeds": 4000,
        "seed_ranges_disjoint_from_development": True,
        "frozen_a_star_rho": 0.55,
        "frozen_b_reference_margin_alpha": 0.15,
        "formal_output_absent": report["formal_output_absent"],
        "formal_metrics_seen_at_freeze": False,
        "formal_metrics_output_exists": report[
            "formal_metrics_output_exists"
        ],
    }


def test_formal_protocol_rejects_parameter_drift(tmp_path: Path) -> None:
    config = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    config["methods"]["b_feedback"]["fixed_reference_margin_alpha"] = 0.20
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Frozen B alpha changed"):
        validate_protocol(changed, require_unstarted=False)
