"""Regression checks for the four publication project figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "_steps"))

from build_project_figures import load_and_validate, select_qualitative_case  # noqa: E402


pytestmark = pytest.mark.artifact


def test_project_figure_inputs_and_qualitative_rule() -> None:
    config = load_and_validate()
    selected, images = select_qualitative_case(config)
    assert selected["eligible_prompt_count"] > 0
    assert len(images) == 32
    assert selected["clipscore_delta_b_minus_a"] >= -selected["clipscore_margin"]
    assert selected["hpsv2_delta_b_minus_a"] >= -selected["hpsv2_margin"]
    assert sorted({int(row["candidate_id"]) for row in images}) == list(range(8))
    assert sorted({row["method"] for row in images}) == [
        "a_star", "b_feedback", "original_cads", "vanilla"
    ]


def test_project_figure_package_is_complete() -> None:
    output = PROJECT_ROOT / "outputs/formal_test_coco_test500/publication/project_figures"
    complete = json.loads((output / "project_figures_complete.json").read_text(encoding="utf-8"))
    assert complete["passed"] is True
    assert complete["formal_acceptance_unchanged"] is True
    assert complete["figures"] == 4
    assert complete["controller_prompts"] == 500
    for name, expected in complete["artifact_sha256"].items():
        path = output / name
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    for stem in (
        "fig1_feedback_cads_method",
        "fig2_qualitative_same_seed",
        "fig3_controller_dynamics",
        "fig4_dev_ablation",
    ):
        assert "<text" in (output / f"{stem}.svg").read_text(encoding="utf-8")
        for suffix in ("pdf", "tiff", "png"):
            assert (output / f"{stem}.{suffix}").is_file()
