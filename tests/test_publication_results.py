"""Checks for the publication-ready table and primary CI figure package."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "_steps"))

from build_publication_results import load_inputs  # noqa: E402


pytestmark = pytest.mark.artifact


def test_publication_inputs_are_frozen_formal_results() -> None:
    config, summary, paired = load_inputs()
    assert config["reporting"]["post_evaluation_only"] is True
    assert config["reporting"]["cannot_change_formal_acceptance"] is True
    assert [row["method"] for row in summary] == [
        "vanilla",
        "original_cads",
        "a_star",
        "b_feedback",
    ]
    assert paired["comparisons"]["b_feedback_vs_a_star"]["primary"] is True


def test_publication_package_is_complete_and_editable() -> None:
    output = PROJECT_ROOT / "outputs/formal_test_coco_test500/publication"
    complete = json.loads(
        (output / "publication_results_complete.json").read_text(encoding="utf-8")
    )
    assert complete["passed"] is True
    assert complete["formal_acceptance_unchanged"] is True
    assert complete["main_table_rows"] == 4
    assert complete["source_metric_rows"] == 3
    assert complete["figure_formats"] == ["svg", "pdf", "tiff", "png"]
    for name, expected_sha256 in complete["artifact_sha256"].items():
        path = output / name
        assert path.is_file()
        import hashlib

        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
    assert "<text" in (output / "primary_b_vs_a_ci_nature.svg").read_text(
        encoding="utf-8"
    )
    with (output / "primary_b_vs_a_ci_source_data.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert float(rows[0]["ci95_lower"]) > 0.0
    assert all(row["noninferior"] == "True" for row in rows)
