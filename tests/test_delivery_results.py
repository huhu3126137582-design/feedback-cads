"""Checks for the Git-friendly numerical result package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_delivery_checksums_and_github_file_limit() -> None:
    rows = [
        line.split("  ", maxsplit=1)
        for line in (RESULTS_ROOT / "CHECKSUMS.sha256")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(rows) >= 60
    for expected, relative in rows:
        path = PROJECT_ROOT / relative
        assert path.is_file()
        assert path.stat().st_size < 100 * 1024 * 1024
        assert _sha256(path) == expected


def test_delivery_primary_result_matches_preregistered_claim() -> None:
    paired = json.loads(
        (
            RESULTS_ROOT
            / "formal_test500/metrics/paired_bootstrap.json"
        ).read_text(encoding="utf-8")
    )
    primary = paired["comparisons"]["b_feedback_vs_a_star"]
    assert primary["primary"] is True
    dino = primary["metrics"]["dino_diversity"]
    assert dino["mean_difference"] > 0.0
    assert dino["ci95_lower"] > 0.0
    assert primary["metrics"]["clipscore"]["noninferior"] is True
    assert primary["metrics"]["hpsv2"]["noninferior"] is True
    assert paired["primary_acceptance"]["strong_research_result"] is True


def test_delivery_text_data_contains_no_checkout_absolute_path() -> None:
    suffixes = {".csv", ".json", ".jsonl", ".md", ".tex"}
    for path in RESULTS_ROOT.rglob("*"):
        if path.is_file() and path.suffix in suffixes:
            assert "/root/autodl-tmp/fdan" not in path.read_text(
                encoding="utf-8"
            )
