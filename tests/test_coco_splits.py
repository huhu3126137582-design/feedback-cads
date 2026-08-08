"""Regression checks for the frozen COCO-Dev-50/Test-500 manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/coco"
DEV_PATH = DATA_ROOT / "coco_dev50.jsonl"
TEST_PATH = DATA_ROOT / "coco_test500.jsonl"
METADATA_PATH = DATA_ROOT / "coco_caption_splits.json"

DEV_SHA256 = "940030cc9f07a787f2041ea8144f831ad65a33cd2a55ee116f74ca4710b35754"
TEST_SHA256 = "ed2251c29003aea93bf2c623861d16187bbddb093382d3d029aa5787e5f361d5"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_frozen_coco_split_hashes_and_metadata_are_exact() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert _sha256(DEV_PATH) == DEV_SHA256
    assert _sha256(TEST_PATH) == TEST_SHA256
    assert metadata["dev_sha256"] == DEV_SHA256
    assert metadata["test_sha256"] == TEST_SHA256
    assert metadata["dev_count"] == 50
    assert metadata["test_count"] == 500
    assert metadata["image_ids_disjoint"] is True
    assert metadata["selection"]["shuffle_seed"] == 20260808
    assert metadata["selection"]["dev_count_requested"] == 50
    assert metadata["selection"]["test_count_requested"] == 500


def test_frozen_coco_splits_are_disjoint_unique_and_well_formed() -> None:
    dev = _read_jsonl(DEV_PATH)
    test = _read_jsonl(TEST_PATH)
    assert len(dev) == 50
    assert len(test) == 500
    assert [row["split_index"] for row in dev] == list(range(50))
    assert [row["split_index"] for row in test] == list(range(500))
    assert {row["split"] for row in dev} == {"COCO-Dev-50"}
    assert {row["split"] for row in test} == {"COCO-Test-500"}

    combined = dev + test
    assert len({row["image_id"] for row in combined}) == 550
    assert len({row["annotation_id"] for row in combined}) == 550
    assert len({row["prompt_id"] for row in combined}) == 550
    assert all(
        row["prompt_id"] == f"coco_{row['image_id']:012d}"
        for row in combined
    )
    assert all(5 <= row["content_token_count"] <= 20 for row in combined)
    assert all(
        row["encoded_token_count"] == row["content_token_count"] + 2
        for row in combined
    )
