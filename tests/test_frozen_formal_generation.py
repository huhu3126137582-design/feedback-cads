"""Safety and pairing tests for the unified four-method generator."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts" / "_steps"
sys.path.insert(0, str(SCRIPTS_ROOT))

from run_frozen_formal_generation import (  # noqa: E402
    FROZEN_FORMAL_PROTOCOL_SHA256,
    FORMAL_PROTOCOL_PATH,
    SMOKE_CONFIG_PATH,
    load_run_spec,
)


SMOKE_CONFIG_SHA256 = (
    "1312d29e77479276915fc3c7090018d4553908db4d82d33c2759ec3e43f1600c"
)
pytestmark = pytest.mark.artifact


def test_smoke_spec_uses_dev_only_and_exact_formal_method_protocol() -> None:
    assert hashlib.sha256(SMOKE_CONFIG_PATH.read_bytes()).hexdigest() == (
        SMOKE_CONFIG_SHA256
    )
    spec = load_run_spec("smoke")
    assert spec["formal_test_data_used"] is False
    assert Path(spec["dataset_path"]).name == "coco_dev50.jsonl"
    assert spec["selected_prompt_indices"] == [0]
    assert len(spec["prompts"]) == 1
    assert spec["method_order"] == [
        "vanilla",
        "original_cads",
        "a_star",
        "b_feedback",
    ]
    assert spec["sampling"]["num_inference_steps"] == 50
    assert spec["sampling"]["candidates_per_prompt"] == 8
    assert spec["expected_images_per_method"] == 8
    assert spec["expected_total_images"] == 32
    assert Path(spec["output_root"]).name == (
        "formal_generation_smoke_coco_dev50"
    )
    assert Path(spec["output_root"]).resolve() != (
        PROJECT_ROOT / "outputs/formal_test_coco_test500"
    ).resolve()


def test_formal_mode_requires_exact_explicit_confirmation() -> None:
    with pytest.raises(ValueError, match="requires --confirm-protocol"):
        load_run_spec("formal")
    with pytest.raises(ValueError, match="requires --confirm-protocol"):
        load_run_spec("formal", confirm_protocol_sha256="wrong")

    spec = load_run_spec(
        "formal",
        confirm_protocol_sha256=FROZEN_FORMAL_PROTOCOL_SHA256,
    )
    assert spec["formal_test_data_used"] is True
    assert len(spec["prompts"]) == 500
    assert spec["selected_prompt_indices"] == list(range(500))
    assert spec["expected_images_per_method"] == 4000
    assert spec["expected_total_images"] == 16000
    assert Path(spec["output_root"]).name == "formal_test_coco_test500"
    assert (Path(spec["output_root"]) / "run_lock.json").is_file()
    assert (Path(spec["output_root"]) / "generation_complete.json").is_file()


def test_smoke_rejects_a_different_formal_protocol_link(tmp_path: Path) -> None:
    config = yaml.safe_load(SMOKE_CONFIG_PATH.read_text(encoding="utf-8"))
    config["formal_protocol"]["sha256"] = "0" * 64
    changed = tmp_path / "changed-smoke.yaml"
    changed.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="not linked to the frozen protocol"):
        load_run_spec("smoke", smoke_config_path=changed)


def test_protocol_path_and_hash_are_not_runtime_overrides() -> None:
    assert FORMAL_PROTOCOL_PATH == (
        PROJECT_ROOT / "configs/formal_test_coco_test500.yaml"
    )
    assert hashlib.sha256(FORMAL_PROTOCOL_PATH.read_bytes()).hexdigest() == (
        FROZEN_FORMAL_PROTOCOL_SHA256
    )
