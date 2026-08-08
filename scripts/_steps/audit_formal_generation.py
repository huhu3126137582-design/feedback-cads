#!/usr/bin/env python3
"""Audit formal generation integrity without computing research metrics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "_steps"))

from feedback_cads.experiments import (  # noqa: E402
    read_jsonl,
    resolve_recorded_path,
    write_json,
)
from run_frozen_formal_generation import (  # noqa: E402
    FROZEN_FORMAL_PROTOCOL_SHA256,
    _a_config,
    _b_config,
    audit_method_stats,
    load_run_spec,
    sha256,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def audit(*, allow_downstream_artifacts: bool = False) -> dict[str, Any]:
    spec = load_run_spec(
        "formal",
        confirm_protocol_sha256=FROZEN_FORMAL_PROTOCOL_SHA256,
    )
    root = Path(spec["output_root"])
    completion_path = root / "generation_complete.json"
    manifest_path = root / "manifest.jsonl"
    completion = _load_json(completion_path)
    manifest = read_jsonl(manifest_path)
    if completion.get("quality_metrics_computed") is not False:
        raise RuntimeError("Generation completion claims quality metrics were used.")
    if completion.get("image_count") != 16_000 or len(manifest) != 16_000:
        raise RuntimeError("Formal manifest does not contain exactly 16,000 images.")
    if completion.get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("Formal manifest SHA-256 changed.")

    method_counts = Counter(str(row["method"]) for row in manifest)
    if method_counts != Counter({method: 4000 for method in spec["method_order"]}):
        raise RuntimeError(f"Per-method image counts are wrong: {method_counts}")
    if len({str(row["image_path"]) for row in manifest}) != len(manifest):
        raise RuntimeError("Formal image paths are not unique.")

    paired: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    verified_png = 0
    image_bytes = 0
    for index, row in enumerate(manifest, start=1):
        image_path = resolve_recorded_path(
            row["image_path"], project_root=PROJECT_ROOT
        )
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if sha256(image_path) != row["image_sha256"]:
            raise RuntimeError(f"Image SHA-256 changed: {image_path}")
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            if image.format != "PNG" or image.size != (512, 512):
                raise RuntimeError(f"Invalid formal PNG: {image_path}")
        image_bytes += image_path.stat().st_size
        verified_png += 1
        paired[(int(row["prompt_source_index"]), int(row["candidate_id"]))].append(
            row
        )
        if index % 1000 == 0:
            print(json.dumps({"verified_png": index, "total": len(manifest)}), flush=True)

    if len(paired) != 4000:
        raise RuntimeError("Expected 4,000 prompt-candidate pairing groups.")
    for key, rows in paired.items():
        if {str(row["method"]) for row in rows} != set(spec["method_order"]):
            raise RuntimeError(f"Incomplete method pairing: {key}")
        if len({int(row["latent_seed"]) for row in rows}) != 1:
            raise RuntimeError(f"Latent seed mismatch: {key}")
        cads_rows = [row for row in rows if row["method"] != "vanilla"]
        if len({int(row["condition_seed"]) for row in cads_rows}) != 1:
            raise RuntimeError(f"Positive condition seed mismatch: {key}")
        vanilla = next(row for row in rows if row["method"] == "vanilla")
        if vanilla["condition_seed"] is not None:
            raise RuntimeError(f"Vanilla consumed a condition seed: {key}")

    a_config = _a_config(spec)
    _, feedback_config = _b_config(spec)
    diagnostic_counts: Counter[str] = Counter()
    b_history_steps = 0
    for method in spec["method_order"]:
        diagnostics = sorted((root / "images" / method).glob("*/diagnostics.json"))
        if len(diagnostics) != 500:
            raise RuntimeError(f"Expected 500 diagnostics for {method}.")
        for path in diagnostics:
            value = _load_json(path)
            if (
                value.get("run_spec_sha256") != spec["run_spec_sha256"]
                or value.get("formal_protocol_sha256")
                != FROZEN_FORMAL_PROTOCOL_SHA256
                or value.get("formal_test_data_used") is not True
                or value.get("global_torch_rng_unchanged") is not True
            ):
                raise RuntimeError(f"Stale formal diagnostics: {path}")
            audit_method_stats(
                method,
                value["pipeline_stats"],
                a_config=a_config,
                feedback_config=feedback_config,
            )
            if method == "b_feedback":
                b_history_steps += len(
                    value["pipeline_stats"]["feedback_control_history"]
                )
            diagnostic_counts[method] += 1

    forbidden_metric_files = [
        str(path)
        for path in root.rglob("*")
        if path.is_file()
        and any(
            token in path.name.lower()
            for token in ("clipscore", "hpsv2", "dino", "quality_metrics")
        )
    ]
    if forbidden_metric_files and not allow_downstream_artifacts:
        raise RuntimeError(f"Formal metric files already exist: {forbidden_metric_files}")

    report = {
        "passed": True,
        "audit_scope": "generation_integrity_only_no_research_metrics",
        "formal_protocol_sha256": FROZEN_FORMAL_PROTOCOL_SHA256,
        "run_spec_sha256": spec["run_spec_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "manifest_rows": len(manifest),
        "method_image_counts": dict(method_counts),
        "verified_png_count": verified_png,
        "verified_png_resolution": [512, 512],
        "total_png_bytes": image_bytes,
        "prompt_candidate_pair_groups": len(paired),
        "diagnostic_counts": dict(diagnostic_counts),
        "b_feedback_history_rows": b_history_steps,
        "all_prompt_candidate_latents_paired": True,
        "all_cads_positive_condition_seeds_paired": True,
        "vanilla_condition_seeds_are_null": True,
        "all_diagnostics_pass_runtime_audits": True,
        "quality_metrics_computed": False,
        "formal_metric_files_seen": bool(forbidden_metric_files),
    }
    # The preregistered audit is written only before metric inspection. A
    # post-evaluation replay is deliberately read-only so it cannot replace
    # the frozen pre-metric audit used by downstream hash locks.
    if not allow_downstream_artifacts:
        write_json(root / "technical_integrity_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-downstream-artifacts",
        action="store_true",
        help=(
            "Read-only integrity replay after formal metrics/reporting exist; "
            "does not overwrite the frozen pre-metric audit."
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(allow_downstream_artifacts=args.allow_downstream_artifacts),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
