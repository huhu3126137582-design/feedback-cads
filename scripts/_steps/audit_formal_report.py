#!/usr/bin/env python3
"""Audit formal report tables, figures, and balanced low-score selections."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "_steps"))

from build_formal_report import CONFIG_SHA256, load_and_validate, sha256  # noqa: E402
from feedback_cads.experiments import (  # noqa: E402
    read_jsonl,
    resolve_recorded_path,
    write_json,
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit() -> dict[str, Any]:
    config = load_and_validate()
    root = Path(config["_output"])
    complete = _json(root / "reporting_complete.json")
    if (
        complete["reporting_config_sha256"] != CONFIG_SHA256
        or complete["reporting_fingerprint"] != config["_fingerprint"]
        or complete["formal_acceptance_unchanged"] is not True
        or complete["low_score_slot_count"] != 40
        or complete["manual_failure_labels_completed"] is not False
    ):
        raise RuntimeError("Formal reporting completion metadata is invalid.")
    for relative, expected in complete["artifact_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Formal report artifact changed: {path}")

    slots = read_jsonl(root / "low_score_case_manifest.jsonl")
    counts = Counter((row["method"], row["criterion"]) for row in slots)
    expected_counts = Counter(
        {
            (method, criterion): 5
            for method in config["methods"]["order"]
            for criterion in config["low_score_cases"]["criteria"]
        }
    )
    if counts != expected_counts:
        raise RuntimeError("Low-score slots are not balanced.")
    selected_lookup = {
        (row["method"], row["criterion"], int(row["rank"])): row for row in slots
    }
    for method in config["methods"]["order"]:
        for criterion in config["low_score_cases"]["criteria"]:
            expected = sorted(
                (row for row in config["_per_prompt"] if row["method"] == method),
                key=lambda row: (float(row[criterion]), int(row["prompt_source_index"])),
            )[:5]
            for rank, prompt_row in enumerate(expected, start=1):
                selected = selected_lookup[(method, criterion, rank)]
                if (
                    selected["prompt_id"] != prompt_row["prompt_id"]
                    or selected["prompt_source_index"]
                    != int(prompt_row["prompt_source_index"])
                    or selected["criterion_value"] != float(prompt_row[criterion])
                    or selected["manual_failure_confirmed"] is not None
                    or selected["failure_type"] is not None
                    or selected["notes"] is not None
                ):
                    raise RuntimeError(f"Low-score rank changed: {method}/{criterion}/{rank}")
                if len(selected["candidate_images"]) != 8:
                    raise RuntimeError("A selected case is missing candidates.")
                for image in selected["candidate_images"]:
                    path = resolve_recorded_path(
                        image["image_path"], project_root=PROJECT_ROOT
                    )
                    if not path.is_file() or sha256(path) != image["image_sha256"]:
                        raise RuntimeError(f"Selected source image changed: {path}")

    sheet_dir = root / "low_score_contact_sheets"
    sheets = sorted(sheet_dir.glob("*.png"))
    if len(sheets) != 8:
        raise RuntimeError("Expected eight low-score contact sheets.")
    for path in sheets + [root / "method_means.png", root / "primary_b_vs_a_ci.png"]:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.width < 1000 or image.height < 500:
                raise RuntimeError(f"Report figure is unexpectedly small: {path}")

    with (root / "manual_failure_annotations.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        annotations = list(csv.DictReader(handle))
    if len(annotations) != 40 or any(
        row["manual_failure_confirmed"]
        or row["failure_type"]
        or row["notes"]
        for row in annotations
    ):
        raise RuntimeError("Manual annotations must start as 40 blank rows.")

    main_table = list(csv.DictReader((root / "main_table.csv").open(encoding="utf-8")))
    if len(main_table) != 4:
        raise RuntimeError("Formal main table must contain four methods.")
    display = config["methods"]["display_names"]
    summary = {display[row["method"]]: row for row in config["_summary"]}
    for row in main_table:
        source = summary[row["method"]]
        for column in (
            "dino_diversity",
            "clip_image_diversity",
            "clipscore",
            "hpsv2",
        ):
            if float(row[column]) != float(source[column]):
                raise RuntimeError(f"Main table changed: {row['method']}/{column}")
    report_text = (root / "FORMAL_REPORT.md").read_text(encoding="utf-8")
    required_statements = (
        "Strong research result: **true**",
        "slightly lower, not higher",
        "does not pass HPSv2 noninferiority",
        "not human-confirmed failures",
    )
    if any(statement not in report_text for statement in required_statements):
        raise RuntimeError("Formal report is missing a required caveat.")

    result = {
        "passed": True,
        "reporting_config_sha256": CONFIG_SHA256,
        "reporting_fingerprint": config["_fingerprint"],
        "main_table_rows_verified": 4,
        "balanced_low_score_slots_verified": 40,
        "source_images_per_slot_verified": 8,
        "contact_sheets_verified": 8,
        "result_figures_verified": 2,
        "manual_annotation_rows_blank": 40,
        "formal_acceptance_unchanged": True,
        "required_claim_caveats_present": True,
    }
    write_json(root / "reporting_audit.json", result)
    return result


def main() -> None:
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
