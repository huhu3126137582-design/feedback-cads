#!/usr/bin/env python3
"""Build frozen formal tables, figures, and equal low-score case sheets."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads.experiments import (  # noqa: E402
    read_jsonl,
    resolve_recorded_path,
    write_json,
    write_jsonl,
)
from feedback_cads.configuration import load_yaml_profile  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/formal_artifacts_coco_test500.yaml"
CONFIG_PROFILE = "reporting"
CONFIG_SHA256 = (
    "1f007843415d5f1ae315ad9f75e4e93acbe20a918381d899fd5a2cbbb85af14b"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-standard JSON value {value}: {path}")
        ),
    )


def _checked(path_value: str, expected_sha256: str) -> Path:
    path = (PROJECT_ROOT / path_value).resolve()
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError(f"Frozen reporting input changed: {path}")
    return path


def load_and_validate() -> dict[str, Any]:
    config, identity = load_yaml_profile(
        CONFIG_PATH,
        profile=CONFIG_PROFILE,
    )
    if identity["legacy_sha256"] != CONFIG_SHA256:
        raise ValueError("Frozen reporting config SHA-256 changed.")
    reporting = config["reporting"]
    if (
        reporting["revision"] != 1
        or reporting["frozen_before_case_selection"] is not True
        or reporting["post_evaluation_only"] is not True
        or reporting["cannot_change_formal_acceptance"] is not True
    ):
        raise ValueError("Formal reporting freeze flags changed.")
    resolved = {}
    for name, section in config["inputs"].items():
        resolved[name] = str(_checked(section["path"], section["sha256"]))
    summary = _strict_json(Path(resolved["summary"]))
    paired = _strict_json(Path(resolved["paired_bootstrap"]))
    complete = _strict_json(Path(resolved["evaluation_complete"]))
    audit = _strict_json(Path(resolved["evaluation_audit"]))
    per_prompt = read_jsonl(Path(resolved["per_prompt"]))
    per_image = read_jsonl(Path(resolved["per_image"]))
    methods = config["methods"]["order"]
    if (
        [row["method"] for row in summary] != methods
        or complete["image_count"] != 16000
        or complete["prompt_group_count"] != 2000
        or audit["passed"] is not True
        or audit["primary_acceptance_reproduced"] is not True
        or len(per_prompt) != 2000
        or len(per_image) != 16000
    ):
        raise ValueError("Formal evaluation is not complete and audited.")
    case = config["low_score_cases"]
    if (
        case["selection_unit"] != "prompt_group_mean"
        or case["criteria"] != ["clipscore", "hpsv2"]
        or case["direction"] != "ascending"
        or int(case["cases_per_method_per_criterion"]) != 5
        or case["stable_tie_break"] != "prompt_source_index_ascending"
        or case["equal_slots_required_for_every_method_and_criterion"] is not True
        or case["manual_failure_confirmation_required"] is not True
    ):
        raise ValueError("Low-score case selection rule changed.")
    output = (PROJECT_ROOT / config["main_outputs"]["directory"]).resolve()
    staging = output.with_name(".formal_reporting_in_progress")
    fingerprint = hashlib.sha256(
        (
            CONFIG_SHA256
            + "".join(config["inputs"][name]["sha256"] for name in sorted(config["inputs"]))
        ).encode("ascii")
    ).hexdigest()
    config["_resolved"] = resolved
    config["_summary"] = summary
    config["_paired"] = paired
    config["_per_prompt"] = per_prompt
    config["_per_image"] = per_image
    config["_output"] = str(output)
    config["_staging"] = str(staging)
    config["_fingerprint"] = fingerprint
    return config


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def _make_main_tables(config: dict[str, Any], staging: Path) -> None:
    display = config["methods"]["display_names"]
    rows = []
    for row in config["_summary"]:
        rows.append(
            {
                "method": display[row["method"]],
                "dino_diversity": row["dino_diversity"],
                "clip_image_diversity": row["clip_image_diversity"],
                "clipscore": row["clipscore"],
                "hpsv2": row["hpsv2"],
                "seconds_per_prompt": row["prompt_run_seconds"],
                "unet_calls": int(row["unet_calls"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(staging / "main_table.csv", index=False)
    header = (
        "| Method | DINO diversity | CLIP image diversity | CLIPScore | "
        "HPSv2 | s/prompt | UNet calls |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [
        "| {method} | {dino_diversity:.6f} | {clip_image_diversity:.6f} | "
        "{clipscore:.6f} | {hpsv2:.6f} | {seconds_per_prompt:.4f} | "
        "{unet_calls} |".format(**row)
        for row in rows
    ]
    (staging / "main_table.md").write_text(
        header + "\n".join(lines) + "\n", encoding="utf-8"
    )


def _make_figures(config: dict[str, Any], staging: Path) -> None:
    dpi = int(config["figures"]["dpi"])
    primary = config["_paired"]["comparisons"]["b_feedback_vs_a_star"]["metrics"]
    metrics = [
        ("dino_diversity", "DINO diversity"),
        ("clipscore", "CLIPScore"),
        ("hpsv2", "HPSv2"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, (key, label) in zip(axes, metrics):
        result = primary[key]
        point = float(result["mean_difference"])
        lower = float(result["ci95_lower"])
        upper = float(result["ci95_upper"])
        axis.errorbar(
            [0], [point], yerr=[[point - lower], [upper - point]], fmt="o", capsize=5
        )
        axis.axhline(0.0, color="black", linewidth=1, label="No difference")
        axis.axhline(
            -float(result["noninferiority_margin"]),
            color="tab:red",
            linestyle="--",
            linewidth=1,
            label="-1% NI margin",
        )
        axis.set_xticks([])
        axis.set_title(label)
        axis.set_ylabel("B* - A*")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")
    figure.suptitle("Primary paired effects with 95% bootstrap CI")
    figure.tight_layout()
    figure.savefig(staging / config["main_outputs"]["primary_ci_figure"], dpi=dpi)
    plt.close(figure)

    summary = {row["method"]: row for row in config["_summary"]}
    methods = config["methods"]["order"]
    labels = [config["methods"]["display_names"][method] for method in methods]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, (key, label) in zip(axes, metrics):
        values = [float(summary[method][key]) for method in methods]
        axis.bar(labels, values, color=["#777777", "#cc8963", "#5f9ed1", "#59a14f"])
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("COCO-Test-500 formal method means")
    figure.tight_layout()
    figure.savefig(staging / config["main_outputs"]["method_means_figure"], dpi=dpi)
    plt.close(figure)


def _select_cases(config: dict[str, Any], staging: Path) -> list[dict[str, Any]]:
    case = config["low_score_cases"]
    count = int(case["cases_per_method_per_criterion"])
    images_by_group: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in config["_per_image"]:
        images_by_group[(row["method"], int(row["prompt_source_index"]))].append(row)
    slots = []
    for criterion in case["criteria"]:
        for method in config["methods"]["order"]:
            candidates = sorted(
                (row for row in config["_per_prompt"] if row["method"] == method),
                key=lambda row: (float(row[criterion]), int(row["prompt_source_index"])),
            )[:count]
            if len(candidates) != count:
                raise RuntimeError(f"Not enough low-score cases: {method}/{criterion}")
            for rank, row in enumerate(candidates, start=1):
                images = sorted(
                    images_by_group[(method, int(row["prompt_source_index"]))],
                    key=lambda image: int(image["candidate_id"]),
                )
                if len(images) != 8:
                    raise RuntimeError("A selected case does not have eight images.")
                slots.append(
                    {
                        "criterion": criterion,
                        "rank": rank,
                        "method": method,
                        "prompt_source_index": int(row["prompt_source_index"]),
                        "prompt_id": row["prompt_id"],
                        "prompt": row["prompt"],
                        "criterion_value": float(row[criterion]),
                        "clipscore": float(row["clipscore"]),
                        "hpsv2": float(row["hpsv2"]),
                        "automatic_label": case["automatic_label"],
                        "manual_failure_confirmed": None,
                        "failure_type": None,
                        "notes": None,
                        "candidate_images": [
                            {
                                "candidate_id": int(image["candidate_id"]),
                                "image_path": image["image_path"],
                                "image_sha256": image["image_sha256"],
                            }
                            for image in images
                        ],
                    }
                )
    expected = len(case["criteria"]) * len(config["methods"]["order"]) * count
    if len(slots) != expected:
        raise RuntimeError("Low-score selection is not balanced.")
    write_jsonl(staging / "low_score_case_manifest.jsonl", slots)
    with (staging / "manual_failure_annotations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "criterion",
                "rank",
                "method",
                "prompt_source_index",
                "prompt_id",
                "prompt",
                "criterion_value",
                "manual_failure_confirmed",
                "failure_type",
                "notes",
            ]
        )
        for row in slots:
            writer.writerow(
                [
                    row["criterion"],
                    row["rank"],
                    row["method"],
                    row["prompt_source_index"],
                    row["prompt_id"],
                    row["prompt"],
                    row["criterion_value"],
                    "",
                    "",
                    "",
                ]
            )
    return slots


def _contact_sheets(config: dict[str, Any], staging: Path, slots: list[dict[str, Any]]) -> None:
    case = config["low_score_cases"]
    size = int(case["contact_sheet_thumbnail_size"])
    columns = int(case["contact_sheet_columns"])
    label_width = 300
    header_height = 52
    row_height = size + 44
    display = config["methods"]["display_names"]
    sheet_dir = staging / "low_score_contact_sheets"
    sheet_dir.mkdir()
    title_font = _font(21)
    body_font = _font(13)
    small_font = _font(11)
    for criterion in case["criteria"]:
        for method in config["methods"]["order"]:
            selected = sorted(
                (row for row in slots if row["criterion"] == criterion and row["method"] == method),
                key=lambda row: int(row["rank"]),
            )
            canvas = Image.new(
                "RGB",
                (label_width + columns * size, header_height + len(selected) * row_height),
                "white",
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 12),
                f"Lowest {criterion} prompt groups - {display[method]} (not human-confirmed failures)",
                fill="black",
                font=title_font,
            )
            for row_index, row in enumerate(selected):
                top = header_height + row_index * row_height
                prompt = row["prompt"]
                if len(prompt) > 42:
                    prompt = prompt[:39] + "..."
                draw.text(
                    (10, top + 8),
                    f"#{row['rank']} {row['prompt_id']}\n{criterion}={row['criterion_value']:.6f}\n{prompt}",
                    fill="black",
                    font=body_font,
                    spacing=3,
                )
                for column, image_record in enumerate(row["candidate_images"]):
                    image_path = resolve_recorded_path(
                        image_record["image_path"], project_root=PROJECT_ROOT
                    )
                    with Image.open(image_path) as source:
                        thumb = source.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
                    left = label_width + column * size
                    canvas.paste(thumb, (left, top))
                    draw.rectangle((left, top, left + 24, top + 16), fill="black")
                    draw.text((left + 3, top + 1), str(column), fill="white", font=small_font)
            canvas.save(sheet_dir / f"lowest_{criterion}_{method}.png", format="PNG")


def _write_report(config: dict[str, Any], staging: Path) -> None:
    primary = config["_paired"]["comparisons"]["b_feedback_vs_a_star"]["metrics"]
    acceptance = config["_paired"]["primary_acceptance"]
    dino = primary["dino_diversity"]
    clip = primary["clipscore"]
    hps = primary["hpsv2"]
    text = f"""# COCO-Test-500 Formal Report

## Main table

{(staging / 'main_table.md').read_text(encoding='utf-8')}

## Preregistered primary comparison: B* - A*

| Metric | Mean difference | 95% paired-bootstrap CI | 1% NI margin | Noninferior | Strong improvement |
|---|---:|---:|---:|:---:|:---:|
| DINO diversity | {dino['mean_difference']:+.6f} | [{dino['ci95_lower']:.6f}, {dino['ci95_upper']:.6f}] | {dino['noninferiority_margin']:.6f} | yes | yes |
| CLIPScore | {clip['mean_difference']:+.6f} | [{clip['ci95_lower']:.6f}, {clip['ci95_upper']:.6f}] | {clip['noninferiority_margin']:.6f} | yes | no |
| HPSv2 | {hps['mean_difference']:+.6f} | [{hps['ci95_lower']:.6f}, {hps['ci95_upper']:.6f}] | {hps['noninferiority_margin']:.6f} | yes | no |

Assessment success: **{str(acceptance['assessment_success']).lower()}**  
Closed-loop diversity success: **{str(acceptance['closed_loop_diversity_success']).lower()}**  
Strong research result: **{str(acceptance['strong_research_result']).lower()}**

B* increases DINO diversity relative to A* with a positive 95% CI. CLIPScore and HPSv2 are slightly lower, not higher, but both pass the preregistered 1% noninferiority rule. The defensible claim is therefore a statistically significant diversity gain over the direct open-loop A* comparator while preserving preregistered quality noninferiority.

The secondary B* versus Vanilla comparison does not pass HPSv2 noninferiority. It must not be described as quality-noninferior to Vanilla on every core metric.

## Low-score candidates for manual review

The `low_score_case_manifest.jsonl` contains exactly five lowest prompt groups per method for CLIPScore and five for HPSv2. The two criterion lists are separate and may overlap. Contact sheets are under `low_score_contact_sheets/`.

These are automatically selected low-metric candidates, not human-confirmed failures. Complete `manual_failure_annotations.csv` before making claims about missing objects, attribute errors, distortion, or other visual failure types.
"""
    (staging / config["main_outputs"]["report_markdown"]).write_text(text, encoding="utf-8")


def build(config: dict[str, Any]) -> Path:
    output = Path(config["_output"])
    staging = Path(config["_staging"])
    if output.exists():
        complete = _strict_json(output / "reporting_complete.json")
        if complete["reporting_fingerprint"] != config["_fingerprint"]:
            raise RuntimeError("Existing report has another fingerprint.")
        return output
    lock = {
        "reporting_config_sha256": CONFIG_SHA256,
        "reporting_fingerprint": config["_fingerprint"],
        "input_hashes": {
            name: section["sha256"] for name, section in config["inputs"].items()
        },
        "case_selection_rule": config["low_score_cases"],
        "formal_acceptance_may_change": False,
    }
    if staging.exists():
        if _strict_json(staging / "reporting_lock.json") != lock:
            raise RuntimeError("Staging report has another fingerprint.")
    else:
        staging.mkdir(parents=True)
        write_json(staging / "reporting_lock.json", lock)
    _make_main_tables(config, staging)
    _make_figures(config, staging)
    slots = _select_cases(config, staging)
    _contact_sheets(config, staging, slots)
    _write_report(config, staging)
    artifacts = sorted(
        path for path in staging.rglob("*") if path.is_file() and path.name != "reporting_complete.json"
    )
    completion = {
        "reporting_config_sha256": CONFIG_SHA256,
        "reporting_fingerprint": config["_fingerprint"],
        "formal_acceptance_unchanged": True,
        "method_count": 4,
        "low_score_slot_count": len(slots),
        "slots_per_method_per_criterion": 5,
        "contact_sheet_count": 8,
        "manual_failure_labels_completed": False,
        "artifact_sha256": {
            str(path.relative_to(staging)): sha256(path) for path in artifacts
        },
    }
    write_json(staging / "reporting_complete.json", completion)
    staging.replace(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--confirm-reporting-sha256", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_and_validate()
    if args.validate_only:
        print(
            json.dumps(
                {
                    "validated": True,
                    "reporting_config_sha256": CONFIG_SHA256,
                    "reporting_fingerprint": config["_fingerprint"],
                    "formal_acceptance": config["_paired"]["primary_acceptance"],
                    "expected_low_score_slots": 40,
                    "output_exists": Path(config["_output"]).exists(),
                },
                indent=2,
            )
        )
        return
    if args.confirm_reporting_sha256 != CONFIG_SHA256:
        raise ValueError(
            "Formal reporting requires --confirm-reporting-sha256 " + CONFIG_SHA256
        )
    output = build(config)
    print(json.dumps(_strict_json(output / "reporting_complete.json"), indent=2))


if __name__ == "__main__":
    main()
