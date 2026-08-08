#!/usr/bin/env python3
"""Export the Git-friendly numerical result and publication package."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"


COPY_MAP = {
    # Formal numerical result and audits.
    "outputs/formal_test_coco_test500/metrics/summary.csv": "formal_test500/metrics/summary.csv",
    "outputs/formal_test_coco_test500/metrics/summary.json": "formal_test500/metrics/summary.json",
    "outputs/formal_test_coco_test500/metrics/per_prompt.jsonl": "formal_test500/metrics/per_prompt.jsonl",
    "outputs/formal_test_coco_test500/metrics/paired_bootstrap.json": "formal_test500/metrics/paired_bootstrap.json",
    "outputs/formal_test_coco_test500/metrics/evaluation_complete.json": "formal_test500/metrics/evaluation_complete.json",
    "outputs/formal_test_coco_test500/metrics/evaluation_audit.json": "formal_test500/metrics/evaluation_audit.json",
    "outputs/formal_test_coco_test500/generation_complete.json": "formal_test500/generation_complete.json",
    "outputs/formal_test_coco_test500/technical_integrity_audit.json": "formal_test500/technical_integrity_audit.json",
    "outputs/formal_test_coco_test500/reporting/FORMAL_REPORT.md": "formal_test500/FORMAL_REPORT.md",
    "outputs/formal_test_coco_test500/reporting/reporting_audit.json": "formal_test500/reporting_audit.json",
    # Primary table and confidence-interval figure.
    "outputs/formal_test_coco_test500/publication/main_table.csv": "publication/main_table.csv",
    "outputs/formal_test_coco_test500/publication/main_table.md": "publication/main_table.md",
    "outputs/formal_test_coco_test500/publication/main_table.tex": "publication/main_table.tex",
    "outputs/formal_test_coco_test500/publication/primary_b_vs_a_ci_source_data.csv": "publication/primary_b_vs_a_ci_source_data.csv",
    "outputs/formal_test_coco_test500/publication/primary_b_vs_a_ci_nature.pdf": "publication/primary_b_vs_a_ci_nature.pdf",
    "outputs/formal_test_coco_test500/publication/primary_b_vs_a_ci_nature.png": "publication/primary_b_vs_a_ci_nature.png",
    "outputs/formal_test_coco_test500/publication/primary_b_vs_a_ci_nature.svg": "publication/primary_b_vs_a_ci_nature.svg",
    "outputs/formal_test_coco_test500/publication/FIGURE_LEGEND.md": "publication/FIGURE_LEGEND.md",
    "outputs/formal_test_coco_test500/publication/QA_NOTES.md": "publication/QA_NOTES.md",
    # Project figures. TIFF is reproducibly generated but omitted from Git.
    "outputs/formal_test_coco_test500/publication/project_figures/FIGURE_LEGENDS.md": "publication/project_figures/FIGURE_LEGENDS.md",
    "outputs/formal_test_coco_test500/publication/project_figures/QA_NOTES.md": "publication/project_figures/QA_NOTES.md",
    "outputs/formal_test_coco_test500/publication/project_figures/fig3_controller_dynamics_source_data.csv": "publication/project_figures/fig3_controller_dynamics_source_data.csv",
    "outputs/formal_test_coco_test500/publication/project_figures/fig4_ablation_source_data.csv": "publication/project_figures/fig4_ablation_source_data.csv",
    # Frozen development selections and reference signals.
    "outputs/a_star_coco_dev50/A_STAR_SELECTION.json": "development/A_STAR_SELECTION.json",
    "outputs/a_star_coco_dev50/REPORT.md": "development/A_STAR_REPORT.md",
    "outputs/b_feedback_margin_selection_coco_dev50/ALPHA_SELECTION.json": "development/ALPHA_SELECTION_STAGE1.json",
    "outputs/b_feedback_margin_selection_extended_coco_dev50/ALPHA_SELECTION_EXTENDED.json": "development/ALPHA_SELECTION_EXTENDED.json",
    "outputs/a_star_reference_coco_dev50/reference_curves.json": "development/reference_curves.json",
    "outputs/a_star_reference_coco_dev50/per_step.jsonl": "development/reference_per_step.jsonl",
    "outputs/a_star_reference_coco_dev50/REPORT.md": "development/REFERENCE_REPORT.md",
}

for stage in ("coarse", "fine"):
    source_root = f"outputs/a_strength_curve_coco_dev50_{stage}/metrics"
    for name in (
        "summary.csv",
        "paired_bootstrap.json",
        "per_prompt.jsonl",
        "curve_audit.json",
        "a_strength_curve.png",
    ):
        COPY_MAP[f"{source_root}/{name}"] = f"development/stage_a_{stage}/{name}"

for profile, source_dir in {
    "alpha_0p00": "b_feedback_coco_dev50",
    "alpha_0p05": "b_feedback_alpha_0p05_coco_dev50",
    "alpha_0p10": "b_feedback_alpha_0p10_coco_dev50",
    "alpha_0p15": "b_feedback_alpha_0p15_coco_dev50",
    "alpha_0p20": "b_feedback_alpha_0p20_coco_dev50",
}.items():
    for name in ("summary.csv", "paired_bootstrap.json", "per_prompt.jsonl"):
        COPY_MAP[f"outputs/{source_dir}/metrics/{name}"] = (
            f"development/stage_b/{profile}/{name}"
        )

for stem in (
    "fig1_feedback_cads_method",
    "fig2_qualitative_same_seed",
    "fig3_controller_dynamics",
    "fig4_dev_ablation",
):
    for suffix in ("pdf", "png"):
        COPY_MAP[
            "outputs/formal_test_coco_test500/publication/project_figures/"
            f"{stem}.{suffix}"
        ] = f"publication/project_figures/{stem}.{suffix}"
    if stem != "fig2_qualitative_same_seed":
        COPY_MAP[
            "outputs/formal_test_coco_test500/publication/project_figures/"
            f"{stem}.svg"
        ] = f"publication/project_figures/{stem}.svg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for source_value, target_value in COPY_MAP.items():
        source = PROJECT_ROOT / source_value
        target = RESULTS_ROOT / target_value
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size >= 100 * 1024 * 1024:
            raise RuntimeError(f"GitHub result file is at least 100 MiB: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    checksums = []
    for path in sorted(RESULTS_ROOT.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            checksums.append(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}")
    (RESULTS_ROOT / "CHECKSUMS.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    print(
        f"Exported {len(COPY_MAP)} result files; "
        f"checksummed {len(checksums)} files under {RESULTS_ROOT}."
    )


if __name__ == "__main__":
    main()
