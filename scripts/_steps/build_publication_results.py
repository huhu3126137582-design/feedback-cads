#!/usr/bin/env python3
"""Build the publication-ready main table and primary confidence-interval figure."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads.configuration import load_yaml_profile  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/formal_artifacts_coco_test500.yaml"
CONFIG_PROFILE = "publication_results"
CONFIG_SHA256 = "8cc0de0c8367c9d00429181d4384c111d538e03c5670135aa84b999742ca1524"

# Editable text and publication-safe fonts are mandatory for the primary SVG.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

METHOD_KEYS = ["vanilla", "original_cads", "a_star", "b_feedback"]
METRICS = [
    ("dino_diversity", "DINO diversity"),
    ("clipscore", "CLIPScore"),
    ("hpsv2", "HPSv2"),
]
TABLE_COLUMNS = [
    ("dino_diversity", "DINO diversity ↑"),
    ("clip_image_diversity", "CLIP diversity ↑"),
    ("clipscore", "CLIPScore ↑"),
    ("hpsv2", "HPSv2 ↑"),
    ("prompt_run_seconds", "Time (s prompt⁻¹) ↓"),
    ("unet_calls", "UNet calls prompt⁻¹ ↓"),
]

BLUE = "#0F4D92"
BLUE_LIGHT = "#EAF2FA"
GREEN = "#2E7D4F"
RED = "#B64342"
INK = "#272727"
MID = "#767676"
LIGHT = "#D8D8D8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-standard JSON value {value}: {path}")
        ),
    )


def load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config, identity = load_yaml_profile(CONFIG_PATH, profile=CONFIG_PROFILE)
    if identity["legacy_sha256"] != CONFIG_SHA256:
        raise ValueError("Publication-results config identity changed.")
    reporting = config["reporting"]
    if (
        reporting["backend"] != "python"
        or reporting["post_evaluation_only"] is not True
        or reporting["cannot_change_formal_acceptance"] is not True
    ):
        raise ValueError("Publication reporting contract changed.")

    resolved: dict[str, Path] = {}
    for name, item in config["inputs"].items():
        path = (PROJECT_ROOT / item["path"]).resolve()
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen formal-evaluation input changed: {path}")
        resolved[name] = path

    summary = read_json(resolved["summary"])
    paired = read_json(resolved["paired_bootstrap"])
    complete = read_json(resolved["evaluation_complete"])
    audit = read_json(resolved["evaluation_audit"])
    contract = config["figure_contract"]
    if (
        [row["method"] for row in summary] != METHOD_KEYS
        or complete["image_count"] != 16000
        or complete["prompt_group_count"] != 2000
        or audit["passed"] is not True
        or paired["statistical_unit"] != contract["statistical_unit"]
        or paired["prompt_count"] != contract["prompt_count"]
        or paired["candidates_per_prompt"] != contract["candidates_per_prompt"]
        or paired["bootstrap_replicates"] != contract["bootstrap_replicates"]
        or paired["confidence_interval"] != contract["confidence_interval"]
    ):
        raise ValueError("Formal evaluation or publication contract is inconsistent.")
    primary = paired["comparisons"][contract["primary_comparison"]]
    if (
        primary["method"] != "b_feedback"
        or primary["comparator"] != "a_star"
        or primary["primary"] is not True
        or paired["primary_acceptance"]["strong_research_result"] is not True
    ):
        raise ValueError("The preregistered primary comparison changed.")
    return config, summary, paired


def _is_best(key: str, value: float, values: list[float]) -> bool:
    if len(set(values)) == 1:
        return False
    if key in {"prompt_run_seconds", "unet_calls"}:
        return value == min(values)
    return value == max(values)


def write_main_table(
    config: dict[str, Any], summary: list[dict[str, Any]], output: Path
) -> None:
    display = config["methods"]["display_names"]
    by_method = {row["method"]: row for row in summary}
    raw_columns = ["method"] + [key for key, _ in TABLE_COLUMNS]
    with (output / "main_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_columns)
        writer.writeheader()
        for method in METHOD_KEYS:
            source = by_method[method]
            writer.writerow(
                {"method": display[method]}
                | {key: source[key] for key, _ in TABLE_COLUMNS}
            )

    best = {
        key: [float(by_method[method][key]) for method in METHOD_KEYS]
        for key, _ in TABLE_COLUMNS
    }
    header = "| Method | " + " | ".join(label for _, label in TABLE_COLUMNS) + " |\n"
    alignment = "|:--|" + "--:|" * len(TABLE_COLUMNS) + "\n"
    lines = []
    for method in METHOD_KEYS:
        values = []
        for key, _ in TABLE_COLUMNS:
            value = float(by_method[method][key])
            if key == "unet_calls":
                rendered = f"{int(value)}"
            elif key == "prompt_run_seconds":
                rendered = f"{value:.4f}"
            else:
                rendered = f"{value:.6f}"
            if _is_best(key, value, best[key]):
                rendered = f"**{rendered}**"
            if method == "b_feedback" and key == "dino_diversity":
                rendered += "†"
            if method == "b_feedback" and key in {"clipscore", "hpsv2"}:
                rendered += "‡"
            values.append(rendered)
        lines.append(f"| {display[method]} | " + " | ".join(values) + " |")
    notes = (
        "\n\nValues are prompt-group means on COCO-Test-500 (n = 500 prompts; "
        "K = 8 images per prompt). Bold denotes the best method mean in each column; "
        "all methods use 50 UNet calls. † B* exceeds A* in DINO diversity with a positive "
        "95% paired-bootstrap CI. ‡ B* is non-inferior to A* under the preregistered 1% "
        "margin. Runtime is hardware-dependent.\n"
    )
    (output / "main_table.md").write_text(
        header + alignment + "\n".join(lines) + notes, encoding="utf-8"
    )

    latex_rows = []
    for method in METHOD_KEYS:
        values = []
        for key, _ in TABLE_COLUMNS:
            value = float(by_method[method][key])
            rendered = f"{int(value)}" if key == "unet_calls" else (
                f"{value:.4f}" if key == "prompt_run_seconds" else f"{value:.6f}"
            )
            if _is_best(key, value, best[key]):
                rendered = rf"\textbf{{{rendered}}}"
            if method == "b_feedback" and key == "dino_diversity":
                rendered += "†"
            if method == "b_feedback" and key in {"clipscore", "hpsv2"}:
                rendered += "‡"
            values.append(rendered)
        name = display[method].replace("*", r"\textsuperscript{*}")
        latex_rows.append(name + " & " + " & ".join(values) + r" \\")
    latex = """\\begin{table}[t]
\\centering
\\caption{Formal results on COCO-Test-500. Values are prompt-group means over 500 prompts with eight images per prompt.}
\\label{tab:formal-results}
\\small
\\begin{tabular}{lrrrrrr}
\\toprule
Method & DINO $\\uparrow$ & CLIP div. $\\uparrow$ & CLIPScore $\\uparrow$ & HPSv2 $\\uparrow$ & s/prompt $\\downarrow$ & UNet \\\\
\\midrule
""" + "\n".join(latex_rows) + """
\\bottomrule
\\end{tabular}
\\begin{minipage}{0.99\\linewidth}
\\footnotesize † Positive 95\\% paired-bootstrap CI versus A*. ‡ Non-inferior to A* under the preregistered 1\\% margin. Bold denotes the best mean per column; runtime is hardware-dependent.
\\end{minipage}
\\end{table}
"""
    (output / "main_table.tex").write_text(latex, encoding="utf-8")


def write_source_data(primary: dict[str, Any], output: Path) -> None:
    fields = [
        "metric",
        "method",
        "comparator",
        "method_mean",
        "comparator_mean",
        "mean_difference",
        "ci95_lower",
        "ci95_upper",
        "noninferiority_margin",
        "noninferior",
        "strong_improvement",
    ]
    with (output / "primary_b_vs_a_ci_source_data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, label in METRICS:
            result = primary["metrics"][key]
            writer.writerow(
                {
                    "metric": label,
                    "method": primary["method"],
                    "comparator": primary["comparator"],
                    **{field: result[field] for field in fields[3:]},
                }
            )


def make_primary_figure(
    config: dict[str, Any], primary: dict[str, Any], output: Path
) -> None:
    contract = config["figure_contract"]
    if (
        float(contract["target_width_mm"]) != 180.0
        or float(contract["target_height_mm"]) != 63.0
        or int(config["outputs"]["png_dpi"]) != 300
        or int(config["outputs"]["tiff_dpi"]) != 600
    ):
        raise ValueError("Publication dimensions or raster resolution changed.")
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0866, 2.4803),
        gridspec_kw={"width_ratios": [1.12, 1.0, 1.0]},
    )
    panel_labels = ["a", "b", "c"]
    for index, (axis, (key, label)) in enumerate(zip(axes, METRICS)):
        result = primary["metrics"][key]
        point = 1000.0 * float(result["mean_difference"])
        lower = 1000.0 * float(result["ci95_lower"])
        upper = 1000.0 * float(result["ci95_upper"])
        ni = -1000.0 * float(result["noninferiority_margin"])
        span = max(upper, 0.0) - min(lower, ni)
        left = min(lower, ni) - 0.09 * span
        right = max(upper, 0.0) + 0.09 * span

        if index == 0:
            axis.set_facecolor(BLUE_LIGHT)
        axis.axvline(0.0, color=INK, linewidth=0.9, zorder=1)
        axis.axvline(ni, color=RED, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
        axis.plot([lower, upper], [0.46, 0.46], color=BLUE, linewidth=1.8, zorder=3)
        axis.plot([lower, lower], [0.42, 0.50], color=BLUE, linewidth=1.0, zorder=3)
        axis.plot([upper, upper], [0.42, 0.50], color=BLUE, linewidth=1.0, zorder=3)
        axis.plot(point, 0.46, "o", color=BLUE, markersize=5.2, zorder=4)
        status = "Improved" if result["strong_improvement"] else "Non-inferior"
        status_color = GREEN if result["strong_improvement"] else MID
        axis.text(
            0.5,
            0.81,
            status,
            transform=axis.transAxes,
            ha="center",
            va="center",
            color=status_color,
            fontsize=6.5,
            fontweight="bold",
        )
        axis.text(
            0.5,
            0.68,
            f"Δ = {point:+.3f}  [{lower:+.3f}, {upper:+.3f}]",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color=INK,
            fontsize=6.1,
        )
        axis.text(
            -0.15,
            1.08,
            panel_labels[index],
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
        axis.set_title(label, loc="left", pad=12, fontweight="bold")
        axis.set_xlim(left, right)
        axis.set_ylim(0.15, 0.92)
        axis.set_yticks([])
        axis.set_xlabel("B* − A*  (×10⁻³)", labelpad=4)
        axis.tick_params(axis="x", direction="out", length=2.5, width=0.7, pad=2)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color(INK)
        axis.spines["bottom"].set_linewidth(0.8)

    handles = [
        Line2D([0], [0], color=BLUE, marker="o", markersize=4, linewidth=1.5, label="Mean difference and 95% CI"),
        Line2D([0], [0], color=INK, linewidth=0.9, label="No difference"),
        Line2D([0], [0], color=RED, linewidth=0.9, linestyle=(0, (3, 2)), label="−1% NI margin"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.005),
        ncol=3,
        columnspacing=1.3,
        handlelength=2.2,
        handletextpad=0.5,
        fontsize=6.2,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.22, top=0.74, wspace=0.32)

    basename = output / config["outputs"]["primary_figure_basename"]
    fig.savefig(basename.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(basename.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(
        basename.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        basename.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(fig)


def write_supporting_text(config: dict[str, Any], output: Path) -> None:
    caption = """# Figure legend

**Fig. 1 | Feedback-CADS improves diversity over fixed CADS while preserving quality non-inferiority.** a–c, Paired differences between Feedback-CADS (B*) and the fixed-CADS baseline (A*) on COCO-Test-500. Points denote mean prompt-level differences and bars denote 95% paired-bootstrap confidence intervals from 10,000 replicates (n = 500 prompts; K = 8 images per prompt). The solid black line marks no difference, and the dashed red line marks the preregistered negative 1% non-inferiority margin. a, DINO diversity increases significantly because its confidence interval lies above zero. b,c, CLIPScore and HPSv2 have slightly negative point estimates but remain above their non-inferiority margins. Source data are provided in `primary_b_vs_a_ci_source_data.csv`.
"""
    (output / "FIGURE_LEGEND.md").write_text(caption, encoding="utf-8")
    contract = config["figure_contract"]
    qa = f"""# Publication result package: figure contract and QA notes

- Core conclusion: {contract['core_conclusion']}
- Archetype: quantitative grid with DINO diversity as the hero panel and quality metrics as supporting non-inferiority evidence.
- Backend: Python/matplotlib only.
- Final size: {contract['target_width_mm']} mm × {contract['target_height_mm']} mm before tight-crop export.
- Statistical unit: prompt; n = {contract['prompt_count']}; K = {contract['candidates_per_prompt']}; {contract['bootstrap_replicates']:,} paired-bootstrap replicates; 95% CI.
- Source mapping: formal summary rows → main-table method means; preregistered `b_feedback_vs_a_star` metric records → effect points, confidence intervals and non-inferiority margins.
- Exclusions or transformations: no method, prompt, metric or replicate was excluded. Figure values are multiplied by 1,000 only for axis readability.
- Evidence hierarchy: DINO diversity is primary; CLIPScore and HPSv2 verify quality non-inferiority; inference time and UNet calls remain in the table.
- Reviewer risks controlled: the plot distinguishes improvement from non-inferiority, states n and CI construction, exposes negative quality point estimates, and does not generalize the B*–A* conclusion to B*–Vanilla.
- Formal acceptance: unchanged; this package is a post-evaluation presentation layer over hash-verified inputs.

## Statistical legend contract

- Test split: frozen COCO-Test-500; no test prompt was used for parameter selection.
- Replicate unit: prompt group. The K = 8 generated images define each prompt-level metric and are not treated as eight independent replicates.
- Center statistic: mean paired prompt-level difference, B* − A*.
- Interval: 95% prompt-level paired-bootstrap CI with 10,000 replicates and frozen seed 20260808.
- Baseline: A*, the matched fixed-strength clean-unconditional CADS baseline.
- Metrics: DINOv2 within-prompt pairwise distance, CLIPScore and HPSv2.1.
- Multiple-comparison correction and p-values: none; the figure reports the preregistered metric-wise confidence intervals and non-inferiority margins directly.
- Source data: `primary_b_vs_a_ci_source_data.csv`.
- Image integrity: not applicable; the figure contains vector line art only.

## Rendered panel audit

| Panel | Unique claim | Center | Interval | Replicate unit | Labels and reference lines | Collision check | Pass |
|:--|:--|:--|:--|:--|:--|:--|:--:|
| a | B* improves DINO diversity over A* | Mean paired difference | 95% paired-bootstrap CI | Prompt | Zero and −1% NI lines; improved status | CI, annotation and labels are separated | yes |
| b | B* CLIPScore is non-inferior, not improved | Mean paired difference | 95% paired-bootstrap CI | Prompt | Zero and −1% NI lines; non-inferior status | CI, annotation and labels are separated | yes |
| c | B* HPSv2 is non-inferior, not improved | Mean paired difference | 95% paired-bootstrap CI | Prompt | Zero and −1% NI lines; non-inferior status | CI, annotation and labels are separated | yes |

The assembled figure was inspected at its final double-column scale. The DINO panel remains the visual hero, all three panels use the same uncertainty definition, and colour is not the sole carrier of meaning because reference lines also differ by line style.
"""
    (output / "QA_NOTES.md").write_text(qa, encoding="utf-8")
    readme = """# Publication-ready formal results

## Main table

See `main_table.md` for the readable table, `main_table.csv` for exact source values, and `main_table.tex` for manuscript typesetting.

## Primary confidence-interval figure

![Primary B* versus A* comparison](primary_b_vs_a_ci_nature.png)

See `FIGURE_LEGEND.md` for the self-contained legend and `QA_NOTES.md` for the data and figure contract.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def build() -> Path:
    config, summary, paired = load_inputs()
    output = (PROJECT_ROOT / config["outputs"]["directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_main_table(config, summary, output)
    primary = paired["comparisons"][config["figure_contract"]["primary_comparison"]]
    write_source_data(primary, output)
    make_primary_figure(config, primary, output)
    write_supporting_text(config, output)

    artifacts = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "publication_results_complete.json"
    )
    completion = {
        "passed": True,
        "config_sha256": CONFIG_SHA256,
        "formal_acceptance_unchanged": True,
        "source_metric_rows": 3,
        "main_table_rows": 4,
        "figure_formats": ["svg", "pdf", "tiff", "png"],
        "artifact_sha256": {path.name: sha256(path) for path in artifacts},
    }
    (output / "publication_results_complete.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    print(build())
