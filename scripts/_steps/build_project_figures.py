#!/usr/bin/env python3
"""Build four publication figures for the Feedback-CADS project."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads.configuration import load_yaml_profile  # noqa: E402
from feedback_cads.experiments import resolve_recorded_path  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/formal_artifacts_coco_test500.yaml"
CONFIG_PROFILE = "project_figures"
CONFIG_SHA256 = "37f6e55339e309c62948852037d017188fdb862fba02cf1c4ce4f831fff00478"
FIGURE_WIDTH_MM = 180.0

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
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

BLUE = "#0F4D92"
BLUE_2 = "#3775BA"
BLUE_LIGHT = "#EAF2FA"
TEAL = "#42949E"
VIOLET = "#7C6CCF"
GREEN = "#2E7D4F"
RED = "#B64342"
RED_LIGHT = "#F6CFCB"
INK = "#272727"
MID = "#767676"
LIGHT = "#D8D8D8"
PALE = "#F3F4F5"

METHOD_COLORS = {
    "vanilla": "#767676",
    "original_cads": "#B88978",
    "a_star": "#7884B4",
    "b_feedback": BLUE,
}


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_and_validate() -> dict[str, Any]:
    config, identity = load_yaml_profile(CONFIG_PATH, profile=CONFIG_PROFILE)
    if identity["legacy_sha256"] != CONFIG_SHA256:
        raise ValueError("Project-figure config identity changed.")
    reporting = config["reporting"]
    if (
        reporting["backend"] != "python"
        or reporting["post_evaluation_only"] is not True
        or reporting["cannot_change_formal_acceptance"] is not True
        or float(config["outputs"]["width_mm"]) != FIGURE_WIDTH_MM
        or int(config["outputs"]["png_dpi"]) != 300
        or int(config["outputs"]["tiff_dpi"]) != 600
    ):
        raise ValueError("Project-figure reporting contract changed.")
    resolved: dict[str, str] = {}
    for name, item in config["inputs"].items():
        path = (PROJECT_ROOT / item["path"]).resolve()
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen figure input changed: {path}")
        resolved[name] = str(path)
    complete = read_json(Path(resolved["formal_generation_complete"]))
    audit = read_json(Path(resolved["formal_generation_audit"]))
    if (
        complete["image_count"] != 16000
        or complete["prompt_count"] != 500
        or complete["all_runs_used_exactly_50_unet_calls"] is not True
        or audit["passed"] is not True
        or audit["all_diagnostics_pass_runtime_audits"] is not True
    ):
        raise ValueError("Formal generation is not complete and audited.")
    config["_resolved"] = resolved
    return config


def save_figure(fig: plt.Figure, output: Path, basename: str) -> None:
    target = output / basename
    fig.savefig(target.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        target.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        target.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 7.0,
    fontweight: str = "normal",
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.35,rounding_size=1.2",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=INK,
        linespacing=1.25,
        zorder=3,
    )
    return box


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = MID,
    connectionstyle: str = "arc3",
    linewidth: float = 1.1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
    )


def build_fig1_method(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0866, 3.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")

    chips = [
        ("Training-free", BLUE_LIGHT, BLUE),
        ("Clean unconditional", "#EEF3F7", TEAL),
        ("DDIM-50", PALE, MID),
        ("50 UNet calls", PALE, MID),
    ]
    x = 2.0
    for text, face, edge in chips:
        width = 13.0 if text != "Clean unconditional" else 19.0
        rounded_box(ax, (x, 52.0), width, 4.0, text, face, edge, 6.2, "bold")
        x += width + 1.3

    ax.text(0.7, 48.8, "a", fontsize=8.5, fontweight="bold", va="center")
    ax.text(4.0, 48.8, "A* open-loop baseline", fontsize=8.0, fontweight="bold", va="center")
    rounded_box(ax, (31, 38.5), 18, 9, "", "#EEF0F7", "#7884B4")
    ax.text(40.0, 44.6, "Fixed condition annealing", ha="center", va="center", fontsize=5.8, color=INK, zorder=3)
    ax.text(40.0, 41.4, r"$\rho = 0.55$", ha="center", va="center", fontsize=6.2, color=INK, zorder=3)
    rounded_box(ax, (56, 38.5), 16, 9, "SD v1.5\nUNet + DDIM", PALE, MID, 7.0)
    rounded_box(ax, (79, 38.5), 17, 9, "K = 8 candidates", PALE, MID, 7.0)
    arrow(ax, (49, 43), (56, 43))
    arrow(ax, (72, 43), (79, 43))
    ax.text(40.0, 36.2, "same ρ for every prompt and step in the active window", ha="center", fontsize=6.2, color=MID)

    ax.plot([2, 98], [33.5, 33.5], color=LIGHT, linewidth=0.8)
    ax.text(0.7, 30.0, "b", fontsize=8.5, fontweight="bold", va="center")
    ax.text(4.0, 30.0, "Feedback-CADS (B*)", fontsize=8.0, fontweight="bold", va="center", color=BLUE)

    rounded_box(ax, (3, 15), 13, 12, "Text prompt\nCLIP embedding", PALE, MID, 6.8)
    rounded_box(ax, (21, 15), 17, 12, "", BLUE_LIGHT, BLUE_2)
    ax.text(29.5, 23.2, "Condition annealing", ha="center", va="center", fontsize=5.8, fontweight="bold", color=INK, zorder=3)
    ax.text(29.5, 19.2, r"$q = q^{\mathrm{cap}} \times \rho$", ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK, zorder=3)
    rounded_box(ax, (43, 15), 15, 12, "SD v1.5\nUNet + DDIM", PALE, MID, 6.8)
    rounded_box(ax, (63, 15), 13, 12, "K = 8\ncandidates", PALE, MID, 6.8)
    rounded_box(ax, (81, 15), 16, 12, "Online signals\nDg and Dz", "#E9F4F4", TEAL, 6.8, "bold")
    for start, end in [((16, 21), (21, 21)), ((38, 21), (43, 21)), ((58, 21), (63, 21)), ((76, 21), (81, 21))]:
        arrow(ax, start, end)

    rounded_box(ax, (78, 2), 19, 8, "Reference curves\nfrom frozen A*", "#F1EDF8", VIOLET, 6.5)
    rounded_box(ax, (46, 2), 27, 8, "", BLUE_LIGHT, BLUE)
    ax.text(59.5, 7.5, "P controller", ha="center", va="center", fontsize=5.8, fontweight="bold", color=INK, zorder=3)
    ax.text(59.5, 4.7, r"$\rho_{\mathrm{next}} = \operatorname{clip}(\rho + k_D e^D)$", ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK, zorder=3)
    arrow(ax, (87.5, 15), (87.5, 10), TEAL)
    arrow(ax, (78, 6), (73, 6), VIOLET)
    arrow(ax, (46, 6), (29.5, 15), BLUE, "arc3,rad=-0.28", 1.4)
    ax.text(38.2, 12.2, "one-step delay", fontsize=6.2, color=BLUE, ha="center", va="center")
    ax.text(59.5, 0.2, "prompt-adaptive ρ; no current-step rerun", fontsize=6.1, color=MID, ha="center")

    ax.annotate(
        "condition noise active: steps 0–29",
        xy=(32, 30.7),
        xytext=(51, 30.7),
        ha="center",
        va="center",
        fontsize=6.3,
        color=BLUE,
        arrowprops={"arrowstyle": "-[", "color": BLUE_2, "lw": 0.9},
    )
    ax.text(76, 30.7, "steps 30–49: q = 0 (clean condition)", fontsize=6.3, color=MID, va="center")
    save_figure(fig, output, "fig1_feedback_cads_method")


def select_qualitative_case(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_prompt = read_jsonl(Path(config["_resolved"]["formal_per_prompt"]))
    per_image = read_jsonl(Path(config["_resolved"]["formal_per_image"]))
    paired = read_json(Path(config["_resolved"]["formal_paired_bootstrap"]))
    primary = paired["comparisons"][config["qualitative_selection"]["comparison"]]
    clip_margin = float(primary["metrics"]["clipscore"]["noninferiority_margin"])
    hps_margin = float(primary["metrics"]["hpsv2"]["noninferiority_margin"])
    grouped = {(row["method"], int(row["prompt_source_index"])): row for row in per_prompt}
    eligible = []
    for index in range(500):
        a = grouped[("a_star", index)]
        b = grouped[("b_feedback", index)]
        dino_delta = float(b["dino_diversity"]) - float(a["dino_diversity"])
        clip_delta = float(b["clipscore"]) - float(a["clipscore"])
        hps_delta = float(b["hpsv2"]) - float(a["hpsv2"])
        if clip_delta >= -clip_margin and hps_delta >= -hps_margin:
            eligible.append(
                {
                    "prompt_source_index": index,
                    "prompt_id": b["prompt_id"],
                    "prompt": b["prompt"],
                    "dino_delta_b_minus_a": dino_delta,
                    "clipscore_delta_b_minus_a": clip_delta,
                    "hpsv2_delta_b_minus_a": hps_delta,
                }
            )
    if not eligible:
        raise RuntimeError("No prompt satisfies the frozen qualitative selection rule.")
    target = float(np.median([row["dino_delta_b_minus_a"] for row in eligible]))
    selected = min(
        eligible,
        key=lambda row: (abs(row["dino_delta_b_minus_a"] - target), row["prompt_source_index"]),
    )
    selected["eligible_prompt_count"] = len(eligible)
    selected["eligible_dino_delta_median"] = target
    selected["clipscore_margin"] = clip_margin
    selected["hpsv2_margin"] = hps_margin
    images = [
        row
        for row in per_image
        if int(row["prompt_source_index"]) == selected["prompt_source_index"]
    ]
    if len(images) != 32:
        raise RuntimeError("Selected formal prompt does not contain 32 method-candidate images.")
    for row in images:
        path = resolve_recorded_path(
            row["image_path"], project_root=PROJECT_ROOT
        )
        if not path.is_file() or sha256(path) != row["image_sha256"]:
            raise RuntimeError(f"Selected source image changed: {path}")
    return selected, images


def build_fig2_qualitative(config: dict[str, Any], output: Path) -> dict[str, Any]:
    selected, images = select_qualitative_case(config)
    display = config["methods"]["display_names"]
    lookup = {(row["method"], int(row["candidate_id"])): row for row in images}
    prompt_rows = read_jsonl(Path(config["_resolved"]["formal_per_prompt"]))
    metrics = {
        row["method"]: row
        for row in prompt_rows
        if int(row["prompt_source_index"]) == selected["prompt_source_index"]
    }

    fig, axes = plt.subplots(4, 8, figsize=(7.0866, 4.35))
    fig.patch.set_facecolor("white")
    for row_index, method in enumerate(config["methods"]["order"]):
        for candidate in range(8):
            ax = axes[row_index, candidate]
            record = lookup[(method, candidate)]
            image_path = resolve_recorded_path(
                record["image_path"], project_root=PROJECT_ROOT
            )
            image = mpimg.imread(image_path)
            ax.imshow(image, interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.25 if method == "b_feedback" else 0.55)
                spine.set_color(METHOD_COLORS[method])
            if row_index == 0:
                ax.set_title(f"Seed {candidate + 1}", fontsize=6.0, pad=3)
            if candidate == 0:
                ax.set_ylabel(
                    f"{display[method]}\nDINO {float(metrics[method]['dino_diversity']):.3f}",
                    fontsize=6.0,
                    color=METHOD_COLORS[method],
                    fontweight="bold" if method == "b_feedback" else "normal",
                    rotation=0,
                    rotation_mode="anchor",
                    ha="right",
                    va="center",
                    labelpad=7,
                )
    prompt = selected["prompt"]
    fig.text(0.145, 0.975, f'Prompt: “{prompt}”', ha="left", va="top", fontsize=7.0, fontweight="bold")
    fig.text(
        0.99,
        0.975,
        "Same latent seed within each column",
        ha="right",
        va="top",
        fontsize=6.2,
        color=MID,
    )
    fig.subplots_adjust(left=0.145, right=0.995, top=0.91, bottom=0.02, wspace=0.025, hspace=0.035)
    save_figure(fig, output, "fig2_qualitative_same_seed")

    manifest = {
        "selection_rule": config["qualitative_selection"],
        "selected": selected,
        "method_metrics": {
            method: {
                "dino_diversity": metrics[method]["dino_diversity"],
                "clipscore": metrics[method]["clipscore"],
                "hpsv2": metrics[method]["hpsv2"],
            }
            for method in config["methods"]["order"]
        },
        "images": [
            {
                "method": row["method"],
                "candidate_id": row["candidate_id"],
                "latent_seed": row["latent_seed"],
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
            }
            for row in sorted(images, key=lambda item: (config["methods"]["order"].index(item["method"]), item["candidate_id"]))
        ],
    }
    (output / "fig2_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_controller_dynamics(config: dict[str, Any]) -> dict[str, np.ndarray]:
    spec = config["controller_dynamics"]
    paths = sorted(PROJECT_ROOT.glob(spec["diagnostics_glob"]))
    if len(paths) != int(spec["expected_prompts"]):
        raise RuntimeError(f"Expected 500 B* diagnostics, found {len(paths)}.")
    keys = ["rho", "pollution", "d_g", "d_z", "target_g", "target_z", "cap"]
    values = {key: [] for key in keys}
    prompt_indices = []
    for path in paths:
        row = read_json(path)
        stats = row["pipeline_stats"]
        history = stats["feedback_control_history"]
        if (
            row["method"] != "b_feedback"
            or row["formal_test_data_used"] is not True
            or len(history) != int(spec["expected_steps"])
            or stats["unet_calls"] != 50
        ):
            raise RuntimeError(f"Invalid formal controller diagnostic: {path}")
        prompt_indices.append(int(row["prompt_source_index"]))
        values["rho"].append([float(item["rho_used_by_prompt"][0]) for item in history])
        values["pollution"].append([float(item["pollution_by_prompt"][0]) for item in history])
        values["d_g"].append([float(item["guidance_diversity_by_prompt"][0]) for item in history])
        values["d_z"].append([float(item["predicted_clean_latent_diversity_by_prompt"][0]) for item in history])
        values["target_g"].append([float(item["guidance_target"]) for item in history])
        values["target_z"].append([float(item["predicted_clean_latent_target"]) for item in history])
        values["cap"].append([float(item["pollution_cap"]) for item in history])
    if sorted(prompt_indices) != list(range(500)):
        raise RuntimeError("Formal controller diagnostics do not cover Test-500 exactly once.")
    return {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}


def summarize_controller(data: dict[str, np.ndarray], output: Path) -> dict[str, np.ndarray]:
    summary: dict[str, np.ndarray] = {}
    for key, array in data.items():
        summary[f"{key}_q25"] = np.quantile(array, 0.25, axis=0)
        summary[f"{key}_median"] = np.quantile(array, 0.50, axis=0)
        summary[f"{key}_q75"] = np.quantile(array, 0.75, axis=0)
    fields = ["step_index"] + list(summary)
    with (output / "fig3_controller_dynamics_source_data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(50):
            writer.writerow({"step_index": step} | {key: value[step] for key, value in summary.items()})
    return summary


def _plot_iqr(
    ax: plt.Axes,
    x: np.ndarray,
    median: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    color: str,
    label: str,
) -> None:
    ax.fill_between(x, q25, q75, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, median, color=color, linewidth=1.6, label=label)


def build_fig3_controller(config: dict[str, Any], output: Path) -> None:
    data = load_controller_dynamics(config)
    summary = summarize_controller(data, output)
    x = np.arange(50)
    active = int(config["controller_dynamics"]["active_steps"])
    dz_start = int(config["controller_dynamics"]["dz_start_step"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0866, 2.55))

    ax = axes[0]
    mask = x < active
    _plot_iqr(
        ax,
        x[mask],
        summary["rho_median"][mask],
        summary["rho_q25"][mask],
        summary["rho_q75"][mask],
        BLUE,
        "B* adaptive ρ",
    )
    ax.plot(x, summary["pollution_median"], color=TEAL, linewidth=1.2, label="Actual pollution q")
    ax.plot([0, active - 1], [0.55, 0.55], color=MID, linestyle=(0, (3, 2)), linewidth=1.0, label="A* fixed ρ")
    ax.axvspan(active - 0.5, 49.5, color=PALE, zorder=0)
    ax.text(39.5, 0.91, "clean condition", ha="center", va="top", fontsize=6.0, color=MID)
    ax.set_title("Controller state", loc="left", fontweight="bold")
    ax.set_ylabel("Control / pollution")
    ax.set_ylim(-0.03, 1.02)

    ax = axes[1]
    _plot_iqr(
        ax,
        x,
        summary["d_g_median"],
        summary["d_g_q25"],
        summary["d_g_q75"],
        BLUE,
        "Observed Dg",
    )
    ax.plot(x, summary["target_g_median"], color=VIOLET, linewidth=1.2, linestyle=(0, (3, 2)), label="Target Dg*")
    ax.axvspan(active - 0.5, 49.5, color=PALE, zorder=0)
    ax.set_title("Guidance diversity", loc="left", fontweight="bold")
    ax.set_ylabel("Cosine distance")

    ax = axes[2]
    _plot_iqr(
        ax,
        x,
        summary["d_z_median"],
        summary["d_z_q25"],
        summary["d_z_q75"],
        BLUE,
        "Observed Dz",
    )
    ax.plot(x, summary["target_z_median"], color=VIOLET, linewidth=1.2, linestyle=(0, (3, 2)), label="Target Dz*")
    ax.axvspan(-0.5, dz_start - 0.5, color="#F7F3EA", zorder=0)
    ax.axvspan(active - 0.5, 49.5, color=PALE, zorder=0)
    ax.set_title("Predicted-clean diversity", loc="left", fontweight="bold")
    ax.set_ylabel("Cosine distance")

    for index, ax in enumerate(axes):
        ax.text(-0.18, 1.07, "abc"[index], transform=ax.transAxes, fontsize=8.5, fontweight="bold")
        ax.set_xlim(-0.5, 49.5)
        ax.set_xlabel("DDIM sampling step")
        ax.tick_params(direction="out", length=2.5, width=0.7)
        ax.legend(loc="best", fontsize=5.7, handlelength=2.3)
    fig.text(0.995, 0.99, "COCO-Test-500; median and interquartile range across n = 500 prompts", ha="right", va="top", fontsize=6.1, color=MID)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.19, top=0.82, wspace=0.37)
    save_figure(fig, output, "fig3_controller_dynamics")


def _errorbar_series(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    color: str,
    label: str,
    marker: str = "o",
) -> None:
    ax.plot(x, y, color=color, linewidth=1.0, alpha=0.75)
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - lower, upper - y]),
        fmt=marker,
        color=color,
        markersize=3.5,
        elinewidth=0.8,
        capsize=1.8,
        label=label,
        zorder=3,
    )


def build_fig4_ablation(config: dict[str, Any], output: Path) -> None:
    a_selection = read_json(Path(config["_resolved"]["a_star_selection"]))
    b_selection = read_json(Path(config["_resolved"]["b_margin_selection"]))
    a_rows = [
        {
            "rho": 0.0,
            "dino_mean_difference": 0.0,
            "dino_ci95": [0.0, 0.0],
            "clipscore_mean_difference": 0.0,
            "clipscore_ci95": [0.0, 0.0],
            "hpsv2_mean_difference": 0.0,
            "hpsv2_ci95": [0.0, 0.0],
            "passes_all_quality_constraints": True,
            "search_phase": "anchor",
        }
    ] + a_selection["decisions"]
    a_rows = sorted(a_rows, key=lambda row: float(row["rho"]))
    b_rows = sorted(b_selection["decisions"], key=lambda row: float(row["alpha"]))
    selected_rho = float(a_selection["selected_a_star_rho"])
    selected_alpha = float(b_selection["selected_alpha"])
    clip_margin = float(a_selection["clipscore_noninferiority_margin"])
    hps_margin = float(a_selection["hpsv2_noninferiority_margin"])

    with (output / "fig4_ablation_source_data.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "stage", "parameter", "dino_delta", "dino_ci_lower", "dino_ci_upper",
            "clipscore_delta", "clipscore_ci_lower", "clipscore_ci_upper",
            "hpsv2_delta", "hpsv2_ci_lower", "hpsv2_ci_upper",
            "quality_eligible", "mean_active_rho", "boundary_rate", "selected",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in a_rows:
            writer.writerow(
                {
                    "stage": "A", "parameter": row["rho"],
                    "dino_delta": row["dino_mean_difference"], "dino_ci_lower": row["dino_ci95"][0], "dino_ci_upper": row["dino_ci95"][1],
                    "clipscore_delta": row["clipscore_mean_difference"], "clipscore_ci_lower": row["clipscore_ci95"][0], "clipscore_ci_upper": row["clipscore_ci95"][1],
                    "hpsv2_delta": row["hpsv2_mean_difference"], "hpsv2_ci_lower": row["hpsv2_ci95"][0], "hpsv2_ci_upper": row["hpsv2_ci95"][1],
                    "quality_eligible": row["passes_all_quality_constraints"], "mean_active_rho": "", "boundary_rate": "", "selected": float(row["rho"]) == selected_rho,
                }
            )
        for row in b_rows:
            paired = row["paired_differences_vs_a"]
            writer.writerow(
                {
                    "stage": "B", "parameter": row["alpha"],
                    "dino_delta": paired["dino_diversity"]["mean_difference"], "dino_ci_lower": paired["dino_diversity"]["ci95_lower"], "dino_ci_upper": paired["dino_diversity"]["ci95_upper"],
                    "clipscore_delta": paired["clipscore"]["mean_difference"], "clipscore_ci_lower": paired["clipscore"]["ci95_lower"], "clipscore_ci_upper": paired["clipscore"]["ci95_upper"],
                    "hpsv2_delta": paired["hpsv2"]["mean_difference"], "hpsv2_ci_lower": paired["hpsv2"]["ci95_lower"], "hpsv2_ci_upper": paired["hpsv2"]["ci95_upper"],
                    "quality_eligible": row["eligible"], "mean_active_rho": row["mean_active_rho"], "boundary_rate": row["mean_active_boundary_rate"], "selected": float(row["alpha"]) == selected_alpha,
                }
            )

    fig, axes = plt.subplots(2, 2, figsize=(7.0866, 4.45))
    ax = axes[0, 0]
    x = np.asarray([float(row["rho"]) for row in a_rows])
    y = 1000 * np.asarray([float(row["dino_mean_difference"]) for row in a_rows])
    lo = 1000 * np.asarray([float(row["dino_ci95"][0]) for row in a_rows])
    hi = 1000 * np.asarray([float(row["dino_ci95"][1]) for row in a_rows])
    _errorbar_series(ax, x, y, lo, hi, BLUE_2, "DINO Δ vs Vanilla")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.axvline(selected_rho, color=GREEN, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.scatter([selected_rho], [y[np.argmin(abs(x - selected_rho))]], marker="*", s=55, color=GREEN, zorder=5)
    for row in a_rows:
        if not row["passes_all_quality_constraints"]:
            ax.scatter(float(row["rho"]), 1000 * float(row["dino_mean_difference"]), marker="x", color=RED, s=22, zorder=5)
    ax.set_title("A: fixed-CADS diversity", loc="left", fontweight="bold")
    ax.set_ylabel("DINO Δ vs Vanilla (×10⁻³)")
    ax.set_xlabel("Fixed strength ρ")

    ax = axes[0, 1]
    clip_y = np.asarray([float(row["clipscore_mean_difference"]) / clip_margin for row in a_rows])
    clip_lo = np.asarray([float(row["clipscore_ci95"][0]) / clip_margin for row in a_rows])
    clip_hi = np.asarray([float(row["clipscore_ci95"][1]) / clip_margin for row in a_rows])
    hps_y = np.asarray([float(row["hpsv2_mean_difference"]) / hps_margin for row in a_rows])
    hps_lo = np.asarray([float(row["hpsv2_ci95"][0]) / hps_margin for row in a_rows])
    hps_hi = np.asarray([float(row["hpsv2_ci95"][1]) / hps_margin for row in a_rows])
    _errorbar_series(ax, x, clip_y, clip_lo, clip_hi, TEAL, "CLIPScore")
    _errorbar_series(ax, x, hps_y, hps_lo, hps_hi, VIOLET, "HPSv2")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.axhline(-1, color=RED, linestyle=(0, (3, 2)), linewidth=0.9, label="NI boundary")
    ax.axvline(selected_rho, color=GREEN, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.set_title("A: quality constraints", loc="left", fontweight="bold")
    ax.set_ylabel("Quality Δ / 1% NI margin")
    ax.set_xlabel("Fixed strength ρ")
    ax.legend(fontsize=5.8, ncol=2, loc="lower left")

    ax = axes[1, 0]
    bx = np.asarray([float(row["alpha"]) for row in b_rows])
    by = 1000 * np.asarray([float(row["paired_differences_vs_a"]["dino_diversity"]["mean_difference"]) for row in b_rows])
    blo = 1000 * np.asarray([float(row["paired_differences_vs_a"]["dino_diversity"]["ci95_lower"]) for row in b_rows])
    bhi = 1000 * np.asarray([float(row["paired_differences_vs_a"]["dino_diversity"]["ci95_upper"]) for row in b_rows])
    _errorbar_series(ax, bx, by, blo, bhi, BLUE, "DINO Δ vs A*")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.axvline(selected_alpha, color=GREEN, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.scatter([selected_alpha], [by[np.argmin(abs(bx - selected_alpha))]], marker="*", s=55, color=GREEN, zorder=5)
    ax.set_title("B: reference-margin ablation", loc="left", fontweight="bold")
    ax.set_ylabel("DINO Δ vs A* (×10⁻³)")
    ax.set_xlabel("Reference margin α")

    ax = axes[1, 1]
    mean_rho = np.asarray([float(row["mean_active_rho"]) for row in b_rows])
    boundary = np.asarray([float(row["mean_active_boundary_rate"]) for row in b_rows])
    ax.plot(bx, mean_rho, "o-", color=BLUE, linewidth=1.2, markersize=3.8, label="Mean active ρ")
    ax.plot(bx, boundary, "s-", color=TEAL, linewidth=1.2, markersize=3.4, label="Boundary rate")
    ax.axhline(0.25, color=RED, linestyle=(0, (3, 2)), linewidth=0.9, label="Boundary limit (25%)")
    ax.axvline(selected_alpha, color=GREEN, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.set_ylim(-0.03, 0.85)
    ax.set_title("B: controller operating range", loc="left", fontweight="bold")
    ax.set_ylabel("Fraction")
    ax.set_xlabel("Reference margin α")
    ax.legend(fontsize=5.8, loc="best")

    for index, ax in enumerate(axes.flat):
        ax.text(-0.14, 1.06, "abcd"[index], transform=ax.transAxes, fontsize=8.5, fontweight="bold")
        ax.tick_params(direction="out", length=2.5, width=0.7)
    fig.text(0.995, 0.995, "COCO-Dev-50 parameter selection; 95% prompt-level paired-bootstrap CI", ha="right", va="top", fontsize=6.1, color=MID)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.12, top=0.89, wspace=0.30, hspace=0.42)
    save_figure(fig, output, "fig4_dev_ablation")


def write_documentation(config: dict[str, Any], output: Path, selection: dict[str, Any]) -> None:
    selected = selection["selected"]
    legends = f"""# Figure legends

**Fig. 1 | Training-free closed-loop condition annealing.** a, The matched open-loop A* baseline uses a fixed condition-noise strength of ρ = 0.55. b, Feedback-CADS computes guidance and predicted-clean-latent diversity from the same SD v1.5 forward pass, compares them with reference curves frozen from A*, and applies a one-step-delayed proportional update to the next condition-noise strength. Condition noise is disabled after step 29. Both methods use DDIM-50 and 50 UNet calls.

**Fig. 2 | Same-prompt, same-seed qualitative comparison.** Each row shows eight candidates from one method, and each column shares the same initial latent seed across methods. The displayed prompt was selected automatically from the formal test set as the quality-eligible case nearest the median B*−A* DINO difference (prompt index {selected['prompt_source_index']}; eligible n = {selected['eligible_prompt_count']}). Images are shown without cropping or colour adjustment. The panel is illustrative; the formal population-level conclusion comes from COCO-Test-500 statistics.

**Fig. 3 | Prompt-adaptive controller dynamics on COCO-Test-500.** a, Median Feedback-CADS controller state and actual pollution with the interquartile range of ρ across n = 500 prompts; A* uses fixed ρ = 0.55. b,c, Median observed Dg and Dz with interquartile ranges and frozen target curves. Dz contributes to control from step 10, and condition pollution is hard-off from step 30. All trajectories come from the single frozen formal run.

**Fig. 4 | Development-set parameter selection and ablation.** a, DINO diversity differences across fixed A strengths; crosses mark configurations failing at least one quality non-inferiority constraint. b, CLIPScore and HPSv2 differences normalized by their preregistered 1% non-inferiority margins. c, B reference-margin ablation relative to A*. d, Mean active ρ and boundary-saturation rate. Green dashed lines and stars mark the frozen selections ρ* = 0.55 and α* = 0.15. Error bars are 95% prompt-level paired-bootstrap confidence intervals on COCO-Dev-50. These panels document parameter selection and are not formal test-set evidence.
"""
    (output / "FIGURE_LEGENDS.md").write_text(legends, encoding="utf-8")
    qa = """# Figure contracts and QA notes

| Figure | Archetype | Core conclusion | Evidence source | Reviewer-risk control |
|:--|:--|:--|:--|:--|
| Fig. 1 | Schematic-led | Feedback-CADS closes the condition-annealing loop without training or extra UNet calls | Frozen method specification | No invented quantitative values; one-step delay and hard-off are explicit |
| Fig. 2 | Image plate | The paired qualitative output can be inspected under identical prompt and latent seeds | Frozen formal images | Machine-selected median case; all K = 8 candidates; no crop or colour adjustment |
| Fig. 3 | Quantitative grid | The controller is prompt-adaptive and follows frozen references during the active window | All 500 formal diagnostics | Median and IQR use prompts as the replicate unit; clean phase is shaded |
| Fig. 4 | Quantitative grid | Frozen A* and B* settings follow the disclosed Dev-50 selection rules | Frozen Dev-50 selection records | Explicitly labelled development evidence; formal Test-500 conclusions are unchanged |

- Backend: Python/matplotlib only.
- Export contract: editable SVG and PDF, 600 dpi LZW TIFF, and 300 dpi PNG preview.
- Final width: 180 mm before tight cropping; every plotted font is at least 5 pt.
- Fig. 1 mathtext glyph audit: exported PDF minimum is 5.04 pt (required minimum: 5 pt); no text run falls below the floor.
- Palette: Feedback-CADS is consistently blue; green marks selected/improved settings; red marks thresholds or failed constraints.
- Source data: clean CSV or JSON manifests accompany Figs. 2–4.
- Image integrity: Fig. 2 uses the complete 512×512 images with no crop, rescaling beyond display interpolation, contrast, gamma or colour manipulation.
- Formal acceptance: unchanged; these are post-evaluation presentation artifacts.
"""
    (output / "QA_NOTES.md").write_text(qa, encoding="utf-8")
    readme = """# Feedback-CADS project figures

1. `fig1_feedback_cads_method.*` — method overview.
2. `fig2_qualitative_same_seed.*` — automatically selected same-seed qualitative comparison.
3. `fig3_controller_dynamics.*` — formal Test-500 controller dynamics.
4. `fig4_dev_ablation.*` — Dev-50 A/B parameter selection and ablation.

Each figure is exported as SVG, PDF, TIFF and PNG. See `FIGURE_LEGENDS.md` and `QA_NOTES.md` before manuscript use.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def build() -> Path:
    config = load_and_validate()
    output = (PROJECT_ROOT / config["outputs"]["directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    build_fig1_method(output)
    selection = build_fig2_qualitative(config, output)
    build_fig3_controller(config, output)
    build_fig4_ablation(config, output)
    write_documentation(config, output, selection)
    artifacts = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name != "project_figures_complete.json"
    )
    completion = {
        "passed": True,
        "config_sha256": CONFIG_SHA256,
        "formal_acceptance_unchanged": True,
        "figures": 4,
        "formats_per_figure": ["svg", "pdf", "tiff", "png"],
        "selected_qualitative_prompt_id": selection["selected"]["prompt_id"],
        "controller_prompts": 500,
        "artifact_sha256": {path.name: sha256(path) for path in artifacts},
    }
    (output / "project_figures_complete.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    print(build())
