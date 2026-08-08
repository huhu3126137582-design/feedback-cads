#!/usr/bin/env python3
"""Unified stage-oriented CLI for the Feedback-CADS project."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
STAGE_A_CONFIG = "configs/a_strength_search_coco_dev50.yaml"
STAGE_A_OUTPUTS = {
    "dev20": "outputs/a_strength_curve_dev20",
    "dev50_confirmation": "outputs/a_strength_curve_dev50_sd15",
    "coarse": "outputs/a_strength_curve_coco_dev50_coarse",
    "fine": "outputs/a_strength_curve_coco_dev50_fine",
}
STAGE_B_PROFILES = (
    "alpha_0p00",
    "alpha_0p05",
    "alpha_0p10",
    "alpha_0p15",
    "alpha_0p20",
)


def run_script(name: str, *arguments: str) -> None:
    command = [
        PYTHON,
        str(PROJECT_ROOT / "scripts" / "_steps" / name),
        *arguments,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "One public entry point grouped by Stage A, Stage B, formal "
            "evaluation, and publication. Unknown trailing options are "
            "forwarded to the selected implementation step."
        )
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    a = stages.add_parser("stage-a")
    a.add_argument(
        "action",
        choices=("run", "evaluate", "audit", "select", "reference", "audit-reference"),
    )
    a.add_argument(
        "--profile",
        choices=tuple(STAGE_A_OUTPUTS),
        default="coarse",
    )

    b = stages.add_parser("stage-b")
    b.add_argument(
        "action",
        choices=("run", "evaluate", "select", "signal-audit", "controller-smoke"),
    )
    b.add_argument("--profile", choices=STAGE_B_PROFILES, default="alpha_0p00")
    b.add_argument("--extended", action="store_true")

    formal = stages.add_parser("formal")
    formal.add_argument(
        "action",
        choices=(
            "validate",
            "generation-smoke",
            "generation",
            "audit-generation",
            "evaluate",
            "audit-evaluation",
            "report",
            "audit-report",
        ),
    )

    publication = stages.add_parser("publication")
    publication.add_argument(
        "action", choices=("results", "figures", "package", "all")
    )

    data = stages.add_parser("data")
    data.add_argument("action", choices=("build-coco-splits",))

    verify = stages.add_parser("verify")
    verify.add_argument(
        "--skip-image-audit",
        action="store_true",
        help="Skip the 16,000-PNG integrity replay; all other checks still run.",
    )
    return parser.parse_known_args()


def stage_a(args: argparse.Namespace, extra: list[str]) -> None:
    if args.action == "run":
        run_script(
            "run_a_strength_curve.py",
            "--config",
            STAGE_A_CONFIG,
            "--profile",
            args.profile,
            *extra,
        )
    elif args.action in {"evaluate", "audit"}:
        script = (
            "evaluate_a_strength_curve.py"
            if args.action == "evaluate"
            else "audit_a_strength_curve.py"
        )
        run_script(script, "--experiment-dir", STAGE_A_OUTPUTS[args.profile], *extra)
    elif args.action == "select":
        run_script("select_parameters.py", "a-star", *extra)
    elif args.action == "reference":
        run_script("build_a_star_reference.py", *extra)
    else:
        run_script("audit_b_reference_signals.py", *extra)


def stage_b(args: argparse.Namespace, extra: list[str]) -> None:
    if args.action == "run":
        run_script("run_b_feedback_coco_dev50.py", "--profile", args.profile, *extra)
    elif args.action == "evaluate":
        run_script("evaluate_b_vs_a_coco_dev50.py", "--profile", args.profile, *extra)
    elif args.action == "select":
        arguments = ["b-margin"]
        if args.extended:
            arguments.append("--extended")
        run_script("select_parameters.py", *arguments, *extra)
    elif args.action == "signal-audit":
        run_script("audit_b_reference_signals.py", *extra)
    else:
        run_script("audit_b_controller_smoke.py", *extra)


def formal(args: argparse.Namespace, extra: list[str]) -> None:
    mapping = {
        "validate": ("validate_formal_test_protocol.py", ["--allow-started"]),
        "generation-smoke": ("run_frozen_formal_generation.py", ["--mode", "smoke"]),
        "generation": ("run_frozen_formal_generation.py", ["--mode", "formal"]),
        "audit-generation": (
            "audit_formal_generation.py",
            ["--allow-downstream-artifacts"],
        ),
        "evaluate": ("evaluate_formal_coco_test500.py", []),
        "audit-evaluation": ("audit_formal_evaluation.py", []),
        "report": ("build_formal_report.py", []),
        "audit-report": ("audit_formal_report.py", []),
    }
    script, arguments = mapping[args.action]
    run_script(script, *arguments, *extra)


def publication(args: argparse.Namespace, extra: list[str]) -> None:
    if args.action in {"results", "all"}:
        run_script("build_publication_results.py", *extra)
    if args.action in {"figures", "all"}:
        run_script("build_project_figures.py", *extra)
    if args.action in {"package", "all"}:
        run_script("export_delivery_results.py", *extra)


def data(args: argparse.Namespace, extra: list[str]) -> None:
    if args.action == "build-coco-splits":
        run_script("build_coco_caption_splits.py", *extra)


def verify(args: argparse.Namespace, extra: list[str]) -> None:
    if extra:
        raise ValueError(f"verify does not accept forwarded arguments: {extra}")
    run_script("validate_formal_test_protocol.py", "--allow-started")
    if not args.skip_image_audit:
        run_script("audit_formal_generation.py", "--allow-downstream-artifacts")
    run_script("audit_formal_evaluation.py")
    run_script("audit_formal_report.py")
    subprocess.run([PYTHON, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args, extra = parse_args()
    if args.stage == "stage-a":
        stage_a(args, extra)
    elif args.stage == "stage-b":
        stage_b(args, extra)
    elif args.stage == "formal":
        formal(args, extra)
    elif args.stage == "publication":
        publication(args, extra)
    elif args.stage == "data":
        data(args, extra)
    else:
        verify(args, extra)


if __name__ == "__main__":
    main()
