"""Checks for the public stage-oriented CLI dispatch contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import feedback_cads_cli as cli  # noqa: E402


def test_stage_a_dispatches_profile_to_internal_step(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_script", lambda *args: calls.append(args))
    args = argparse.Namespace(action="run", profile="fine")
    cli.stage_a(args, ["--limit-prompts", "1"])
    assert calls == [
        (
            "run_a_strength_curve.py",
            "--config",
            cli.STAGE_A_CONFIG,
            "--profile",
            "fine",
            "--limit-prompts",
            "1",
        )
    ]


def test_formal_post_evaluation_audit_is_explicitly_read_only(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_script", lambda *args: calls.append(args))
    args = argparse.Namespace(action="audit-generation")
    cli.formal(args, [])
    assert calls == [
        ("audit_formal_generation.py", "--allow-downstream-artifacts")
    ]


def test_publication_all_dispatches_both_builders(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_script", lambda *args: calls.append(args))
    cli.publication(argparse.Namespace(action="all"), [])
    assert calls == [
        ("build_publication_results.py",),
        ("build_project_figures.py",),
        ("export_delivery_results.py",),
    ]


def test_a_star_reference_audit_uses_descriptive_step_name(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_script", lambda *args: calls.append(args))
    cli.stage_a(argparse.Namespace(action="audit-reference", profile="fine"), [])
    assert calls == [("audit_b_reference_signals.py",)]


def test_b_controller_smoke_uses_descriptive_step_name(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_script", lambda *args: calls.append(args))
    cli.stage_b(
        argparse.Namespace(
            action="controller-smoke",
            profile="alpha_0p00",
            extended=False,
        ),
        [],
    )
    assert calls == [("audit_b_controller_smoke.py",)]
