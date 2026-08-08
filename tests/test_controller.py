"""Tests for the frozen Stage-B reference and proportional controller."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from feedback_cads import (
    FeedbackMVPConfig,
    GroupProportionalController,
    load_frozen_diversity_reference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    PROJECT_ROOT
    / "results/development/reference_curves.json"
)
REFERENCE_SHA256 = (
    "c2ae2024f06c699eb9e51e063ab9aea2f118c03663d6918965c1e639f9667382"
)


def test_frozen_reference_loads_with_exact_hash_and_protocol() -> None:
    reference = load_frozen_diversity_reference(
        REFERENCE_PATH,
        expected_sha256=REFERENCE_SHA256,
    )
    assert reference.step_count == 50
    assert reference.reference_margin_alpha == 0.0
    assert reference.rho_a_star == pytest.approx(0.55)
    assert reference.dz_control_eligible[:10] == (False,) * 10
    assert reference.dz_control_eligible[10:] == (True,) * 40
    reference.validate_step(
        0,
        scheduler_timestep=981.0,
        progress=0.0,
        pollution_cap=1.0,
    )
    with pytest.raises(ValueError, match="scheduler_timestep mismatch"):
        reference.validate_step(
            0,
            scheduler_timestep=980.0,
            progress=0.0,
            pollution_cap=1.0,
        )


def test_reference_rejects_test_leakage_and_wrong_hash(tmp_path: Path) -> None:
    data = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    data["formal_test_data_used"] = True
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="formal test data"):
        load_frozen_diversity_reference(bad_path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_frozen_diversity_reference(
            REFERENCE_PATH,
            expected_sha256="0" * 64,
        )


def _controller(group_count: int = 1) -> GroupProportionalController:
    return GroupProportionalController(
        FeedbackMVPConfig(
            initial_rho=0.55,
            k_diversity=0.08,
            epsilon=1e-6,
            reference_margin_alpha=0.0,
        ),
        group_count=group_count,
        device="cpu",
    )


def test_control_direction_and_one_step_state_transition() -> None:
    controller = _controller()
    rho_before_observation = controller.rho
    update = controller.update(
        guidance_diversity=torch.tensor([0.20]),
        predicted_clean_latent_diversity=torch.tensor([0.50]),
        guidance_reference=0.40,
        predicted_clean_latent_reference=0.40,
        dz_control_eligible=False,
        pollution_cap=1.0,
    )
    assert update.rho_used.item() == pytest.approx(0.55)
    assert torch.equal(rho_before_observation, update.rho_used)
    assert update.diversity_deficit.item() > 0.0
    assert update.rho_next.item() > update.rho_used.item()
    assert controller.rho.item() == pytest.approx(update.rho_next.item())

    second = controller.update(
        guidance_diversity=torch.tensor([0.60]),
        predicted_clean_latent_diversity=torch.tensor([0.60]),
        guidance_reference=0.40,
        predicted_clean_latent_reference=0.40,
        dz_control_eligible=True,
        pollution_cap=0.5,
    )
    assert second.rho_used.item() == pytest.approx(update.rho_next.item())
    assert second.diversity_deficit.item() < 0.0
    assert second.rho_next.item() < second.rho_used.item()


def test_dz_is_ignored_before_threshold_and_maximized_afterward() -> None:
    observations = {
        "guidance_diversity": torch.tensor([0.50]),
        "predicted_clean_latent_diversity": torch.tensor([0.10]),
        "guidance_reference": 0.40,
        "predicted_clean_latent_reference": 0.40,
        "pollution_cap": 1.0,
    }
    early = _controller().update(
        **observations,
        dz_control_eligible=False,
    )
    late = _controller().update(
        **observations,
        dz_control_eligible=True,
    )
    assert early.diversity_deficit.item() < 0.0
    assert early.rho_next.item() < early.rho_used.item()
    assert late.diversity_deficit.item() > 0.0
    assert late.rho_next.item() > late.rho_used.item()


def test_groups_update_independently_and_clip_to_unit_interval() -> None:
    controller = GroupProportionalController(
        FeedbackMVPConfig(initial_rho=0.5, k_diversity=2.0),
        group_count=2,
        device="cpu",
    )
    update = controller.update(
        guidance_diversity=torch.tensor([0.0, 1.0]),
        predicted_clean_latent_diversity=torch.tensor([0.0, 1.0]),
        guidance_reference=0.5,
        predicted_clean_latent_reference=0.5,
        dz_control_eligible=True,
        pollution_cap=1.0,
    )
    torch.testing.assert_close(update.rho_next, torch.tensor([1.0, 0.0]))


def test_hard_off_stops_updates_exactly() -> None:
    controller = _controller(group_count=2)
    update = controller.update(
        guidance_diversity=torch.tensor([0.0, 1.0]),
        predicted_clean_latent_diversity=torch.tensor([0.0, 1.0]),
        guidance_reference=0.5,
        predicted_clean_latent_reference=0.5,
        dz_control_eligible=True,
        pollution_cap=0.0,
    )
    assert not update.control_updated
    assert torch.equal(update.delta_rho, torch.zeros(2))
    assert torch.equal(update.rho_next, update.rho_used)
    assert torch.equal(controller.rho, torch.full((2,), 0.55))


def test_alpha_zero_uses_unmodified_reference_and_shapes_are_checked() -> None:
    controller = _controller(group_count=2)
    with pytest.raises(ValueError, match="one value per prompt group"):
        controller.update(
            guidance_diversity=torch.tensor([0.2]),
            predicted_clean_latent_diversity=torch.tensor([0.2]),
            guidance_reference=0.4,
            predicted_clean_latent_reference=0.4,
            dz_control_eligible=False,
            pollution_cap=1.0,
        )
    valid = controller.update(
        guidance_diversity=torch.tensor([0.4, 0.4]),
        predicted_clean_latent_diversity=torch.tensor([0.4, 0.4]),
        guidance_reference=0.4,
        predicted_clean_latent_reference=0.4,
        dz_control_eligible=True,
        pollution_cap=1.0,
    )
    assert valid.guidance_target == pytest.approx(0.4)
    assert valid.predicted_clean_latent_target == pytest.approx(0.4)
    torch.testing.assert_close(
        valid.diversity_deficit,
        torch.zeros(2),
        atol=3e-6,
        rtol=0,
    )


def test_positive_alpha_scales_targets_only_in_controller() -> None:
    controller = GroupProportionalController(
        FeedbackMVPConfig(reference_margin_alpha=0.05),
        group_count=1,
        device="cpu",
    )
    update = controller.update(
        guidance_diversity=torch.tensor([0.4]),
        predicted_clean_latent_diversity=torch.tensor([0.2]),
        guidance_reference=0.4,
        predicted_clean_latent_reference=0.2,
        dz_control_eligible=True,
        pollution_cap=1.0,
    )
    assert update.guidance_target == pytest.approx(0.42)
    assert update.predicted_clean_latent_target == pytest.approx(0.21)
    assert update.diversity_deficit.item() > 0.0
    assert update.rho_next.item() > update.rho_used.item()
