"""GPU integration checks for the SD v1.5 Original CADS adaptation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from feedback_cads import (
    CleanUnconditionalCADSConfig,
    OriginalCADSStableDiffusionPipeline,
    PaperCADSConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "models/huggingface/hub/"
    "models--stable-diffusion-v1-5--stable-diffusion-v1-5/"
    "snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)


def _model_dir() -> Path:
    return Path(os.environ.get("SD15_MODEL_DIR", DEFAULT_MODEL_DIR))


@pytest.mark.gpu
@pytest.mark.integration
def test_sd15_original_cads_audit_and_reproducibility() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the SD v1.5 integration test.")
    model_dir = _model_dir()
    required = [
        model_dir / "model_index.json",
        model_dir / "unet/diffusion_pytorch_model.fp16.safetensors",
        model_dir / "vae/diffusion_pytorch_model.fp16.safetensors",
        model_dir / "text_encoder/model.fp16.safetensors",
    ]
    if not all(path.exists() for path in required):
        pytest.skip(f"Complete local SD v1.5 cache not found at {model_dir}")

    official = StableDiffusionPipeline.from_pretrained(
        model_dir,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    official.scheduler = DDIMScheduler.from_config(official.scheduler.config)
    official.to(device="cuda", dtype=torch.float16)
    official.set_progress_bar_config(disable=True)
    custom = OriginalCADSStableDiffusionPipeline.from_vanilla_pipeline(
        official
    )
    custom.set_progress_bar_config(disable=True)

    latent_generator = torch.Generator(device="cuda").manual_seed(20260807)
    initial_latents = torch.randn(
        (1, 4, 64, 64),
        generator=latent_generator,
        device="cuda",
        dtype=torch.float16,
    )
    short_kwargs = {
        "prompt": "a red fox in a forest",
        "height": 512,
        "width": 512,
        "num_inference_steps": 10,
        "guidance_scale": 7.5,
        "eta": 0.0,
        "output_type": "latent",
    }
    reference = official(
        **short_kwargs,
        latents=initial_latents.clone(),
    ).images
    vanilla_custom = custom(
        **short_kwargs,
        latents=initial_latents.clone(),
    ).images
    assert torch.equal(vanilla_custom, reference)

    zero_a = custom(
        **short_kwargs,
        latents=initial_latents.clone(),
        clean_unconditional_cads_config=CleanUnconditionalCADSConfig(
            fixed_rho_a=0.0
        ),
        condition_seed=701,
    ).images
    zero_a_stats = dict(custom.last_run_stats)
    zero_a_other_condition_seed = custom(
        **short_kwargs,
        latents=initial_latents.clone(),
        clean_unconditional_cads_config=CleanUnconditionalCADSConfig(
            fixed_rho_a=0.0
        ),
        condition_seed=999,
    ).images
    assert torch.equal(zero_a, reference)
    assert torch.equal(zero_a_other_condition_seed, reference)
    assert zero_a_stats["pollution_history"] == [0.0] * 10
    assert zero_a_stats["positive_condition_max_delta"] == 0.0
    assert zero_a_stats["negative_condition_max_delta"] == 0.0
    assert zero_a_stats["positive_condition_noise_draw_calls"] == 0
    assert zero_a_stats["negative_condition_noise_draw_calls"] == 0

    stage_a_config = CleanUnconditionalCADSConfig()
    stage_a = custom(
        **short_kwargs,
        latents=initial_latents.clone(),
        clean_unconditional_cads_config=stage_a_config,
        condition_seed=702,
    ).images
    stage_a_stats = dict(custom.last_run_stats)
    stage_a_replay = custom(
        **short_kwargs,
        latents=initial_latents.clone(),
        clean_unconditional_cads_config=stage_a_config,
        condition_seed=702,
    ).images
    stage_a_other_seed = custom(
        **short_kwargs,
        latents=initial_latents.clone(),
        clean_unconditional_cads_config=stage_a_config,
        condition_seed=703,
    ).images

    assert torch.equal(stage_a, stage_a_replay)
    assert not torch.equal(stage_a, stage_a_other_seed)
    assert not torch.equal(stage_a, vanilla_custom)
    assert torch.isfinite(stage_a).all()
    assert stage_a_stats["mode"] == "clean_unconditional_cads"
    assert stage_a_stats["positive_condition_max_delta"] > 0.0
    assert stage_a_stats["negative_condition_max_delta"] == 0.0
    assert stage_a_stats["non_content_positive_max_delta"] == 0.0
    assert stage_a_stats["negative_condition_seed"] is None
    assert stage_a_stats["condition_seed"] == 702
    assert stage_a_stats["fixed_rho_a"] == pytest.approx(0.55)
    assert stage_a_stats["pollution_cap_history"][0] == pytest.approx(1.0)
    assert stage_a_stats["pollution_cap_history"][-1] == pytest.approx(0.0)
    assert stage_a_stats["pollution_history"][0] == pytest.approx(0.55)
    assert stage_a_stats["pollution_history"][-1] == pytest.approx(0.0)
    assert all(
        left >= right
        for left, right in zip(
            stage_a_stats["pollution_history"],
            stage_a_stats["pollution_history"][1:],
        )
    )
    assert stage_a_stats["progress_history"][0] == pytest.approx(0.0)
    assert stage_a_stats["progress_history"][-1] == pytest.approx(1.0)
    assert stage_a_stats["content_token_count_per_sample"] == [6]

    config = PaperCADSConfig()
    cads_a = custom(
        **short_kwargs,
        latents=initial_latents.clone(),
        original_cads_config=config,
        condition_seed=501,
    ).images
    stats = dict(custom.last_run_stats)
    cads_replay = custom(
        **short_kwargs,
        latents=initial_latents.clone(),
        original_cads_config=config,
        condition_seed=501,
    ).images
    cads_other_seed = custom(
        **short_kwargs,
        latents=initial_latents.clone(),
        original_cads_config=config,
        condition_seed=502,
    ).images

    assert torch.equal(cads_a, cads_replay)
    assert not torch.equal(cads_a, cads_other_seed)
    assert not torch.equal(cads_a, vanilla_custom)
    assert torch.isfinite(cads_a).all()
    assert stats["positive_condition_max_delta"] > 0.0
    assert stats["negative_condition_max_delta"] > 0.0
    assert stats["condition_seed"] != stats["negative_condition_seed"]
    assert stats["pollution_history"][0] == pytest.approx(1.0)
    assert stats["pollution_history"][-1] == pytest.approx(0.0)
    assert all(
        left >= right
        for left, right in zip(
            stats["pollution_history"],
            stats["pollution_history"][1:],
        )
    )
    assert stats["timestep_history"][0] > stats["timestep_history"][-1]
    assert stats["paper_source_protocol"] == {
        "model": "stable-diffusion-v2-1",
        "sampler": "DDPM",
        "num_steps": 100,
        "cfg_scale": 9.0,
    }
    assert stats["project_target_protocol"] == {
        "model": "stable-diffusion-v1-5",
        "sampler": "DDIM",
        "num_steps": 50,
        "cfg_scale": 7.5,
    }

    full_result = custom(
        **{**short_kwargs, "num_inference_steps": 50},
        latents=initial_latents.clone(),
        original_cads_config=config,
        condition_seed=503,
    ).images
    full_stats = custom.last_run_stats
    assert torch.isfinite(full_result).all()
    assert full_stats["actual_timesteps"] == 50
    assert full_stats["unet_calls"] == 50
    assert full_stats["scheduler_step_calls"] == 50
    assert full_stats["grad_enabled_at_entry"] is False
    assert full_stats["grad_enabled_during_loop"] is False

    full_stage_a = custom(
        **{**short_kwargs, "num_inference_steps": 50},
        latents=initial_latents.clone(),
        clean_unconditional_cads_config=stage_a_config,
        condition_seed=704,
    ).images
    full_stage_a_stats = custom.last_run_stats
    assert torch.isfinite(full_stage_a).all()
    assert full_stage_a_stats["actual_timesteps"] == 50
    assert full_stage_a_stats["unet_calls"] == 50
    assert full_stage_a_stats["scheduler_step_calls"] == 50
    assert full_stage_a_stats["negative_condition_max_delta"] == 0.0
    assert full_stage_a_stats["pollution_history"][-1] == 0.0

    formal_latent_generator = torch.Generator(device="cuda").manual_seed(
        20260808
    )
    formal_initial_latents = torch.randn(
        (2, 4, 64, 64),
        generator=formal_latent_generator,
        device="cuda",
        dtype=torch.float16,
    )
    formal_kwargs = {
        "prompt": "a red fox in a forest",
        "negative_prompt": "blurry, distorted, low quality",
        "height": 512,
        "width": 512,
        "num_inference_steps": 50,
        "num_images_per_prompt": 2,
        "guidance_scale": 7.5,
        "eta": 0.0,
        "output_type": "latent",
    }
    formal_reference = official(
        **formal_kwargs,
        latents=formal_initial_latents.clone(),
    ).images
    formal_custom_vanilla = custom(
        **formal_kwargs,
        latents=formal_initial_latents.clone(),
    ).images
    cpu_rng_before = torch.random.get_rng_state()
    cuda_rng_before = torch.cuda.get_rng_state()
    formal_q_zero = custom(
        **formal_kwargs,
        latents=formal_initial_latents.clone(),
        clean_unconditional_cads_config=CleanUnconditionalCADSConfig(
            fixed_rho_a=0.0
        ),
        condition_seed=810,
    ).images
    formal_q_zero_stats = custom.last_run_stats

    assert torch.equal(formal_custom_vanilla, formal_reference)
    assert torch.equal(formal_q_zero, formal_reference)
    assert (formal_q_zero - formal_reference).abs().max().item() == 0.0
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)
    assert formal_q_zero_stats["mode"] == "clean_unconditional_cads"
    assert formal_q_zero_stats["actual_timesteps"] == 50
    assert formal_q_zero_stats["unet_calls"] == 50
    assert formal_q_zero_stats["scheduler_step_calls"] == 50
    assert formal_q_zero_stats["pollution_history"] == [0.0] * 50
    assert formal_q_zero_stats["positive_condition_max_delta"] == 0.0
    assert formal_q_zero_stats["negative_condition_max_delta"] == 0.0
    assert formal_q_zero_stats["non_content_positive_max_delta"] == 0.0
    assert formal_q_zero_stats["positive_condition_noise_draw_calls"] == 0
    assert formal_q_zero_stats["negative_condition_noise_draw_calls"] == 0
    assert formal_q_zero_stats["content_token_count_per_sample"] == [6, 6]
