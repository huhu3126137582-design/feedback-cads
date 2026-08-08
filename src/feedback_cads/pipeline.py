"""SD v1.5 pipeline with a paper-algorithm-faithful CADS path."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
    retrieve_timesteps,
)

from .condition_noise import (
    CleanUnconditionalCADSConfig,
    ConditionNoiseStream,
    PaperCADSConfig,
    build_content_token_mask,
    corrupt_condition,
    corrupt_cfg_conditions,
    expand_content_token_mask,
    negative_seed_from_positive,
)
from .controller import (
    FeedbackMVPConfig,
    FrozenDiversityReference,
    GroupProportionalController,
)
from .schedules import paper_pollution, progress_pollution_cap
from .signals import grouped_pairwise_cosine_diversity


class OriginalCADSStableDiffusionPipeline(StableDiffusionPipeline):
    """Explicit vanilla, Original CADS, and clean-unconditional A loops.

    The CADS branch follows Algorithm 1: it perturbs the full positive and
    null conditions with independent noise and rescales each branch
    separately. Its source hyperparameters are from the paper's SD v2.1
    experiment; the target model/sampler are SD v1.5 and DDIM in this project.
    """

    @property
    def last_run_stats(self) -> Optional[Dict[str, Any]]:
        """Return diagnostics from the most recent successful call."""

        return getattr(self, "_last_run_stats", None)

    @classmethod
    def from_vanilla_pipeline(
        cls,
        pipeline: StableDiffusionPipeline,
    ) -> "OriginalCADSStableDiffusionPipeline":
        """Share components directly, preserving the caller's FP16 dtype."""

        return cls(
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            safety_checker=pipeline.safety_checker,
            feature_extractor=pipeline.feature_extractor,
            image_encoder=getattr(pipeline, "image_encoder", None),
            requires_safety_checker=pipeline.config.requires_safety_checker,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[
            Union[torch.Generator, List[torch.Generator]]
        ] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        ip_adapter_image: Any = None,
        ip_adapter_image_embeds: Any = None,
        callback_on_step_end: Any = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        original_cads_config: Optional[PaperCADSConfig] = None,
        clean_unconditional_cads_config: Optional[
            CleanUnconditionalCADSConfig
        ] = None,
        content_token_mask: Optional[torch.Tensor] = None,
        condition_seed: int = 0,
        negative_condition_seed: Optional[int] = None,
        collect_group_diversity: bool = False,
        diversity_group_size: Optional[int] = None,
        diversity_pool_size: int = 8,
        feedback_mvp_config: Optional[FeedbackMVPConfig] = None,
        feedback_reference: Optional[FrozenDiversityReference] = None,
        **kwargs: Any,
    ) -> Union[StableDiffusionPipelineOutput, tuple]:
        """Run vanilla, CADS/A, or feedback B."""

        grad_enabled_at_entry = torch.is_grad_enabled()
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            raise NotImplementedError(
                "The audited baseline currently supports text conditioning."
            )
        if callback_on_step_end is not None:
            raise NotImplementedError(
                "Step callbacks are disabled for reproducible auditing."
            )
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        active_feedback_config = feedback_mvp_config
        feedback_configured = (
            feedback_mvp_config is not None
            or feedback_reference is not None
        )
        if (feedback_mvp_config is None) != (feedback_reference is None):
            raise ValueError(
                "Feedback controller config and its frozen reference must "
                "be provided together."
            )
        if feedback_configured and clean_unconditional_cads_config is None:
            raise ValueError(
                "Feedback control requires the clean-unconditional CADS "
                "condition-corruption configuration."
            )
        if feedback_configured and original_cads_config is not None:
            raise ValueError("Feedback control cannot use Original CADS.")
        if (
            original_cads_config is not None
            and clean_unconditional_cads_config is not None
        ):
            raise ValueError(
                "Original CADS and clean-unconditional CADS are mutually "
                "exclusive."
            )
        cads_configured = (
            original_cads_config is not None
            or clean_unconditional_cads_config is not None
        )
        if cads_configured and guidance_scale <= 1.0:
            raise ValueError(
                "CADS auditing requires both CFG branches; use "
                "guidance_scale > 1."
            )
        if (
            clean_unconditional_cads_config is not None
            and prompt is None
            and content_token_mask is None
        ):
            raise ValueError(
                "A content-token mask is required when prompt_embeds are "
                "provided without prompt text."
            )
        diversity_enabled = collect_group_diversity or feedback_configured
        if diversity_enabled:
            if clean_unconditional_cads_config is None:
                raise ValueError(
                    "Group-diversity collection is defined for the A/B "
                    "clean-unconditional path."
                )
            required_group_size = (
                active_feedback_config.group_size
                if active_feedback_config is not None
                else num_images_per_prompt
            )
            if diversity_group_size is None:
                diversity_group_size = required_group_size
            if diversity_group_size != num_images_per_prompt:
                raise ValueError(
                    "diversity_group_size must equal num_images_per_prompt "
                    "to prevent cross-prompt pairs."
                )
            if diversity_group_size < 2:
                raise ValueError(
                    "At least two candidates are required for diversity."
                )
            if active_feedback_config is not None:
                if diversity_group_size != active_feedback_config.group_size:
                    raise ValueError(
                        "Feedback group_size must equal "
                        "num_images_per_prompt."
                    )
                if diversity_pool_size != active_feedback_config.pool_size:
                    raise ValueError(
                        "diversity_pool_size must match the frozen feedback "
                        "pool_size."
                    )
                if feedback_reference.reference_margin_alpha != 0.0:
                    raise ValueError(
                        "Feedback control requires the alpha=0 base "
                        "reference; target margins belong in its controller "
                        "configuration."
                    )
                if abs(
                    active_feedback_config.initial_rho
                    - feedback_reference.rho_a_star
                ) > 1e-12:
                    raise ValueError(
                        "Feedback initial_rho must equal the frozen A* rho."
                    )

        callback_tensor_inputs = callback_on_step_end_tensor_inputs or [
            "latents"
        ]
        if not height or not width:
            sample_size = self.unet.config.sample_size
            if self._is_unet_config_sample_size_int:
                height = sample_size
                width = sample_size
            else:
                height = sample_size[0]
                width = sample_size[1]
            height *= self.vae_scale_factor
            width *= self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            None,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            None,
            None,
            callback_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            if prompt_embeds is None:
                raise ValueError("Provide either prompt or prompt_embeds.")
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = (
            self.cross_attention_kwargs.get("scale")
            if self.cross_attention_kwargs is not None
            else None
        )
        clean_positive, clean_negative = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        effective_batch = batch_size * num_images_per_prompt
        feedback_controller = None
        if feedback_mvp_config is not None:
            feedback_controller = GroupProportionalController(
                feedback_mvp_config,
                group_count=batch_size,
                device=device,
            )
        positive_stream = None
        negative_stream = None
        expanded_content_mask = None
        resolved_negative_seed = negative_condition_seed
        if original_cads_config is not None:
            if clean_negative is None:
                raise RuntimeError(
                    "Original CADS requires an encoded null/negative branch."
                )
            positive_stream = ConditionNoiseStream(
                seed=condition_seed,
                batch_size=effective_batch,
                device=device,
            )
            if resolved_negative_seed is None:
                resolved_negative_seed = negative_seed_from_positive(
                    condition_seed
                )
            negative_stream = ConditionNoiseStream(
                seed=resolved_negative_seed,
                batch_size=effective_batch,
                device=device,
            )
            model_prompt_embeds = clean_positive
        elif clean_unconditional_cads_config is not None:
            if clean_negative is None:
                raise RuntimeError(
                    "Clean-unconditional CADS requires an encoded "
                    "null/negative branch."
                )
            prompt_mask = (
                build_content_token_mask(self.tokenizer, prompt)
                if content_token_mask is None
                else content_token_mask
            )
            expanded_content_mask = expand_content_token_mask(
                prompt_mask,
                prompt_batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                sequence_length=clean_positive.shape[1],
                device=device,
            )
            positive_stream = ConditionNoiseStream(
                seed=condition_seed,
                batch_size=effective_batch,
                device=device,
            )
            model_prompt_embeds = clean_positive
        elif self.do_classifier_free_guidance:
            model_prompt_embeds = torch.cat(
                [clean_negative, clean_positive]
            )
        else:
            model_prompt_embeds = clean_positive

        timesteps_tensor, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
        if (
            feedback_reference is not None
            and len(timesteps_tensor) != feedback_reference.step_count
        ):
            raise ValueError(
                "Actual scheduler step count does not match the frozen "
                "feedback reference."
            )
        latents = self.prepare_latents(
            effective_batch,
            self.unet.config.in_channels,
            height,
            width,
            clean_positive.dtype,
            device,
            generator,
            latents,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(
                self.guidance_scale - 1
            ).repeat(effective_batch)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self.unet.config.time_cond_proj_dim,
            ).to(device=device, dtype=latents.dtype)

        num_warmup_steps = (
            len(timesteps_tensor)
            - num_inference_steps * self.scheduler.order
        )
        self._num_timesteps = len(timesteps_tensor)
        timestep_history: List[float] = []
        normalized_timestep_history: List[float] = []
        gamma_history: List[float] = []
        pollution_history: List[Any] = []
        progress_history: List[float] = []
        pollution_cap_history: List[float] = []
        positive_condition_max_delta = 0.0
        negative_condition_max_delta = 0.0
        non_content_positive_max_delta = 0.0
        scheduler_step_calls = 0
        main_unet_calls = 0
        unet_calls = 0
        grad_enabled_during_loop = False
        diversity_signal_history: List[Dict[str, Any]] = []
        feedback_control_history: List[Dict[str, Any]] = []

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, timestep in enumerate(timesteps_tensor):
                if self.interrupt:
                    continue

                grad_enabled_during_loop |= torch.is_grad_enabled()
                rho_used_by_prompt = None
                pollution_by_prompt = None
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input,
                        timestep,
                    )

                if original_cads_config is not None:
                    timestep_value = float(timestep.item())
                    num_train_timesteps = (
                        self.scheduler.config.num_train_timesteps
                    )
                    normalized_timestep = timestep_value / float(
                        num_train_timesteps - 1
                    )
                    pollution = paper_pollution(
                        timestep_value,
                        num_train_timesteps=num_train_timesteps,
                        tau1=original_cads_config.tau1,
                        tau2=original_cads_config.tau2,
                    )
                    timestep_history.append(timestep_value)
                    normalized_timestep_history.append(
                        normalized_timestep
                    )
                    pollution_history.append(pollution)
                    gamma_history.append(1.0 - pollution)

                    if pollution == 0.0:
                        step_positive = clean_positive
                        step_negative = clean_negative
                    else:
                        step_positive, step_negative = (
                            corrupt_cfg_conditions(
                                clean_positive,
                                clean_negative,
                                pollution=pollution,
                                positive_noise=positive_stream.sample(
                                    clean_positive.shape
                                ),
                                negative_noise=negative_stream.sample(
                                    clean_negative.shape
                                ),
                                config=original_cads_config,
                            )
                        )
                        positive_condition_max_delta = max(
                            positive_condition_max_delta,
                            float(
                                (
                                    step_positive.float()
                                    - clean_positive.float()
                                )
                                .abs()
                                .max()
                                .item()
                            ),
                        )
                        negative_condition_max_delta = max(
                            negative_condition_max_delta,
                            float(
                                (
                                    step_negative.float()
                                    - clean_negative.float()
                                )
                                .abs()
                                .max()
                                .item()
                            ),
                        )
                    model_prompt_embeds = torch.cat(
                        [step_negative, step_positive]
                    )
                elif clean_unconditional_cads_config is not None:
                    progress = i / max(len(timesteps_tensor) - 1, 1)
                    pollution_cap = progress_pollution_cap(
                        progress,
                        hold_until=(
                            clean_unconditional_cads_config
                            .hold_until_progress
                        ),
                        hard_off=(
                            clean_unconditional_cads_config
                            .hard_off_progress
                        ),
                    )
                    timestep_history.append(float(timestep.item()))
                    progress_history.append(progress)
                    pollution_cap_history.append(pollution_cap)

                    if feedback_controller is not None:
                        feedback_reference.validate_step(
                            i,
                            scheduler_timestep=float(timestep.item()),
                            progress=progress,
                            pollution_cap=pollution_cap,
                        )
                        rho_used_by_prompt = feedback_controller.rho
                        pollution_by_prompt = (
                            rho_used_by_prompt * float(pollution_cap)
                        )
                        pollution = pollution_by_prompt.repeat_interleave(
                            num_images_per_prompt
                        )
                        pollution_history.append(
                            pollution_by_prompt.detach().cpu().tolist()
                        )
                    else:
                        pollution = (
                            clean_unconditional_cads_config.fixed_rho_a
                            * pollution_cap
                        )
                        pollution_history.append(pollution)

                    should_corrupt = (
                        pollution_cap > 0.0
                        if feedback_controller is not None
                        else pollution != 0.0
                    )
                    if not should_corrupt:
                        step_positive = clean_positive
                    else:
                        # In feedback mode every active schedule step consumes
                        # exactly one bank draw, even if a group's rho is zero.
                        # This keeps later condition-noise draws paired with A*.
                        step_positive = corrupt_condition(
                            clean_positive,
                            pollution=pollution,
                            noise=positive_stream.sample(
                                clean_positive.shape
                            ),
                            noise_scale=(
                                clean_unconditional_cads_config.noise_scale
                            ),
                            rescale_mix=(
                                clean_unconditional_cads_config.rescale_mix
                            ),
                            rescale=(
                                clean_unconditional_cads_config.rescale
                            ),
                            mask=expanded_content_mask,
                            eps=clean_unconditional_cads_config.eps,
                        )
                        positive_condition_max_delta = max(
                            positive_condition_max_delta,
                            float(
                                (
                                    step_positive.float()
                                    - clean_positive.float()
                                )
                                .abs()
                                .max()
                                .item()
                            ),
                        )

                    # This assignment is intentional: the unconditional
                    # branch is never copied, perturbed, or rescaled in A.
                    step_negative = clean_negative
                    negative_condition_max_delta = max(
                        negative_condition_max_delta,
                        float(
                            (
                                step_negative.float()
                                - clean_negative.float()
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                    )
                    non_content_delta = (
                        step_positive.float() - clean_positive.float()
                    ).abs().masked_select(
                        ~expanded_content_mask.unsqueeze(-1).expand_as(
                            step_positive
                        )
                    )
                    if non_content_delta.numel() > 0:
                        non_content_positive_max_delta = max(
                            non_content_positive_max_delta,
                            float(non_content_delta.max().item()),
                        )
                    model_prompt_embeds = torch.cat(
                        [step_negative, step_positive]
                    )

                noise_pred = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=model_prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]
                main_unet_calls += 1
                unet_calls += 1

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noisy_guidance = noise_pred_text - noise_pred_uncond
                    if diversity_enabled:
                        guidance_diversity = (
                            grouped_pairwise_cosine_diversity(
                                noisy_guidance,
                                group_size=diversity_group_size,
                                pool_size=diversity_pool_size,
                            )
                        )
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noisy_guidance
                    )
                if (
                    self.do_classifier_free_guidance
                    and self.guidance_rescale > 0.0
                ):
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                if diversity_enabled:
                    scheduler_output = self.scheduler.step(
                        noise_pred,
                        timestep,
                        latents,
                        **extra_step_kwargs,
                        return_dict=True,
                    )
                    latents = scheduler_output.prev_sample
                    predicted_clean = getattr(
                        scheduler_output, "pred_original_sample", None
                    )
                    if predicted_clean is None:
                        raise RuntimeError(
                            "The scheduler must expose pred_original_sample "
                            "for D^z collection."
                        )
                    predicted_clean_diversity = (
                        grouped_pairwise_cosine_diversity(
                            predicted_clean,
                            group_size=diversity_group_size,
                            pool_size=diversity_pool_size,
                        )
                    )
                    signal_record = {
                        "step_index": i,
                        "scheduler_timestep": float(timestep.item()),
                        "progress": progress_history[-1],
                        "pollution_cap": pollution_cap_history[-1],
                        "pollution": pollution_history[-1],
                        "guidance_diversity_by_prompt": (
                            guidance_diversity.detach().cpu().tolist()
                        ),
                        "predicted_clean_latent_diversity_by_prompt": (
                            predicted_clean_diversity.detach().cpu().tolist()
                        ),
                    }
                    diversity_signal_history.append(signal_record)

                    if feedback_controller is not None:
                        control_update = feedback_controller.update(
                            guidance_diversity=guidance_diversity,
                            predicted_clean_latent_diversity=(
                                predicted_clean_diversity
                            ),
                            guidance_reference=(
                                feedback_reference.guidance_diversity[i]
                            ),
                            predicted_clean_latent_reference=(
                                feedback_reference
                                .predicted_clean_latent_diversity[i]
                            ),
                            dz_control_eligible=(
                                feedback_reference.dz_control_eligible[i]
                            ),
                            pollution_cap=pollution_cap,
                        )
                        if not torch.equal(
                            control_update.rho_used, rho_used_by_prompt
                        ):
                            raise RuntimeError(
                                "Controller rho changed before the current "
                                "step observation was applied."
                            )
                        feedback_control_history.append(
                            {
                                "step_index": i,
                                "scheduler_timestep": float(timestep.item()),
                                "progress": progress,
                                "pollution_cap": pollution_cap,
                                "rho_used_by_prompt": (
                                    control_update.rho_used.detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "pollution_by_prompt": (
                                    pollution_by_prompt.detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "guidance_diversity_by_prompt": (
                                    guidance_diversity.detach().cpu().tolist()
                                ),
                                "predicted_clean_latent_diversity_by_prompt": (
                                    predicted_clean_diversity.detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "guidance_target": (
                                    control_update.guidance_target
                                ),
                                "predicted_clean_latent_target": (
                                    control_update
                                    .predicted_clean_latent_target
                                ),
                                "guidance_deficit_by_prompt": (
                                    control_update.guidance_deficit.detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "predicted_clean_latent_deficit_by_prompt": (
                                    control_update
                                    .predicted_clean_latent_deficit.detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "diversity_deficit_by_prompt": (
                                    control_update.diversity_deficit.detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "delta_rho_by_prompt": (
                                    control_update.delta_rho.detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "rho_next_by_prompt": (
                                    control_update.rho_next.detach()
                                    .cpu()
                                    .tolist()
                                ),
                                "dz_used": control_update.dz_used,
                                "control_updated": (
                                    control_update.control_updated
                                ),
                            }
                        )
                else:
                    latents = self.scheduler.step(
                        noise_pred,
                        timestep,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )[0]
                scheduler_step_calls += 1

                should_update = i == len(timesteps_tensor) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                )
                if should_update:
                    progress_bar.update()

        if output_type != "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image,
                device,
                clean_positive.dtype,
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [
                not has_nsfw for has_nsfw in has_nsfw_concept
            ]
        image = self.image_processor.postprocess(
            image,
            output_type=output_type,
            do_denormalize=do_denormalize,
        )

        if feedback_controller is not None:
            run_mode = "feedback_mvp"
        elif original_cads_config is not None:
            run_mode = "original_cads"
        elif clean_unconditional_cads_config is not None:
            run_mode = "clean_unconditional_cads"
        else:
            run_mode = "vanilla"

        self._last_run_stats = {
            "mode": run_mode,
            "cads_enabled": cads_configured,
            "actual_timesteps": len(timesteps_tensor),
            "scheduler_step_calls": scheduler_step_calls,
            "main_unet_calls": main_unet_calls,
            "unet_calls": unet_calls,
            "cfg_equivalent_unet_cost_ratio": (
                1.0
                if self.do_classifier_free_guidance
                and main_unet_calls > 0
                else None
            ),
            "grad_enabled_at_entry": grad_enabled_at_entry,
            "grad_enabled_during_loop": grad_enabled_during_loop,
            "timestep_history": timestep_history,
            "normalized_timestep_history": normalized_timestep_history,
            "gamma_history": gamma_history,
            "pollution_history": pollution_history,
            "progress_history": progress_history,
            "pollution_cap_history": pollution_cap_history,
            "diversity_signal_history": diversity_signal_history,
            "feedback_control_history": feedback_control_history,
            "diversity_group_size": (
                diversity_group_size if diversity_enabled else None
            ),
            "diversity_pool_size": (
                diversity_pool_size if diversity_enabled else None
            ),
            "fixed_rho_a": (
                clean_unconditional_cads_config.fixed_rho_a
                if clean_unconditional_cads_config is not None
                and feedback_controller is None
                else None
            ),
            "feedback_initial_rho": (
                active_feedback_config.initial_rho
                if active_feedback_config is not None
                else None
            ),
            "feedback_k_diversity": (
                active_feedback_config.k_diversity
                if active_feedback_config is not None
                else None
            ),
            "feedback_reference_margin_alpha": (
                active_feedback_config.reference_margin_alpha
                if active_feedback_config is not None
                else None
            ),
            "feedback_reference_path": (
                feedback_reference.path
                if feedback_reference is not None
                else None
            ),
            "feedback_reference_sha256": (
                feedback_reference.sha256
                if feedback_reference is not None
                else None
            ),
            "content_token_count_per_sample": (
                expanded_content_mask.sum(dim=1).tolist()
                if expanded_content_mask is not None
                else None
            ),
            "condition_seed": (
                int(condition_seed)
                if cads_configured
                else None
            ),
            "positive_condition_noise_draw_calls": (
                positive_stream.draw_calls
                if positive_stream is not None
                else 0
            ),
            "negative_condition_noise_draw_calls": (
                negative_stream.draw_calls
                if negative_stream is not None
                else 0
            ),
            "negative_condition_seed": (
                int(resolved_negative_seed)
                if original_cads_config is not None
                else None
            ),
            "positive_condition_max_delta": (
                positive_condition_max_delta
            ),
            "negative_condition_max_delta": (
                negative_condition_max_delta
            ),
            "non_content_positive_max_delta": (
                non_content_positive_max_delta
            ),
            "paper_source_protocol": (
                {
                    "model": original_cads_config.source_model,
                    "sampler": original_cads_config.source_sampler,
                    "num_steps": original_cads_config.source_num_steps,
                    "cfg_scale": original_cads_config.source_cfg_scale,
                }
                if original_cads_config is not None
                else None
            ),
            "project_target_protocol": (
                {
                    "model": original_cads_config.target_model,
                    "sampler": original_cads_config.target_sampler,
                    "num_steps": original_cads_config.target_num_steps,
                    "cfg_scale": original_cads_config.target_cfg_scale,
                }
                if original_cads_config is not None
                else None
            ),
        }

        self.maybe_free_model_hooks()
        if not return_dict:
            return image, has_nsfw_concept
        return StableDiffusionPipelineOutput(
            images=image,
            nsfw_content_detected=has_nsfw_concept,
        )

