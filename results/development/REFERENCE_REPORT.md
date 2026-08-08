# Frozen A* intermediate-diversity reference

## Result

The base reference curves for Feedback-CADS B are frozen from A* with
`rho_A=0.55` and `reference_margin_alpha=0` on COCO-Dev-50.

- Prompts: 50.
- Candidates per prompt: 8.
- DDIM sampling steps: 50.
- Prompt-step records: 2,500.
- Dg reference: prompt median of group guidance diversity at each step.
- Dz reference: prompt median of group predicted-clean-latent diversity at
  each step.
- Dz is recorded at every step and becomes control-eligible at progress >=
  0.20 (step index 10 for this schedule).
- COCO-Test-300 was not read or generated.

Both signals use the mean unique-pair `(1-cosine)/2` after adaptive 8x8
pooling and FP32 calculation. Dg is computed from `eps_text-eps_uncond` from
the ordinary CFG forward pass. Dz is computed from the DDIM scheduler's
`pred_original_sample`. Signal collection adds no UNet forward pass.

## Validation

- All 400 regenerated A* images exactly match the frozen rho=0.55 images
  pixel by pixel.
- Every prompt used exactly 50 UNet calls and 50 scheduler calls.
- The final 20 sampling steps have exactly zero condition pollution.
- All 50 condition seeds and all 50 latent-seed groups are distinct.
- An independent recomputation exactly reproduced all 50 Dg and Dz medians.
- Full test suite: 37 passed.

The Dg reference ranges from 0.246647 to 0.496153. The Dz reference ranges
from 0.242371 to 0.409098. These ranges describe the frozen schedule and are
not additional tuning targets.

## Artifacts

- `reference_curves.json`: SHA-256
  `c2ae2024f06c699eb9e51e063ab9aea2f118c03663d6918965c1e639f9667382`
- `per_step.jsonl`: SHA-256
  `2be0b8607ae640239e561f1d4819700c7618b719df0480be02674b4fae434d45`
- `validation_report.json`: completeness, image-regression and call-count
  audit.
- `run_config.json`: frozen data, model, sampling, CADS, seed and reference
  configuration.
