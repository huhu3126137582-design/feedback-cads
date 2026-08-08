# COCO-Dev-50 Stage-A selection report

## Frozen result

The preregistered coarse-to-fine search selects **rho_A*=0.55**.

- Data: frozen COCO-Dev-50 only; COCO-Test-300 was not read or generated.
- Coarse grid: 0, 0.2, 0.4, 0.6, 0.8, 1.0.
- Quality-safe coarse points: 0.2 and 0.4.
- Frozen fine-grid center: 0.4.
- New fine points: 0.25, 0.30, 0.35, 0.45, 0.50, 0.55.
- Reused coarse points in the fine neighborhood: 0.2, 0.4, 0.6.

| Metric | Vanilla | rho_A*=0.55 | Paired mean difference [95% CI] |
| --- | ---: | ---: | ---: |
| DINOv2 diversity | 0.215869 | 0.219388 | +0.003519 [-0.001371, 0.008778] |
| CLIPScore | 0.767918 | 0.765975 | -0.001943 [-0.005648, 0.001633] |
| HPSv2 | 0.256304 | 0.255322 | -0.000982 [-0.002237, 0.000303] |

The CLIPScore and HPSv2 lower confidence bounds pass their preregistered 1%
non-inferiority margins (0.007679 and 0.002563). DINO diversity has a positive
point difference, so rho=0.55 satisfies the development-set selection rule.
Its DINO confidence interval still crosses zero; this is not evidence of a
statistically significant improvement. Freeze this value before any
COCO-Test-300 run.

The machine-readable decision, every candidate's paired bootstrap interval,
and the test-data policy are stored in `A_STAR_SELECTION.json`.
