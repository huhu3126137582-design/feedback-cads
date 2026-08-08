# Publication result package: figure contract and QA notes

- Core conclusion: Feedback-CADS significantly improves DINO diversity over the matched fixed-CADS baseline while retaining preregistered CLIPScore and HPSv2 noninferiority.
- Archetype: quantitative grid with DINO diversity as the hero panel and quality metrics as supporting non-inferiority evidence.
- Backend: Python/matplotlib only.
- Final size: 180 mm × 63 mm before tight-crop export.
- Statistical unit: prompt; n = 500; K = 8; 10,000 paired-bootstrap replicates; 95% CI.
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
