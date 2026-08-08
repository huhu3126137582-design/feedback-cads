# Result data package

This directory is the Git-friendly evidence package for Feedback-CADS. It does
not contain model weights, formal PNG generations, per-image feature arrays, or
other large caches.

## Formal Test-500

`formal_test500/metrics/` contains:

- method-level summaries (`summary.csv`, `summary.json`);
- all 2,000 method-prompt records (`per_prompt.jsonl`);
- the four paired comparisons and 10,000-replicate confidence intervals
  (`paired_bootstrap.json`);
- independent evaluation and completion audits.

The formal generation completion and 16,000-image integrity-audit records are
included, but the corresponding 7 GB image archive is not stored in Git.

## Development selections

`development/` contains the frozen A* strength selection, Stage-B reference
margin selections, reference curves and per-step signal records, and the
COCO-Dev-50 A/B ablation measurements used to make those selections.

## Figures

`publication/` contains the main table, primary paired-CI source data and
figure, all four project figures, legends, QA notes, and source data for the
quantitative controller/ablation figures. Reproducible TIFF exports remain in
the local `outputs/` archive and are omitted here to keep Git history compact.

Fig. 2 is an image plate derived from the full formal image archive. Its PDF
and PNG are included, but the 32 source PNGs and large embedded-image SVG are
not included in the Git package.

## Replotting and slides

The repository contains presentation-ready PDF and PNG exports for all four
project figures. Fig. 1 also has an SVG suitable for vector editing. For new
quantitative visualizations, use:

- `publication/main_table.csv` for the four-method result table;
- `publication/primary_b_vs_a_ci_source_data.csv` for the primary confidence-
  interval plot;
- `publication/project_figures/fig3_controller_dynamics_source_data.csv` for
  controller trajectories and reference curves;
- `publication/project_figures/fig4_ablation_source_data.csv` for the Stage-A
  and Stage-B parameter-selection panels;
- `formal_test500/metrics/per_prompt.jsonl` and `paired_bootstrap.json` for
  custom prompt-level distributions and paired-comparison plots;
- `development/` for A*/B* selections, reference signals and development-set
  ablations.

Fig. 2 can be placed directly into slides from the included PDF or PNG. To
change its image selection or layout, first regenerate or restore the omitted
formal PNG archive as described in the root `REPRODUCE.md`.

Run `sha256sum -c results/CHECKSUMS.sha256` from the repository root to verify
the package. Rebuild it from a full local `outputs/` archive with:

```bash
python scripts/feedback_cads_cli.py publication package
```
