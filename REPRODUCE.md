# Feedback-CADS reproducibility entry points

Run every command from the project root. `scripts/feedback_cads_cli.py` is the public,
stage-oriented entry point; the narrower Python files are implementation modules
used by its subcommands. Frozen legacy YAML identities are retained inside the
consolidated profile bundles.

Create the pinned environment first with `conda env create -f environment.yml`
or `python -m pip install -r requirements.txt`. Model checkpoints and full
generated-image archives are deliberately excluded from Git; see `README.md`
and `THIRD_PARTY.md` for exact revisions and cache locations.

The canonical shared metric configuration is
`configs/evaluation_metrics_sd15.yaml`. The legacy
`configs/a_strength_curve_metrics.yaml` path remains as a symbolic-link alias
because the already-frozen formal-test protocol records that exact path.

The frozen COCO split files are included in Git. To create a new provenance
record from raw COCO annotations, use:

```bash
python scripts/feedback_cads_cli.py data build-coco-splits
```

Do not replace the repository-provided `coco_caption_splits.json` when
reproducing the published formal protocol hash: its historical provenance
strings are intentionally part of that frozen hash.

## 1. Stage A: coarse-to-fine A* selection

```bash
python scripts/feedback_cads_cli.py stage-a run --profile coarse
python scripts/feedback_cads_cli.py stage-a evaluate --profile coarse
python scripts/feedback_cads_cli.py stage-a audit --profile coarse

python scripts/feedback_cads_cli.py stage-a run --profile fine
python scripts/feedback_cads_cli.py stage-a evaluate --profile fine
python scripts/feedback_cads_cli.py stage-a audit --profile fine

python scripts/feedback_cads_cli.py stage-a select
python scripts/feedback_cads_cli.py stage-a reference
python scripts/feedback_cads_cli.py stage-a audit-reference
```

The consolidated profiles retain the same canonical configuration SHA-256 as
the original runs. Replaying `select_parameters.py a-star` produces the frozen
`A_STAR_SELECTION.json` byte-for-byte.

## 2. Stage B: reference-margin search

Run generation and evaluation for each profile in order:

```bash
python scripts/feedback_cads_cli.py stage-b run --profile alpha_0p00
python scripts/feedback_cads_cli.py stage-b evaluate --profile alpha_0p00

python scripts/feedback_cads_cli.py stage-b run --profile alpha_0p05
python scripts/feedback_cads_cli.py stage-b evaluate --profile alpha_0p05
python scripts/feedback_cads_cli.py stage-b run --profile alpha_0p10
python scripts/feedback_cads_cli.py stage-b evaluate --profile alpha_0p10
python scripts/feedback_cads_cli.py stage-b select

python scripts/feedback_cads_cli.py stage-b run --profile alpha_0p15
python scripts/feedback_cads_cli.py stage-b evaluate --profile alpha_0p15
python scripts/feedback_cads_cli.py stage-b run --profile alpha_0p20
python scripts/feedback_cads_cli.py stage-b evaluate --profile alpha_0p20
python scripts/feedback_cads_cli.py stage-b select --extended
```

The default output directory is derived from the selected profile. The two B
selection JSON files are reproduced byte-for-byte, including the legacy config
hash identities embedded in the consolidated bundle.

## 3. Verify the frozen formal run without regenerating images

```bash
python scripts/feedback_cads_cli.py verify
```

This command requires the local model cache and full `outputs/` archive. A
GitHub-sized checkout can instead verify unit logic and the delivered numerical
results with:

```bash
python -m pytest -q -m "not artifact and not gpu"
sha256sum -c results/CHECKSUMS.sha256
```

## 4. Rebuild publication artifacts from frozen results

```bash
python scripts/feedback_cads_cli.py publication all
```

`publication all` now rebuilds the publication figures and refreshes the
Git-friendly `results/` package. To refresh only that package, run
`python scripts/feedback_cads_cli.py publication package`.

Starting the full formal generation or formal metric computation remains
confirmation-gated. First run the corresponding command with `--validate-only`
and use the exact SHA-256 printed by that command; do not tune any parameter on
COCO-Test-500.
