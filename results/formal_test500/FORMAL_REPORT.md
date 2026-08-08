# COCO-Test-500 Formal Report

## Main table

| Method | DINO diversity | CLIP image diversity | CLIPScore | HPSv2 | s/prompt | UNet calls |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 0.200100 | 0.076901 | 0.785446 | 0.260481 | 7.1916 | 50 |
| Original CADS | 0.235120 | 0.096201 | 0.778002 | 0.246984 | 7.2249 | 50 |
| A* | 0.203794 | 0.078464 | 0.782338 | 0.258324 | 7.2304 | 50 |
| B* | 0.204895 | 0.078677 | 0.781730 | 0.257690 | 7.2686 | 50 |


## Preregistered primary comparison: B* - A*

| Metric | Mean difference | 95% paired-bootstrap CI | 1% NI margin | Noninferior | Strong improvement |
|---|---:|---:|---:|:---:|:---:|
| DINO diversity | +0.001101 | [0.000289, 0.001926] | 0.002038 | yes | yes |
| CLIPScore | -0.000608 | [-0.001272, -0.000012] | 0.007823 | yes | no |
| HPSv2 | -0.000634 | [-0.000856, -0.000415] | 0.002583 | yes | no |

Assessment success: **true**  
Closed-loop diversity success: **true**  
Strong research result: **true**

B* increases DINO diversity relative to A* with a positive 95% CI. CLIPScore and HPSv2 are slightly lower, not higher, but both pass the preregistered 1% noninferiority rule. The defensible claim is therefore a statistically significant diversity gain over the direct open-loop A* comparator while preserving preregistered quality noninferiority.

The secondary B* versus Vanilla comparison does not pass HPSv2 noninferiority. It must not be described as quality-noninferior to Vanilla on every core metric.

## Low-score candidates for manual review

The `low_score_case_manifest.jsonl` contains exactly five lowest prompt groups per method for CLIPScore and five for HPSv2. The two criterion lists are separate and may overlap. Contact sheets are under `low_score_contact_sheets/`.

These are automatically selected low-metric candidates, not human-confirmed failures. Complete `manual_failure_annotations.csv` before making claims about missing objects, attribute errors, distortion, or other visual failure types.
