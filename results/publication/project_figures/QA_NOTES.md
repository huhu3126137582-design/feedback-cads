# Figure contracts and QA notes

| Figure | Archetype | Core conclusion | Evidence source | Reviewer-risk control |
|:--|:--|:--|:--|:--|
| Fig. 1 | Schematic-led | Feedback-CADS closes the condition-annealing loop without training or extra UNet calls | Frozen method specification | No invented quantitative values; one-step delay and hard-off are explicit |
| Fig. 2 | Image plate | The paired qualitative output can be inspected under identical prompt and latent seeds | Frozen formal images | Machine-selected median case; all K = 8 candidates; no crop or colour adjustment |
| Fig. 3 | Quantitative grid | The controller is prompt-adaptive and follows frozen references during the active window | All 500 formal diagnostics | Median and IQR use prompts as the replicate unit; clean phase is shaded |
| Fig. 4 | Quantitative grid | Frozen A* and B* settings follow the disclosed Dev-50 selection rules | Frozen Dev-50 selection records | Explicitly labelled development evidence; formal Test-500 conclusions are unchanged |

- Backend: Python/matplotlib only.
- Export contract: editable SVG and PDF, 600 dpi LZW TIFF, and 300 dpi PNG preview.
- Final width: 180 mm before tight cropping; every plotted font is at least 5 pt.
- Fig. 1 mathtext glyph audit: exported PDF minimum is 5.04 pt (required minimum: 5 pt); no text run falls below the floor.
- Palette: Feedback-CADS is consistently blue; green marks selected/improved settings; red marks thresholds or failed constraints.
- Source data: clean CSV or JSON manifests accompany Figs. 2–4.
- Image integrity: Fig. 2 uses the complete 512×512 images with no crop, rescaling beyond display interpolation, contrast, gamma or colour manipulation.
- Formal acceptance: unchanged; these are post-evaluation presentation artifacts.
