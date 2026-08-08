| Method | DINO diversity ↑ | CLIP diversity ↑ | CLIPScore ↑ | HPSv2 ↑ | Time (s prompt⁻¹) ↓ | UNet calls prompt⁻¹ ↓ |
|:--|--:|--:|--:|--:|--:|--:|
| Vanilla SD v1.5 | 0.200100 | 0.076901 | **0.785446** | **0.260481** | **7.1916** | 50 |
| Original CADS | **0.235120** | **0.096201** | 0.778002 | 0.246984 | 7.2249 | 50 |
| A* (fixed CADS) | 0.203794 | 0.078464 | 0.782338 | 0.258324 | 7.2304 | 50 |
| Feedback-CADS (B*) | 0.204895† | 0.078677 | 0.781730‡ | 0.257690‡ | 7.2686 | 50 |

Values are prompt-group means on COCO-Test-500 (n = 500 prompts; K = 8 images per prompt). Bold denotes the best method mean in each column; all methods use 50 UNet calls. † B* exceeds A* in DINO diversity with a positive 95% paired-bootstrap CI. ‡ B* is non-inferior to A* under the preregistered 1% margin. Runtime is hardware-dependent.
