# Third-party models and data

This repository does not redistribute model weights or the full generated-image
archive. Users must obtain each resource under its own upstream terms.

## Models and metric checkpoints

- Stable Diffusion v1.5: `stable-diffusion-v1-5/stable-diffusion-v1-5`, revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
- DINOv2 base: `facebook/dinov2-base`, revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.
- CLIP ViT-B/32 safetensors conversion: `sentence-transformers/clip-ViT-B-32`, revision `327ab6726d33c0e22f920c83f2ff9e4bd38ca37f`.
- HPSv2 checkpoint: `xswu/HPSv2`, revision `697403c78157020a1ae59d23f111aa58ced35b0a`.
- LPIPS calibration weights are supplied by the pinned `lpips` package.

Optional historical development assets such as BLIP-VQA are not needed to
verify the final frozen result package.

## Dataset

The frozen development/test prompt manifests are derived from MS-COCO 2017
validation captions. They contain one caption per selected image and do not
include the corresponding COCO images. Obtain raw COCO annotations from the
official COCO distribution if rebuilding the split.

Before redistributing any third-party checkpoint or raw dataset file, review
the corresponding upstream license and terms. The repository owner's project
license does not override third-party terms.
