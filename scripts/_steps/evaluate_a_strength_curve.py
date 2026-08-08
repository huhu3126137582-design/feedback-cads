#!/usr/bin/env python3
"""Evaluate the frozen stage-A curve at the prompt-group level."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads.experiments import (  # noqa: E402
    STAGE_A_RHO_GRID,
    mean_pairwise_cosine_distance,
    paired_bootstrap_mean_difference,
    read_jsonl,
    resolve_recorded_path,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/a_strength_curve_dev20",
    )
    parser.add_argument(
        "--metric-config",
        type=Path,
        default=PROJECT_ROOT / "configs/evaluation_metrics_sd15.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-hps", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def validate_manifest(
    records: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> None:
    sampling = run_config["sampling"]
    expected = (
        int(run_config["dataset"]["expected_prompts"])
        * int(sampling["candidates_per_prompt"])
        * len(sampling["rho_grid"])
    )
    if len(records) != expected:
        raise ValueError(f"Expected {expected} images, found {len(records)}.")
    expected_rhos = tuple(float(value) for value in sampling["rho_grid"])
    if tuple(sorted({float(row["rho"]) for row in records})) != expected_rhos:
        raise ValueError("Manifest does not contain the frozen rho grid.")

    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, int], dict[float, int]] = defaultdict(dict)
    for row in records:
        path = resolve_recorded_path(
            row["image_path"], project_root=PROJECT_ROOT
        )
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        groups[(float(row["rho"]), row["prompt_id"])].append(row)
        paired[(row["prompt_id"], int(row["candidate_id"]))][
            float(row["rho"])
        ] = int(row["latent_seed"])

    candidate_count = int(sampling["candidates_per_prompt"])
    for key, group in groups.items():
        if len(group) != candidate_count:
            raise ValueError(f"Group {key} has {len(group)} candidates.")
    for key, rho_to_seed in paired.items():
        if len(rho_to_seed) != len(expected_rhos):
            raise ValueError(f"Pair {key} is missing a rho setting.")
        if len(set(rho_to_seed.values())) != 1:
            raise ValueError(f"Pair {key} does not share its latent seed.")


def image_batches(
    records: list[dict[str, Any]],
    batch_size: int,
):
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = []
        for row in batch:
            image_path = resolve_recorded_path(
                row["image_path"], project_root=PROJECT_ROOT
            )
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        yield start, batch, images


@torch.no_grad()
def extract_dino_features(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    batch_size: int,
    cache_dir: Path,
    revision: str,
) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        revision=revision,
    )
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        revision=revision,
    ).to("cuda").eval()
    features = []
    for _, _, images in image_batches(records, batch_size):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        output = model(**inputs).last_hidden_state[:, 0]
        output = torch.nn.functional.normalize(output.float(), dim=-1)
        features.append(output.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(features, axis=0)


@torch.no_grad()
def extract_clip_metrics(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    batch_size: int,
    cache_dir: Path,
    clipscore_weight: float,
    revision: str,
    subfolder: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        revision=revision,
        subfolder=subfolder,
    )
    model = CLIPModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        revision=revision,
        subfolder=subfolder,
        use_safetensors=True,
    ).to("cuda").eval()
    unique_prompts = sorted({row["prompt"] for row in records})
    text_inputs = processor(
        text=unique_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {key: value.to("cuda") for key, value in text_inputs.items()}
    text_features = model.get_text_features(**text_inputs)
    text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    text_by_prompt = {
        prompt: feature.cpu().numpy()
        for prompt, feature in zip(unique_prompts, text_features)
    }

    image_features = []
    alignment = []
    clipscore = []
    for _, batch, images in image_batches(records, batch_size):
        image_inputs = processor(images=images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to("cuda")
        batch_features = model.get_image_features(pixel_values=pixel_values)
        batch_features = torch.nn.functional.normalize(
            batch_features.float(), dim=-1
        )
        batch_array = batch_features.cpu().numpy()
        batch_text = np.stack(
            [text_by_prompt[row["prompt"]] for row in batch]
        )
        cosine = np.sum(batch_array * batch_text, axis=1)
        image_features.append(batch_array)
        alignment.append(cosine)
        clipscore.append(float(clipscore_weight) * np.maximum(cosine, 0.0))
    del model
    torch.cuda.empty_cache()
    return (
        np.concatenate(image_features, axis=0),
        np.concatenate(alignment, axis=0),
        np.concatenate(clipscore, axis=0),
    )


@torch.no_grad()
def compute_lpips_by_group(
    groups: dict[tuple[float, str], list[int]],
    records: list[dict[str, Any]],
) -> dict[tuple[float, str], float]:
    import lpips
    from torchvision.transforms import functional as tvf

    os.environ.setdefault(
        "TORCH_HOME", str(PROJECT_ROOT / "models/torch")
    )
    model = lpips.LPIPS(
        net="alex",
        pretrained=False,
        verbose=False,
    )
    calibration_path = (
        Path(lpips.__file__).resolve().parent
        / "weights/v0.1/alex.pth"
    )
    calibration = torch.load(
        calibration_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(calibration, strict=False)
    model = model.to("cuda").eval()
    values = {}
    for key, indices in groups.items():
        tensors = []
        for index in indices:
            image_path = resolve_recorded_path(
                records[index]["image_path"], project_root=PROJECT_ROOT
            )
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize((256, 256), Image.BICUBIC)
                tensors.append(tvf.pil_to_tensor(image).float() / 127.5 - 1.0)
        pairs = list(itertools.combinations(range(len(tensors)), 2))
        first = torch.stack([tensors[left] for left, _ in pairs]).to("cuda")
        second = torch.stack([tensors[right] for _, right in pairs]).to("cuda")
        values[key] = float(model(first, second).flatten().mean().item())
    del model
    torch.cuda.empty_cache()
    return values


def compute_hps_scores(
    groups: dict[tuple[float, str], list[int]],
    records: list[dict[str, Any]],
    *,
    version: str,
    batch_size: int = 16,
) -> np.ndarray:
    del groups
    os.environ.setdefault(
        "HPS_ROOT", str(PROJECT_ROOT / "models/hpsv2")
    )
    import open_clip
    from huggingface_hub import hf_hub_download

    hps_version_map = {
        "v2.0": "HPS_v2_compressed.pt",
        "v2.1": "HPS_v2.1_compressed.pt",
    }
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained=None,
        device="cpu",
    )
    checkpoint_path = hf_hub_download(
        "xswu/HPSv2",
        hps_version_map[version],
        cache_dir=PROJECT_ROOT / "models/huggingface/hub",
        revision="697403c78157020a1ae59d23f111aa58ced35b0a",
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to("cuda").eval()
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    scores = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = []
        for row in batch:
            image_path = resolve_recorded_path(
                row["image_path"], project_root=PROJECT_ROOT
            )
            with Image.open(image_path) as image:
                images.append(preprocess(image.convert("RGB")))
        image_tensor = torch.stack(images).to("cuda", non_blocking=True)
        text_tensor = tokenizer([row["prompt"] for row in batch]).to(
            "cuda", non_blocking=True
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            image_features = model.encode_image(image_tensor, normalize=True)
            text_features = model.encode_text(text_tensor, normalize=True)
            batch_scores = torch.diagonal(
                image_features @ text_features.T
            )
        scores.append(batch_scores.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(scores, axis=0).astype(np.float64)


def make_plot(summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("dino_diversity", "DINOv2 diversity"),
        ("clipscore", "CLIPScore"),
        ("hpsv2", "HPSv2"),
    ]
    available = [(key, title) for key, title in metrics if key in summary]
    figure, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for axis, (key, title) in zip(axes, available):
        axis.plot(summary["rho"], summary[key], marker="o")
        axis.set_xlabel(r"$\rho_A$")
        axis.set_ylabel(title)
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    manifest_path = experiment_dir / "manifest.jsonl"
    run_config = json.loads(
        (experiment_dir / "run_config.json").read_text(encoding="utf-8")
    )
    metric_protocol = yaml.safe_load(
        args.metric_config.read_text(encoding="utf-8")
    )
    records = read_jsonl(manifest_path)
    validate_manifest(records, run_config)
    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "models/huggingface/hub"
    evaluation = metric_protocol

    cache_path = metrics_dir / "feature_cache.npz"
    manifest_sha = sha256(manifest_path)
    if cache_path.exists():
        cached = np.load(cache_path)
        if str(cached["manifest_sha"].item()) != manifest_sha:
            raise RuntimeError("Feature cache belongs to another manifest.")
        dino_features = cached["dino_features"]
        clip_features = cached["clip_features"]
        clip_alignment = cached["clip_alignment"]
        clipscores = cached["clipscores"]
    else:
        dino_features = extract_dino_features(
            records,
            model_name=evaluation["dino_model"],
            batch_size=args.batch_size,
            cache_dir=cache_dir,
            revision=evaluation["dino_revision"],
        )
        clip_features, clip_alignment, clipscores = extract_clip_metrics(
            records,
            model_name=evaluation["clip_model"],
            batch_size=args.batch_size,
            cache_dir=cache_dir,
            clipscore_weight=float(evaluation["clipscore_weight"]),
            revision=evaluation["clip_revision"],
            subfolder=evaluation["clip_subfolder"],
        )
        np.savez_compressed(
            cache_path,
            manifest_sha=np.array(manifest_sha),
            dino_features=dino_features,
            clip_features=clip_features,
            clip_alignment=clip_alignment,
            clipscores=clipscores,
        )

    groups: dict[tuple[float, str], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[(float(row["rho"]), row["prompt_id"])].append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: int(records[index]["candidate_id"]))

    lpips_values = (
        {} if args.skip_lpips else compute_lpips_by_group(groups, records)
    )
    hps_scores = (
        np.full(len(records), np.nan)
        if args.skip_hps
        else compute_hps_scores(
            groups,
            records,
            version=evaluation["hps_version"],
            batch_size=min(args.batch_size, 16),
        )
    )
    per_image = []
    for index, row in enumerate(records):
        per_image.append(
            {
                **row,
                "clip_alignment_cosine": float(clip_alignment[index]),
                "clipscore": float(clipscores[index]),
                "hpsv2": (
                    None if np.isnan(hps_scores[index]) else float(hps_scores[index])
                ),
            }
        )
    write_jsonl(metrics_dir / "per_image.jsonl", per_image)

    per_prompt = []
    for (rho, prompt_id), indices in sorted(groups.items()):
        first = records[indices[0]]
        per_prompt.append(
            {
                "rho": rho,
                "prompt_id": prompt_id,
                "category": first["category"],
                "prompt": first["prompt"],
                "candidate_count": len(indices),
                "dino_diversity": mean_pairwise_cosine_distance(
                    dino_features[indices]
                ),
                "clip_image_diversity": mean_pairwise_cosine_distance(
                    clip_features[indices]
                ),
                "lpips_diversity": lpips_values.get((rho, prompt_id)),
                "clip_alignment_cosine": float(clip_alignment[indices].mean()),
                "clipscore": float(clipscores[indices].mean()),
                "hpsv2": (
                    None
                    if np.isnan(hps_scores[indices]).all()
                    else float(np.nanmean(hps_scores[indices]))
                ),
                "prompt_run_seconds": float(first["prompt_run_seconds"]),
                "rho_mean": rho,
                "rho_variance": 0.0,
                "rho_boundary_rate": float(rho in (0.0, 1.0)),
                "hard_guard_trigger_rate": 0.0,
                "unet_calls": int(run_config["sampling"]["num_inference_steps"]),
            }
        )
    write_jsonl(metrics_dir / "per_prompt.jsonl", per_prompt)
    frame = pd.DataFrame(per_prompt)
    numeric_metrics = [
        "dino_diversity",
        "clip_image_diversity",
        "lpips_diversity",
        "clip_alignment_cosine",
        "clipscore",
        "hpsv2",
        "prompt_run_seconds",
        "rho_mean",
        "rho_variance",
        "rho_boundary_rate",
        "hard_guard_trigger_rate",
        "unet_calls",
    ]
    available_metrics = [key for key in numeric_metrics if not frame[key].isna().all()]
    summary = frame.groupby("rho", as_index=False)[available_metrics].mean()
    summary.to_csv(metrics_dir / "summary.csv", index=False)

    comparisons = []
    baseline = frame[frame["rho"] == 0.0].sort_values("prompt_id")
    bootstrap_metrics = ["dino_diversity", "clipscore"]
    if not frame["hpsv2"].isna().all():
        bootstrap_metrics.append("hpsv2")
    rho_grid = tuple(float(value) for value in run_config["sampling"]["rho_grid"])
    for rho in rho_grid[1:]:
        method = frame[frame["rho"] == rho].sort_values("prompt_id")
        if method["prompt_id"].tolist() != baseline["prompt_id"].tolist():
            raise RuntimeError("Prompt pairing was lost during aggregation.")
        for metric in bootstrap_metrics:
            result = paired_bootstrap_mean_difference(
                method[metric].to_numpy(),
                baseline[metric].to_numpy(),
                replicates=int(evaluation["bootstrap_replicates"]),
                seed=int(evaluation["bootstrap_seed"]),
            )
            comparisons.append({"rho": rho, "metric": metric, **result})
    write_json(
        metrics_dir / "paired_bootstrap.json",
        {
            "statistical_unit": "prompt",
            "baseline_rho": 0.0,
            "replicates": int(evaluation["bootstrap_replicates"]),
            "comparisons": comparisons,
        },
    )
    make_plot(summary, metrics_dir / "a_strength_curve.png")
    write_json(
        metrics_dir / "evaluation_complete.json",
        {
            "manifest_sha256": manifest_sha,
            "image_count": len(records),
            "prompt_group_count": len(per_prompt),
            "rho_grid": list(rho_grid),
            "dino_definition": "mean unique-pair (1-cosine)/2 on CLS features",
            "clipscore_definition": (
                f"{evaluation['clipscore_weight']} * max(image-text cosine, 0)"
            ),
            "dino_model": evaluation["dino_model"],
            "dino_revision": evaluation["dino_revision"],
            "clip_model": evaluation["clip_model"],
            "clip_revision": evaluation["clip_revision"],
            "clip_subfolder": evaluation["clip_subfolder"],
            "clip_weights_format": evaluation["clip_weights_format"],
            "lpips_definition": (
                None if args.skip_lpips else "AlexNet LPIPS on 256x256 RGB"
            ),
            "hps_version": None if args.skip_hps else evaluation["hps_version"],
            "hps_checkpoint_revision": (
                None
                if args.skip_hps
                else "697403c78157020a1ae59d23f111aa58ced35b0a"
            ),
            "hps_implementation": (
                None
                if args.skip_hps
                else "HPSv2.1 weights loaded into compatible open_clip ViT-H-14"
            ),
            "safe_weight_loading": True,
            "combined_semantic_failure_rate": None,
            "semantic_failure_note": (
                "Not estimated on this development-only curve; formal "
                "TIFA/GenEval or human annotation remains required."
            ),
            "formal_test_data_used": False,
        },
    )
    print(summary.to_string(index=False))
    print(f"Metrics written to {metrics_dir}")


if __name__ == "__main__":
    main()
