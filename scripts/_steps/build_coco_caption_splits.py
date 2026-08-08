#!/usr/bin/env python3
"""Build deterministic COCO development and formal-test caption splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import CLIPTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feedback_cads.experiments import write_json, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "data/coco/annotations/captions_val2017.json",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=PROJECT_ROOT
        / "models/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5"
        / "snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/tokenizer",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/coco"
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--dev-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=500)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_caption(value: str) -> str:
    return " ".join(value.strip().split())


def main() -> None:
    args = parse_args()
    annotations_path = args.annotations.resolve()
    if not annotations_path.is_file():
        raise FileNotFoundError(annotations_path)
    if args.min_tokens <= 0 or args.max_tokens < args.min_tokens:
        raise ValueError("Invalid token-length interval.")
    if args.dev_count <= 0 or args.test_count <= 0:
        raise ValueError("Development and test counts must be positive.")

    tokenizer = CLIPTokenizer.from_pretrained(
        args.tokenizer.resolve(), local_files_only=True
    )
    source: dict[str, Any] = json.loads(
        annotations_path.read_text(encoding="utf-8")
    )
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    eligible_annotation_count = 0
    for annotation in source["annotations"]:
        caption = normalize_caption(str(annotation["caption"]))
        token_ids = tokenizer(
            caption,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        content_tokens = len(token_ids)
        encoded_with_special = content_tokens + tokenizer.num_special_tokens_to_add()
        if not args.min_tokens <= content_tokens <= args.max_tokens:
            continue
        if encoded_with_special > tokenizer.model_max_length:
            continue
        eligible_annotation_count += 1
        by_image[int(annotation["image_id"])].append(
            {
                "annotation_id": int(annotation["id"]),
                "image_id": int(annotation["image_id"]),
                "caption": caption,
                "content_token_count": content_tokens,
                "encoded_token_count": encoded_with_special,
            }
        )

    selected_by_image = {
        image_id: min(rows, key=lambda row: row["annotation_id"])
        for image_id, rows in by_image.items()
    }
    image_ids = sorted(selected_by_image)
    random.Random(args.seed).shuffle(image_ids)
    required_image_count = args.dev_count + args.test_count
    if len(image_ids) < required_image_count:
        raise RuntimeError(
            f"Only {len(image_ids)} eligible image IDs; at least "
            f"{required_image_count} are required."
        )

    def make_records(ids: list[int], split: str) -> list[dict[str, Any]]:
        records = []
        for split_index, image_id in enumerate(ids):
            selected = selected_by_image[image_id]
            records.append(
                {
                    "prompt_id": f"coco_{image_id:012d}",
                    "category": "coco_caption",
                    "prompt": selected["caption"],
                    "source": "MS-COCO 2017 validation captions",
                    "split": split,
                    "split_index": split_index,
                    "image_id": image_id,
                    "annotation_id": selected["annotation_id"],
                    "content_token_count": selected["content_token_count"],
                    "encoded_token_count": selected["encoded_token_count"],
                }
            )
        return records

    test_stop = args.dev_count + args.test_count
    dev = make_records(
        image_ids[: args.dev_count],
        f"COCO-Dev-{args.dev_count}",
    )
    test = make_records(
        image_ids[args.dev_count : test_stop],
        f"COCO-Test-{args.test_count}",
    )
    if {row["image_id"] for row in dev} & {row["image_id"] for row in test}:
        raise RuntimeError("COCO development and test image IDs overlap.")

    output_dir = args.output_dir.resolve()
    dev_path = output_dir / f"coco_dev{args.dev_count}.jsonl"
    test_path = output_dir / f"coco_test{args.test_count}.jsonl"
    write_jsonl(dev_path, dev)
    write_jsonl(test_path, test)
    write_json(
        output_dir / "coco_caption_splits.json",
        {
            "source": "MS-COCO 2017 validation captions",
            "source_path": str(annotations_path),
            "source_sha256": sha256(annotations_path),
            "selection": {
                "one_caption_per_image": True,
                "caption_choice": "minimum annotation ID among eligible captions",
                "content_token_interval_inclusive": [
                    args.min_tokens,
                    args.max_tokens,
                ],
                "tokenizer_path": str(args.tokenizer.resolve()),
                "tokenizer_model_max_length": tokenizer.model_max_length,
                "shuffle_input": "ascending eligible image IDs",
                "shuffle_algorithm": "Python random.Random(seed).shuffle",
                "shuffle_seed": args.seed,
                "dev_count_requested": args.dev_count,
                "test_count_requested": args.test_count,
            },
            "eligible_annotation_count": eligible_annotation_count,
            "eligible_image_count": len(image_ids),
            "dev_count": len(dev),
            "test_count": len(test),
            "dev_sha256": sha256(dev_path),
            "test_sha256": sha256(test_path),
            "image_ids_disjoint": True,
        },
    )
    print(f"Wrote {len(dev)} development and {len(test)} test captions.")


if __name__ == "__main__":
    main()
