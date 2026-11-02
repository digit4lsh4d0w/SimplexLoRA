"""Create a deterministic O0 subset for the vendored Simplex LoRA baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, load_from_disk


def default_dataset_path() -> Path:
    cache_root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_root / "datasets--LLM4Binary--decompile-bench"
    revision = (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
    return repo_dir / "snapshots" / revision


def default_cluster_path() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "simplexlora" / "optimization-clusters" / "decompile-bench-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--optimization-clusters", default=None)
    parser.add_argument("--output-dir", default="/tmp/decompile-bench-o0-2000")
    parser.add_argument("--max-samples", type=int, default=2_000)
    parser.add_argument("--min-confidence", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_samples < 1:
        raise ValueError(f"--max-samples must be positive, got {args.max_samples}")
    dataset_path = Path(args.dataset_path) if args.dataset_path else default_dataset_path()
    cluster_dir = Path(args.optimization_clusters) if args.optimization_clusters else default_cluster_path()
    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"expected a Dataset, got {type(dataset).__name__}")
    manifest = json.loads((cluster_dir / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest["dataset_rows"]) != len(dataset):
        raise ValueError(f"cluster mapping has {manifest['dataset_rows']} rows, dataset has {len(dataset)}")
    mapping = Dataset.from_parquet(str(cluster_dir / "optimization_clusters.parquet"))
    indices = [
        int(index)
        for index, level, confidence in zip(
            mapping["dataset_index"],
            mapping["optimization_level"],
            mapping["confidence"],
            strict=True,
        )
        if level == "O0" and float(confidence) >= args.min_confidence
    ]
    if not indices:
        raise ValueError("O0 selection is empty")
    selected = dataset.select(indices).shuffle(seed=args.seed).select(range(min(len(indices), args.max_samples)))
    selected.save_to_disk(args.output_dir)
    print(f"Saved {len(selected)} O0 rows to {args.output_dir}")


if __name__ == "__main__":
    main()
