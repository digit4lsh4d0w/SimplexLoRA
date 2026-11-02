"""Build reproducible optimization-level clusters for Decompile-Bench rows.

The released training dataset omits the ``opt`` column. Some source paths contain
an explicit ``build_O0`` ... ``build_O3`` component; those labels are retained as
ground truth. A grouped-by-project classifier predicts the remaining rows from
assembly features and records provenance and confidence for every assignment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from simplexlora.calibration.models import atomic_write_text, dump_json

OPTIMIZATION_LEVELS = ("O0", "O1", "O2", "O3")
CLUSTER_SCHEMA_VERSION = "1.0"
CLUSTERS_FILENAME = "optimization_clusters.parquet"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_O0_THRESHOLD = 0.98

_DECLARED_OPTIMIZATION = re.compile(
    r"(?:build[_-]?|[-_])(O[0-3])(?:[/_.]|$)",
    re.IGNORECASE,
)


def default_cluster_dir() -> Path:
    """Return the shared derived-data location outside the repository."""
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "simplexlora" / "optimization-clusters" / "decompile-bench-v1"


def declared_optimization_level(file_path: str) -> str | None:
    """Extract an exact level only when the path contains a compiler build marker."""
    match = _DECLARED_OPTIMIZATION.search(file_path)
    return match.group(1).upper() if match else None


def project_key(file_path: str) -> str:
    """Best-effort repository identity used to avoid grouped-validation leakage."""
    normalized = file_path.strip().lstrip("/")
    return normalized.split("/", 1)[0]


def assembly_features(asm: str, code: str) -> np.ndarray:
    """Cheap deterministic assembly/code features used by the classifiers.

    These capture the strongest optimization-level signals without tokenizing
    or executing any source code.
    """
    line_count = asm.count("\n") + 1
    code_line_count = code.count("\n") + 1
    line_denominator = max(line_count, 1)
    text = asm
    return np.asarray(
        [
            math.log1p(line_count),
            math.log1p(len(asm)),
            math.log1p(len(code)),
            math.log1p(code_line_count),
            math.log1p(line_count / max(code_line_count, 1)),
            text.count("%rbp") / line_denominator,
            text.count("%rsp") / line_denominator,
            text.count("%rip") / line_denominator,
            text.count("call") / line_denominator,
            text.count("pus") / line_denominator,
            text.count("pop") / line_denominator,
            text.count("lea") / line_denominator,
            text.count("j") / line_denominator,
            text.count("mov") / line_denominator,
            text.count("cmp") / line_denominator,
            text.count("test") / line_denominator,
            text.count("add") / line_denominator,
            text.count("sub") / line_denominator,
            text.count("imul") / line_denominator,
            text.count("idiv") / line_denominator,
            text.count("div") / line_denominator,
            text.count("xmm") / line_denominator,
            text.count("ymm") / line_denominator,
            text.count("zmm") / line_denominator,
            text.count("movdqa") / line_denominator,
            text.count("call") / max(text.count("j"), 1),
            float(text.startswith("pushq %rbp")),
            float(text.startswith("endbr64")),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class _LabeledRow:
    dataset_index: int
    level: str
    project: str
    asm: str
    code: str


@dataclass
class _Classifiers:
    scaler: StandardScaler
    multiclass: LogisticRegression
    o0_binary: LogisticRegression


def _collect_explicit_labels(dataset: Dataset) -> list[_LabeledRow]:
    labeled: list[_LabeledRow] = []
    columns = dataset.select_columns(["file", "asm", "code"])
    position = 0
    for batch in columns.iter(batch_size=100_000):
        batch_size = len(batch["file"])
        for offset in range(batch_size):
            file_path = str(batch["file"][offset])
            level = declared_optimization_level(file_path)
            if level is None:
                continue
            labeled.append(
                _LabeledRow(
                    dataset_index=position + offset,
                    level=level,
                    project=project_key(file_path),
                    asm=str(batch["asm"][offset]),
                    code=str(batch["code"][offset]),
                )
            )
        position += batch_size
    return labeled


def _feature_matrix(rows: list[_LabeledRow]) -> np.ndarray:
    return np.stack([assembly_features(row.asm, row.code) for row in rows])


def _fit_classifiers(
    labeled: list[_LabeledRow],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[_Classifiers, dict[str, Any]]:
    features = _feature_matrix(labeled)
    labels = np.asarray([row.level for row in labeled])
    groups = np.asarray([row.project for row in labeled])
    split = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_indices, test_indices = next(split.split(features, labels, groups))

    scaler = StandardScaler().fit(features[train_indices])
    train_features = scaler.transform(features[train_indices])
    test_features = scaler.transform(features[test_indices])
    multiclass = LogisticRegression(max_iter=500, class_weight="balanced").fit(train_features, labels[train_indices])
    multiclass_predictions = multiclass.predict(test_features)

    o0_train = (labels[train_indices] == "O0").astype(np.int8)
    o0_test = (labels[test_indices] == "O0").astype(np.int8)
    o0_binary = LogisticRegression(max_iter=500, class_weight="balanced").fit(train_features, o0_train)
    o0_probabilities = o0_binary.predict_proba(test_features)[:, 1]

    metrics = {
        "labeled_rows": len(labeled),
        "validation_rows": len(test_indices),
        "validation_groups": len(set(groups[test_indices])),
        "multiclass_accuracy": float(accuracy_score(labels[test_indices], multiclass_predictions)),
        "multiclass_confusion_matrix": confusion_matrix(
            labels[test_indices],
            multiclass_predictions,
            labels=list(OPTIMIZATION_LEVELS),
        ).tolist(),
        "multiclass_report": classification_report(
            labels[test_indices],
            multiclass_predictions,
            labels=list(OPTIMIZATION_LEVELS),
            output_dict=True,
            zero_division=0,
        ),
        "o0_probability_curve": [
            _binary_threshold_metrics(float(threshold), o0_test, o0_probabilities)
            for threshold in (0.5, 0.9, 0.95, 0.98, 0.99)
        ],
    }

    final_scaler = StandardScaler().fit(features)
    final_features = final_scaler.transform(features)
    final_multiclass = LogisticRegression(max_iter=500, class_weight="balanced").fit(final_features, labels)
    final_o0 = LogisticRegression(max_iter=500, class_weight="balanced").fit(
        final_features,
        (labels == "O0").astype(np.int8),
    )
    return _Classifiers(final_scaler, final_multiclass, final_o0), metrics


def _binary_threshold_metrics(threshold: float, truth: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities >= threshold
    true_positive = int(np.sum(predicted & (truth == 1)))
    false_positive = int(np.sum(predicted & (truth == 0)))
    false_negative = int(np.sum(~predicted & (truth == 1)))
    return {
        "threshold": threshold,
        "predicted_positive": int(predicted.sum()),
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(true_positive + false_negative, 1),
    }


def cluster_optimization_levels(
    dataset: Dataset,
    *,
    output_dir: Path,
    o0_threshold: float = DEFAULT_O0_THRESHOLD,
    validation_fraction: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Assign every row to one of O0-O3 and persist index-only clusters."""
    if not 0.5 <= o0_threshold <= 1.0:
        raise ValueError(f"o0_threshold must be within [0.5, 1], got {o0_threshold}")
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError(f"validation_fraction must be within [0.05, 0.5], got {validation_fraction}")

    labeled = _collect_explicit_labels(dataset)
    label_counts = Counter(row.level for row in labeled)
    if set(label_counts) != set(OPTIMIZATION_LEVELS):
        raise ValueError(f"path-derived labels do not cover all optimization levels: {dict(label_counts)}")
    print(f"Explicit path labels: {dict(sorted(label_counts.items()))}")
    print("Training grouped optimization-level classifiers...")
    classifiers, metrics = _fit_classifiers(labeled, validation_fraction=validation_fraction, seed=seed)
    print(
        "Grouped validation: "
        f"multiclass accuracy={metrics['multiclass_accuracy']:.4f}, "
        f"O0 threshold={o0_threshold:.3f}"
    )

    dataset_indices: list[int] = []
    levels: list[str] = []
    confidences: list[float] = []
    sources: list[str] = []
    counts = Counter[str]()
    columns = dataset.select_columns(["file", "asm", "code"])
    position = 0
    print(f"Clustering {len(dataset):,} dataset rows...")
    for batch in columns.iter(batch_size=8192):
        features = np.stack(
            [assembly_features(str(asm), str(code)) for asm, code in zip(batch["asm"], batch["code"], strict=True)]
        )
        scaled = classifiers.scaler.transform(features)
        multiclass_probabilities = classifiers.multiclass.predict_proba(scaled)
        multiclass_classes = list(classifiers.multiclass.classes_)
        non_o0_positions = [multiclass_classes.index(level) for level in ("O1", "O2", "O3")]
        non_o0_predictions = np.asarray(
            [multiclass_classes[index] for index in non_o0_positions],
            dtype=object,
        )
        o0_probabilities = classifiers.o0_binary.predict_proba(scaled)[:, 1]

        for offset, file_path in enumerate(batch["file"]):
            dataset_index = position + offset
            level = declared_optimization_level(str(file_path))
            if level is not None:
                confidence = 1.0
                source = "path"
            elif o0_probabilities[offset] >= o0_threshold:
                level = "O0"
                confidence = float(o0_probabilities[offset])
                source = "assembly_o0_model"
            else:
                non_o0_probabilities = multiclass_probabilities[offset, non_o0_positions]
                best = int(np.argmax(non_o0_probabilities))
                level = str(non_o0_predictions[best])
                confidence = float(non_o0_probabilities[best])
                source = "assembly_multiclass_model"

            dataset_indices.append(dataset_index)
            levels.append(level)
            confidences.append(confidence)
            sources.append(source)
            counts[level] += 1
        position += len(batch["file"])
        if position % 100_000 < len(batch["file"]):
            print(f"  clustered {position:,} / {len(dataset):,}")

    if position != len(dataset):
        raise RuntimeError(f"clustered {position} rows, dataset contains {len(dataset)}")

    mapping = Dataset.from_dict(
        {
            "dataset_index": dataset_indices,
            "optimization_level": levels,
            "confidence": confidences,
            "source": sources,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / CLUSTERS_FILENAME
    mapping.to_parquet(str(mapping_path))
    for level in OPTIMIZATION_LEVELS:
        level_rows = mapping.filter(lambda row, value=level: row["optimization_level"] == value)
        level_rows.to_parquet(str(output_dir / f"{level}.parquet"))

    source_counts = Counter(sources)
    manifest = {
        "schema_version": CLUSTER_SCHEMA_VERSION,
        "dataset_rows": len(dataset),
        "o0_threshold": o0_threshold,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "path_label_counts": dict(sorted(label_counts.items())),
        "cluster_counts": dict(sorted(counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "metrics": metrics,
    }
    atomic_write_text(output_dir / MANIFEST_FILENAME, dump_json(manifest) + "\n", overwrite=True)
    return manifest


def load_optimization_indices(
    cluster_dir: Path,
    *,
    level: str,
    min_confidence: float = 0.0,
    dataset_rows: int | None = None,
) -> list[int]:
    """Load validated source-dataset indices for one derived cluster."""
    if level not in OPTIMIZATION_LEVELS:
        raise ValueError(f"level must be one of {OPTIMIZATION_LEVELS}, got {level!r}")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError(f"min_confidence must be within [0, 1], got {min_confidence}")
    manifest_path = cluster_dir / MANIFEST_FILENAME
    mapping_path = cluster_dir / CLUSTERS_FILENAME
    if not manifest_path.is_file() or not mapping_path.is_file():
        raise FileNotFoundError(
            f"optimization clusters not found in {cluster_dir}; run `uv run simplexlora-cluster-optimization` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CLUSTER_SCHEMA_VERSION:
        raise ValueError(f"unsupported optimization cluster schema: {manifest.get('schema_version')!r}")
    if dataset_rows is not None and int(manifest["dataset_rows"]) != dataset_rows:
        raise ValueError(
            f"optimization cluster mapping was built for {manifest['dataset_rows']} rows, "
            f"but the loaded dataset has {dataset_rows}"
        )
    mapping = Dataset.from_parquet(str(mapping_path))
    indices: list[int] = []
    for batch in mapping.iter(batch_size=100_000):
        for dataset_index, assigned_level, confidence in zip(
            batch["dataset_index"],
            batch["optimization_level"],
            batch["confidence"],
            strict=True,
        ):
            if assigned_level == level and float(confidence) >= min_confidence:
                index = int(dataset_index)
                if not 0 <= index < int(manifest["dataset_rows"]):
                    raise ValueError(f"cluster index {index} is outside dataset size {manifest['dataset_rows']}")
                indices.append(index)
    if not indices:
        raise ValueError(f"optimization cluster {level} is empty at min_confidence={min_confidence}")
    return indices


def _default_dataset_path() -> Path:
    cache_root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    repo_dir = cache_root / "datasets--LLM4Binary--decompile-bench"
    ref_file = repo_dir / "refs" / "main"
    if not ref_file.is_file():
        raise FileNotFoundError(
            "decompile-bench cache not found; run `hf download LLM4Binary/decompile-bench --type dataset`"
        )
    return repo_dir / "snapshots" / ref_file.read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--o0-threshold", type=float, default=DEFAULT_O0_THRESHOLD)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path) if args.dataset_path else _default_dataset_path()
    output_dir = Path(args.output_dir) if args.output_dir else default_cluster_dir()
    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"expected a Dataset, got {type(dataset).__name__}")
    manifest = cluster_optimization_levels(
        dataset,
        output_dir=output_dir,
        o0_threshold=args.o0_threshold,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    print(f"Optimization clusters written to: {output_dir}")


if __name__ == "__main__":
    main()
