"""Tests for derived optimization-level filtering of Decompile-Bench."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from datasets import Dataset

from simplexlora.experiments.decompile import select_dataset
from simplexlora.tools.optimization_clusters import (
    CLUSTER_SCHEMA_VERSION,
    CLUSTERS_FILENAME,
    MANIFEST_FILENAME,
    _collect_explicit_labels,
    assembly_features,
    declared_optimization_level,
    load_optimization_indices,
    project_key,
)


class OptimizationClusterTests(unittest.TestCase):
    def test_declared_level_extraction_is_conservative(self) -> None:
        self.assertEqual(declared_optimization_level("/repo/build_O0/src/a.c"), "O0")
        self.assertEqual(declared_optimization_level("repo/build-O1/src/a.c"), "O1")
        self.assertEqual(declared_optimization_level("/repo/run-O2/a.c"), "O2")
        self.assertEqual(declared_optimization_level("/repo/build_O3/a.c"), "O3")
        self.assertIsNone(declared_optimization_level("/repo/build/src/a.c"))
        self.assertEqual(project_key("/owner[P]project/src/a.c"), "owner[P]project")

    def test_assembly_features_are_finite_and_differentiate_unoptimized_code(self) -> None:
        unoptimized = "pushq %rbp\nmovq %rsp, %rbp\nmovl -4(%rbp), %eax\npopq %rbp\nretq"
        optimized = "lea (%rdi,%rsi), %eax\nretq"
        source = "int add(int a, int b) { return a + b; }"
        first = assembly_features(unoptimized, source)
        second = assembly_features(optimized, source)
        self.assertEqual(first.shape, second.shape)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(first[0], second[0])
        self.assertGreater(first[5], second[5])

    def test_explicit_label_indices_are_global_dataset_indices(self) -> None:
        dataset = Dataset.from_dict(
            {
                "file": ["unknown/src.c", "repo/build_O0/src.c"],
                "asm": ["retq", "retq"],
                "code": ["void a(void) {}", "void b(void) {}"],
            }
        )
        labeled = _collect_explicit_labels(dataset)
        self.assertEqual(len(labeled), 1)
        self.assertEqual(labeled[0].dataset_index, 1)

    def test_cluster_index_loader_applies_confidence(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            mapping = Dataset.from_dict(
                {
                    "dataset_index": [0, 1, 2, 3],
                    "optimization_level": ["O0", "O0", "O1", "O0"],
                    "confidence": [1.0, 0.97, 0.99, 0.995],
                    "source": ["path", "assembly_o0_model", "path", "assembly_o0_model"],
                }
            )
            mapping.to_parquet(str(output / CLUSTERS_FILENAME))
            manifest = {
                "schema_version": CLUSTER_SCHEMA_VERSION,
                "dataset_rows": 4,
            }
            (output / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

            self.assertEqual(
                load_optimization_indices(output, level="O0", min_confidence=0.98, dataset_rows=4),
                [0, 3],
            )
            self.assertEqual(load_optimization_indices(output, level="O1", min_confidence=0.0), [2])
            with self.assertRaisesRegex(ValueError, "built for 4 rows"):
                load_optimization_indices(output, level="O0", dataset_rows=5)

    def test_select_dataset_filters_o0_and_uses_all_rows_without_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = Dataset.from_dict(
                {
                    "name": [f"func_{index}" for index in range(12)],
                    "code": [f"int func_{index}(void) {{ return {index}; }}" for index in range(12)],
                    "asm": [f"movl ${index}, %eax\nretq" for index in range(12)],
                    "file": [f"project/src/file_{index}.c" for index in range(12)],
                }
            )
            dataset_path = root / "raw"
            raw.save_to_disk(str(dataset_path))
            cluster_dir = root / "clusters"
            cluster_dir.mkdir()
            levels = ["O0" if index % 2 == 0 else "O1" for index in range(12)]
            Dataset.from_dict(
                {
                    "dataset_index": list(range(12)),
                    "optimization_level": levels,
                    "confidence": [1.0] * 12,
                    "source": ["path"] * 12,
                }
            ).to_parquet(str(cluster_dir / CLUSTERS_FILENAME))
            (cluster_dir / MANIFEST_FILENAME).write_text(
                json.dumps({"schema_version": CLUSTER_SCHEMA_VERSION, "dataset_rows": 12}),
                encoding="utf-8",
            )

            train, validation = select_dataset(
                dataset_path,
                max_train=5,
                max_validation=0,
                seed=7,
                optimization_level="O0",
                optimization_clusters=cluster_dir,
                optimization_min_confidence=0.0,
            )

            self.assertEqual(len(train), 5)
            self.assertEqual(len(validation), 0)
            self.assertTrue(all(int(row["name"].split("_")[-1]) % 2 == 0 for row in train))


if __name__ == "__main__":
    unittest.main()
