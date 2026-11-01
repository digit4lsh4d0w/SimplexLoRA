"""Unit tests for ``simplexlora.calibration.metrics``."""

from __future__ import annotations

import math
import unittest

from simplexlora.calibration import compute_calibration_metrics
from simplexlora.calibration.models import CalibrationRecorderError


class ComputeCalibrationMetricsTests(unittest.TestCase):
    def test_uniform_distribution(self) -> None:
        metrics = compute_calibration_metrics(
            [1.0, 1.0, 1.0, 1.0],
            [4, 4, 4, 4],
            default_lora_rank=16,
            components=["q", "q", "v", "v"],
            top_k=2,
        )
        self.assertEqual(metrics.module_count, 4)
        self.assertEqual(metrics.selected_count, 4)
        self.assertEqual(metrics.zero_rank_count, 0)
        self.assertEqual(metrics.total_rank, 16)
        self.assertAlmostEqual(metrics.rank_utilization, 16 / (16 * 4))
        self.assertAlmostEqual(metrics.entropy, math.log(4))
        self.assertAlmostEqual(metrics.normalized_entropy, 1.0)
        self.assertAlmostEqual(metrics.effective_module_count, 4.0, places=6)
        self.assertAlmostEqual(metrics.hhi, 0.25)
        self.assertAlmostEqual(metrics.inverse_hhi, 4.0)
        self.assertAlmostEqual(metrics.top_1_share, 0.25)
        self.assertAlmostEqual(metrics.top_5_share, 1.0)
        self.assertAlmostEqual(metrics.top_k_share, 0.5)
        self.assertAlmostEqual(metrics.component_shares["q"], 0.5)
        self.assertAlmostEqual(metrics.component_shares["v"], 0.5)
        self.assertEqual(metrics.rank_by_component["q"], 8)
        self.assertEqual(metrics.max_rank, 4)
        self.assertAlmostEqual(metrics.mean_selected_rank, 4.0)

    def test_single_winner(self) -> None:
        metrics = compute_calibration_metrics(
            [4.0, 0.0, 0.0, 0.0],
            [16, 0, 0, 0],
            default_lora_rank=16,
            components=["q", "v", "o", "down"],
            top_k=3,
        )
        self.assertEqual(metrics.selected_count, 1)
        self.assertEqual(metrics.zero_rank_count, 3)
        self.assertAlmostEqual(metrics.entropy, 0.0, places=7)
        self.assertAlmostEqual(metrics.normalized_entropy, 0.0, places=7)
        self.assertAlmostEqual(metrics.effective_module_count, 1.0, places=6)
        self.assertAlmostEqual(metrics.hhi, 1.0)
        self.assertAlmostEqual(metrics.top_1_share, 1.0)

    def test_several_nonzero_components(self) -> None:
        metrics = compute_calibration_metrics(
            [2.0, 1.0, 1.0, 0.0],
            [8, 4, 4, 0],
            default_lora_rank=16,
            components=["q", "k", "v", "o"],
            top_k=10,
        )
        self.assertEqual(metrics.selected_count, 3)
        self.assertEqual(metrics.zero_rank_count, 1)
        # top_k larger than module count is clamped to module count
        self.assertAlmostEqual(metrics.top_k_share, 1.0)
        self.assertAlmostEqual(metrics.top_1_share, 0.5)
        self.assertAlmostEqual(metrics.mean_selected_rank, 16 / 3)

    def test_single_module_normalized_entropy_is_one(self) -> None:
        metrics = compute_calibration_metrics([3.0], [12], default_lora_rank=16, components=["q"], top_k=1)
        self.assertEqual(metrics.normalized_entropy, 1.0)

    def test_empty_inputs_rejected(self) -> None:
        with self.assertRaises(CalibrationRecorderError):
            compute_calibration_metrics([], [], default_lora_rank=16, components=[], top_k=1)

    def test_length_mismatch_rejected(self) -> None:
        with self.assertRaises(CalibrationRecorderError):
            compute_calibration_metrics([1.0, 1.0], [4], default_lora_rank=16, components=["q"], top_k=1)

    def test_negative_weight_rejected(self) -> None:
        with self.assertRaises(CalibrationRecorderError):
            compute_calibration_metrics([-1.0, 2.0], [0, 8], default_lora_rank=16, components=["q", "v"], top_k=1)

    def test_nan_rejected(self) -> None:
        with self.assertRaises(CalibrationRecorderError):
            compute_calibration_metrics(
                [float("nan"), 1.0], [4, 4], default_lora_rank=16, components=["q", "v"], top_k=1
            )

    def test_bad_rank_and_top_k_rejected(self) -> None:
        with self.assertRaises(CalibrationRecorderError):
            compute_calibration_metrics([1.0], [-2], default_lora_rank=16, components=["q"], top_k=1)
        with self.assertRaises(CalibrationRecorderError):
            compute_calibration_metrics([1.0], [4], default_lora_rank=0, components=["q"], top_k=1)
        with self.assertRaises(CalibrationRecorderError):
            compute_calibration_metrics([1.0], [4], default_lora_rank=16, components=["q"], top_k=0)


if __name__ == "__main__":
    unittest.main()
