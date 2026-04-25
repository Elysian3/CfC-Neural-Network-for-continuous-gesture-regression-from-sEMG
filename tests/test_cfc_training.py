import pathlib
import sys
import unittest

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "src" / "deep learning"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from train import (
    RecordingFeatures,
    build_blocked_sequence_splits,
    build_sequence_split,
    compute_regression_metrics,
)


class CfCTrainingHelpersTests(unittest.TestCase):
    def test_build_sequence_split_uses_last_window_target(self) -> None:
        recording = RecordingFeatures(
            recording_id="demo.mat",
            x_windows=np.array(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                    [5.0, 50.0],
                ],
                dtype=np.float32,
            ),
            y_windows=np.array([[11.0], [22.0], [33.0], [44.0], [55.0]], dtype=np.float32),
            target_alignment_indices=np.array([100, 200, 300, 400, 500], dtype=np.int32),
            feature_names=["f1", "f2"],
            target_names=["angle_1"],
            fs=100.0,
        )

        split = build_sequence_split([recording], seq_len=3, seq_stride=1)

        self.assertEqual(split.x.shape, (3, 3, 2))
        self.assertEqual(split.y.shape, (3, 1))
        np.testing.assert_allclose(split.y[:, 0], np.array([33.0, 44.0, 55.0], dtype=np.float32))
        np.testing.assert_array_equal(split.alignment_indices, np.array([300, 400, 500], dtype=np.int32))
        np.testing.assert_allclose(split.time_s, np.array([3.0, 4.0, 5.0], dtype=np.float32))

    def test_compute_regression_metrics_matches_perfect_prediction(self) -> None:
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        y_pred = y_true.copy()

        metrics = compute_regression_metrics(y_true, y_pred)

        np.testing.assert_allclose(metrics["mae_by_target"], np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(metrics["rmse_by_target"], np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(metrics["r2_by_target"], np.ones(2, dtype=np.float32))
        self.assertAlmostEqual(metrics["mae_mean"], 0.0)
        self.assertAlmostEqual(metrics["rmse_mean"], 0.0)
        self.assertAlmostEqual(metrics["r2_mean"], 1.0)

    def test_blocked_time_split_uses_all_three_partitions(self) -> None:
        recording = RecordingFeatures(
            recording_id="demo.mat",
            x_windows=np.arange(80, dtype=np.float32).reshape(40, 2),
            y_windows=np.arange(40, dtype=np.float32).reshape(40, 1),
            target_alignment_indices=np.arange(40, dtype=np.int32) * 10,
            feature_names=["f1", "f2"],
            target_names=["angle_1"],
            fs=100.0,
        )

        splits = build_blocked_sequence_splits(
            [recording],
            seq_len=4,
            seq_stride=1,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            gap_windows=2,
        )

        self.assertGreater(splits["train"].x.shape[0], 0)
        self.assertGreater(splits["val"].x.shape[0], 0)
        self.assertGreater(splits["test"].x.shape[0], 0)
        self.assertTrue(np.all(splits["train"].recording_ids == "demo.mat"))
        self.assertTrue(np.all(splits["val"].recording_ids == "demo.mat"))
        self.assertTrue(np.all(splits["test"].recording_ids == "demo.mat"))


if __name__ == "__main__":
    unittest.main()
