import pathlib
import sys
import unittest

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATAFLOW_DIR = REPO_ROOT / "src" / "dataflow"
if str(DATAFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(DATAFLOW_DIR))

from SwRectify import sliding_window
from feature_extraction import extract_emg_features, prepare_regression_data


class RegressionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        # Small synthetic recording:
        # - 10 samples
        # - 2 EMG channels
        # - one monotonic target and one scaled copy
        self.emg = np.arange(20, dtype=np.float32).reshape(10, 2)
        self.targets = np.column_stack(
            [
                np.arange(10, dtype=np.float32),
                np.arange(10, dtype=np.float32) * 10.0,
            ]
        )

    def test_last_sample_alignment(self) -> None:
        windows = sliding_window(
            self.emg,
            self.targets,
            fs=10,
            window_ms=400,
            stride_ms=200,
            target_mode="last",
            target_names=["angle_a", "angle_b"],
        )

        np.testing.assert_array_equal(windows["window_start_indices"], np.array([0, 2, 4, 6]))
        np.testing.assert_array_equal(windows["window_end_indices"], np.array([4, 6, 8, 10]))
        np.testing.assert_array_equal(windows["window_center_indices"], np.array([1, 3, 5, 7]))
        np.testing.assert_array_equal(windows["target_alignment_indices"], np.array([3, 5, 7, 9]))
        np.testing.assert_allclose(
            windows["target_values"],
            np.array(
                [
                    [3.0, 30.0],
                    [5.0, 50.0],
                    [7.0, 70.0],
                    [9.0, 90.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_center_and_mean_alignment(self) -> None:
        center_windows = sliding_window(
            self.emg,
            self.targets,
            fs=10,
            window_ms=400,
            stride_ms=200,
            target_mode="center",
        )
        mean_windows = sliding_window(
            self.emg,
            self.targets,
            fs=10,
            window_ms=400,
            stride_ms=200,
            target_mode="mean",
        )

        np.testing.assert_array_equal(center_windows["target_alignment_indices"], np.array([1, 3, 5, 7]))
        np.testing.assert_allclose(
            center_windows["target_values"],
            np.array(
                [
                    [1.0, 10.0],
                    [3.0, 30.0],
                    [5.0, 50.0],
                    [7.0, 70.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            mean_windows["target_values"],
            np.array(
                [
                    [1.5, 15.0],
                    [3.5, 35.0],
                    [5.5, 55.0],
                    [7.5, 75.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_feature_extraction_and_regression_preparation(self) -> None:
        windows = sliding_window(
            self.emg,
            self.targets[:, :1],
            fs=10,
            window_ms=400,
            stride_ms=200,
            target_mode="last",
            target_names=["angle_a"],
        )
        feature_set = extract_emg_features(windows)
        regression = prepare_regression_data(feature_set, normalize=True)

        self.assertEqual(feature_set["feature_tensor"].shape, (4, 2, 5))
        self.assertEqual(feature_set["feature_matrix"].shape, (4, 10))
        self.assertEqual(regression["x"].shape, (4, 10))
        self.assertEqual(regression["y"].shape, (4, 1))
        self.assertEqual(regression["target_names"], ["angle_a"])
        np.testing.assert_allclose(regression["x"].mean(axis=0), np.zeros(10), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
