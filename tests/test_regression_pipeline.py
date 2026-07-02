import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATAFLOW_DIR = REPO_ROOT / "src" / "dataflow"
if str(DATAFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(DATAFLOW_DIR))

from SwRectify import sliding_window
from doa_mapping import DB8_OFFICIAL_W, DB8_TO_DB2_CHANNELS, DOA5_NAMES, DOA5_W, apply_linear_doa_mapping
from feature_extraction import (
    _compute_rest_thresholds,
    extract_emg_features,
    prepare_regression_data,
    run_feature_pipeline,
)


class RegressionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.emg = np.arange(20, dtype=np.float32).reshape(10, 2)
        self.targets = np.column_stack(
            [
                np.arange(10, dtype=np.float32),
                np.arange(10, dtype=np.float32) * 10.0,
            ]
        )

    # ── Window alignment tests ───────────────────────────────────────────────

    def test_last_sample_alignment(self) -> None:
        windows = sliding_window(
            self.emg,
            self.targets,
            fs=10,
            window_ms=400,
            stride_ms=200,
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

    def test_positive_target_offset_drops_tail_windows(self) -> None:
        windows = sliding_window(
            self.emg,
            self.targets,
            fs=10,
            window_ms=400,
            stride_ms=200,
            target_offset_samples=2,
            target_names=["angle_a", "angle_b"],
        )

        np.testing.assert_array_equal(windows["window_start_indices"], np.array([0, 2, 4]))
        np.testing.assert_array_equal(windows["window_end_indices"], np.array([4, 6, 8]))
        np.testing.assert_array_equal(windows["target_alignment_indices"], np.array([5, 7, 9]))
        self.assertEqual(windows["n_windows"], 3)
        self.assertEqual(windows["rectified"].shape[0], 3)
        np.testing.assert_allclose(
            windows["target_values"],
            np.array(
                [
                    [5.0, 50.0],
                    [7.0, 70.0],
                    [9.0, 90.0],
                ],
                dtype=np.float32,
            ),
        )

    # ── Feature extraction and preparation tests ─────────────────────────────

    def test_feature_extraction_and_regression_preparation(self) -> None:
        windows = sliding_window(
            self.emg,
            self.targets[:, :1],
            fs=10,
            window_ms=400,
            stride_ms=200,
            target_names=["angle_a"],
        )
        feature_set = extract_emg_features(
            windows,
            zc_threshold=1e-8,
            ssc_threshold=1e-8,
        )
        regression = prepare_regression_data(feature_set, normalize=True)

        self.assertEqual(feature_set["feature_tensor"].shape, (4, 2, 5))
        self.assertEqual(feature_set["feature_matrix"].shape, (4, 10))
        self.assertEqual(regression["x"].shape, (4, 10))
        self.assertEqual(regression["y"].shape, (4, 1))
        self.assertEqual(regression["target_names"], ["angle_a"])
        np.testing.assert_allclose(regression["x"].mean(axis=0), np.zeros(10), atol=1e-5)
        np.testing.assert_allclose(feature_set["zc_thresholds"], np.full(2, 1e-8, dtype=np.float32))
        np.testing.assert_allclose(feature_set["ssc_thresholds"], np.full(2, 1e-8, dtype=np.float32))

    def test_feature_extraction_can_emit_rms_only(self) -> None:
        windows = {
            "unrectified": np.array(
                [
                    [[3.0, 4.0], [0.0, 0.0]],
                    [[1.0, 2.0], [1.0, 2.0]],
                ],
                dtype=np.float32,
            ),
            "rectified": np.array(
                [
                    [[3.0, 4.0], [0.0, 0.0]],
                    [[1.0, 2.0], [1.0, 2.0]],
                ],
                dtype=np.float32,
            ),
            "window_start_indices": np.array([0, 1], dtype=np.int32),
            "window_end_indices": np.array([2, 3], dtype=np.int32),
            "window_center_indices": np.array([0, 1], dtype=np.int32),
            "window_size": 2,
            "stride": 1,
            "window_ms": 100,
            "stride_ms": 0.5,
            "fs": 2000.0,
        }

        feature_set = extract_emg_features(windows, feature_order=("rms",))

        self.assertEqual(feature_set["feature_order"], ["rms"])
        self.assertEqual(feature_set["feature_tensor"].shape, (2, 2, 1))
        np.testing.assert_allclose(
            feature_set["feature_matrix"],
            np.array([[np.sqrt(4.5), np.sqrt(8.0)], [1.0, 2.0]], dtype=np.float32),
        )
        self.assertNotIn("zc", feature_set)
        self.assertNotIn("ssc", feature_set)
        self.assertNotIn("zc_thresholds", feature_set)
        self.assertNotIn("ssc_thresholds", feature_set)

    def test_explicit_zc_ssc_thresholds_are_used(self) -> None:
        windows = sliding_window(
            self.emg - np.mean(self.emg, axis=0, keepdims=True),
            self.targets[:, :1],
            fs=10,
            window_ms=400,
            stride_ms=200,
            target_names=["angle_a"],
        )

        feature_set = extract_emg_features(
            windows,
            zc_threshold=1e-8,
            ssc_threshold=1e-8,
        )

        np.testing.assert_allclose(feature_set["zc_thresholds"], np.full(2, 1e-8, dtype=np.float32))
        np.testing.assert_allclose(feature_set["ssc_thresholds"], np.full(2, 1e-8, dtype=np.float32))

    # ── DoA mapping tests ────────────────────────────────────────────────────

    def test_doa5_mapping_has_expected_shape_and_names(self) -> None:
        self.assertEqual(DOA5_W.shape, (5, 22))
        self.assertEqual(
            DOA5_NAMES,
            (
                "thumb_rotation",
                "thumb_flexion",
                "index_flexion",
                "middle_flexion",
                "ring_little_flexion",
            ),
        )

        glove = np.arange(44, dtype=np.float32).reshape(2, 22)
        mapped = apply_linear_doa_mapping(glove)

        self.assertEqual(mapped.shape, (2, 5))
        np.testing.assert_allclose(mapped, glove @ DOA5_W.T)

    def test_doa5_mapping_accepts_direct_db8_glove_shape(self) -> None:
        glove = np.arange(36, dtype=np.float32).reshape(2, 18)
        mapped = apply_linear_doa_mapping(glove)

        self.assertEqual(mapped.shape, (2, 5))
        np.testing.assert_allclose(mapped, glove @ DB8_OFFICIAL_W.T)

    def test_doa5_mapping_uses_official_db8_weights_remapped_to_db2(self) -> None:
        self.assertEqual(
            DB8_TO_DB2_CHANNELS,
            (1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22),
        )
        expected = np.zeros((5, 22), dtype=np.float32)
        expected[0, 0] = 0.6390
        expected[0, 1] = 0.3830
        expected[0, 3] = -0.6390
        expected[0, 19] = -0.1900
        expected[1, 2] = 1.0
        expected[2, 4] = 0.4
        expected[2, 5] = 0.6
        expected[3, 7] = 0.4
        expected[3, 8] = 0.6
        expected[4, 11] = 0.1667
        expected[4, 12] = 0.3333
        expected[4, 15] = 0.1667
        expected[4, 16] = 0.3333

        np.testing.assert_allclose(DOA5_W, expected)

    @patch("feature_extraction.preprocess_emg")
    @patch("feature_extraction.load_data")
    def test_feature_pipeline_can_emit_five_doa_targets(self, mock_load_data, mock_preprocess_emg) -> None:
        emg = np.arange(120, dtype=np.float32).reshape(20, 6)
        glove = np.arange(440, dtype=np.float32).reshape(20, 22)
        mock_load_data.return_value = {"emg": emg, "glove": glove}
        mock_preprocess_emg.side_effect = lambda value, fs: value

        pipeline = run_feature_pipeline(
            "synthetic.mat",
            target_mapping="doa5",
            fs=10,
            window_ms=400,
            stride_ms=200,
        )

        self.assertEqual(pipeline["target_source"], "doa5")
        self.assertEqual(pipeline["target_mapping"], "doa5")
        self.assertEqual(pipeline["target_names"], list(DOA5_NAMES))
        self.assertEqual(pipeline["feature_set"]["target_values"].shape[1], 5)

    # ── Rest-state threshold calibration tests ───────────────────────────────

    def test_stimulus_based_rest_threshold(self) -> None:
        rng = np.random.default_rng(42)
        n_samples = 1000
        n_channels = 4

        emg = np.zeros((n_samples, n_channels), dtype=np.float32)
        emg[:400] = rng.normal(0, 2e-6, (400, n_channels)).astype(np.float32)
        emg[400:800] = rng.normal(0, 5e-5, (400, n_channels)).astype(np.float32)
        emg[800:] = rng.normal(0, 1e-6, (200, n_channels)).astype(np.float32)

        stimulus = np.zeros(n_samples, dtype=np.float32)
        stimulus[400:800] = 1

        thresholds = _compute_rest_thresholds(emg, stimulus=stimulus, fs=2000.0)
        rest_emg = np.concatenate([emg[:400], emg[800:]], axis=0)
        expected = np.sqrt(np.mean(np.square(rest_emg), axis=0))

        self.assertEqual(thresholds.shape, (n_channels,))
        np.testing.assert_allclose(thresholds, expected, rtol=0.15)

    def test_percentile_fallback_when_no_stimulus(self) -> None:
        rng = np.random.default_rng(123)
        n_samples = 2000
        n_channels = 3

        emg = rng.normal(0, 1e-6, (n_samples, n_channels)).astype(np.float32)
        emg[500:700] = rng.normal(0, 1e-4, (200, n_channels)).astype(np.float32)
        emg[1200:1400] = rng.normal(0, 8e-5, (200, n_channels)).astype(np.float32)

        thresholds = _compute_rest_thresholds(emg, stimulus=None, fs=2000.0)

        self.assertEqual(thresholds.shape, (n_channels,))
        self.assertTrue(np.all(thresholds < 5e-6),
                        f"thresholds {thresholds} should be near noise floor, not active level")

    def test_restimulus_preferred_over_stimulus(self) -> None:
        n_samples = 200
        n_channels = 2
        emg = np.random.default_rng(7).normal(0, 3e-6, (n_samples, n_channels)).astype(np.float32)

        restimulus = np.ones(n_samples, dtype=np.float32)
        restimulus[50:150] = 0

        stimulus = np.zeros(n_samples, dtype=np.float32)

        threshold_from_restimulus = _compute_rest_thresholds(emg, stimulus=restimulus)
        threshold_from_stimulus = _compute_rest_thresholds(emg, stimulus=stimulus)

        self.assertFalse(
            np.allclose(threshold_from_restimulus, threshold_from_stimulus, rtol=0.01),
            "restimulus and stimulus should produce different thresholds when masks differ",
        )

    def test_zc_ssc_skipped_when_not_in_feature_order(self) -> None:
        windows = {
            "unrectified": np.random.default_rng(1).normal(0, 1e-6, (3, 10, 2)).astype(np.float32),
            "rectified": np.abs(np.random.default_rng(1).normal(0, 1e-6, (3, 10, 2)).astype(np.float32)),
            "window_start_indices": np.array([0, 1, 2], dtype=np.int32),
            "window_end_indices": np.array([10, 11, 12], dtype=np.int32),
            "window_center_indices": np.array([5, 6, 7], dtype=np.int32),
            "window_size": 10,
            "stride": 1,
            "window_ms": 200,
            "stride_ms": 50,
            "fs": 2000.0,
        }

        feature_set = extract_emg_features(windows, feature_order=("rms", "mav", "wl"))

        self.assertNotIn("zc", feature_set)
        self.assertNotIn("ssc", feature_set)
        self.assertNotIn("zc_thresholds", feature_set)
        self.assertNotIn("ssc_thresholds", feature_set)
        self.assertIn("mav", feature_set)
        self.assertIn("wl", feature_set)
        self.assertIn("rms", feature_set)


if __name__ == "__main__":
    unittest.main()
