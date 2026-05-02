import pathlib
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "src" / "deep learning"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from train import (
    RecordingFeatures,
    WeightedSmoothL1Loss,
    build_action_stratified_sequence_splits,
    build_blocked_sequence_splits,
    build_sequence_split,
    compute_grouped_regression_metrics,
    compute_regression_metrics,
    apply_target_normalizer,
    fit_target_normalizer,
    inverse_target_normalizer,
)
from run_doa5_subject_adaptation import (
    assert_selection_summary_is_uncontaminated,
    omit_split,
    parse_args as parse_doa5_args,
    run_doa5_subject_adaptation,
    select_validation_candidate,
    per_doa_pass,
)


def candidate_summary(
    candidate_id: str,
    hidden_units: int,
    loss_profile: str,
    r2_values: list[float],
    mae_mean: float,
) -> dict:
    r2 = np.asarray(r2_values, dtype=np.float32)
    return {
        "candidate_id": candidate_id,
        "candidate_config": {
            "hidden_units": hidden_units,
            "loss_profile": loss_profile,
        },
        "adapted_val_metrics": {
            "r2_by_target": r2,
            "r2_mean": float(np.mean(r2)),
            "mae_mean": mae_mean,
        },
        "selection_values": {
            "worst_doa_r2": float(np.min(r2)),
            "pass_count": int(np.sum(r2 >= 0.5)),
            "mean_r2": float(np.mean(r2)),
            "mean_mae": mae_mean,
        },
    }


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

    def test_compute_regression_metrics_keeps_five_targets_separate(self) -> None:
        y_true = np.arange(30, dtype=np.float32).reshape(6, 5)
        y_pred = y_true.copy()
        y_pred[:, 4] = 0.0

        metrics = compute_regression_metrics(y_true, y_pred)

        self.assertEqual(metrics["r2_by_target"].shape, (5,))
        self.assertTrue(np.all(metrics["r2_by_target"][:4] > 0.99))
        self.assertLess(metrics["r2_by_target"][4], 0.5)
        self.assertLess(metrics["r2_mean"], 1.0)

    def test_weighted_smooth_l1_applies_per_target_weights_before_reduction(self) -> None:
        loss_fn = WeightedSmoothL1Loss([1.0, 2.0, 3.0])
        pred = torch.tensor([[0.0, 2.0, 4.0]])
        target = torch.tensor([[0.0, 0.0, 0.0]])

        loss = loss_fn(pred, target)

        per_target = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
        expected = (per_target * torch.tensor([[1.0, 2.0, 3.0]])).mean()
        self.assertAlmostEqual(float(loss), float(expected))

    def test_compute_regression_metrics_reports_target_range_and_normalized_mae(self) -> None:
        y_true = np.array(
            [
                [0.0, 0.0],
                [10.0, 100.0],
            ],
            dtype=np.float32,
        )
        y_pred = y_true + np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        metrics = compute_regression_metrics(y_true, y_pred)

        np.testing.assert_allclose(metrics["mae_by_target"], np.array([1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(metrics["target_range_by_target"], np.array([10.0, 100.0], dtype=np.float32))
        np.testing.assert_allclose(metrics["normalized_mae_by_target"], np.array([0.1, 0.01], dtype=np.float32))
        self.assertAlmostEqual(metrics["normalized_mae_mean"], 0.055, places=6)

    def test_grouped_action_metrics_reports_per_action_metrics(self) -> None:
        y_true = np.array(
            [
                [0.0, 0.0],
                [2.0, 2.0],
                [10.0, 10.0],
                [12.0, 12.0],
            ],
            dtype=np.float32,
        )
        y_pred = y_true.copy()
        y_pred[2:, 1] += 2.0
        actions = np.array([1, 1, 2, 2], dtype=np.int16)

        grouped = compute_grouped_regression_metrics(y_true, y_pred, actions)

        self.assertEqual(set(grouped), {"1", "2"})
        np.testing.assert_allclose(grouped["1"]["mae_by_target"], np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(grouped["2"]["mae_by_target"], np.array([0.0, 2.0], dtype=np.float32))
        self.assertLess(grouped["2"]["r2_by_target"][1], 0.0)

    def test_per_doa_pass_does_not_use_mean_r2(self) -> None:
        metrics = {
            "r2_by_target": np.array([0.8, 0.7, 0.6, 0.55, 0.1], dtype=np.float32),
            "r2_mean": 0.55,
        }
        target_names = ["a", "b", "c", "d", "e"]

        result = per_doa_pass(metrics, target_names)

        self.assertEqual(result, {"a": True, "b": True, "c": True, "d": True, "e": False})

    def test_candidate_selection_ranks_by_worst_doa_before_mean_r2(self) -> None:
        weak_mean = candidate_summary("h64_baseline", 64, "baseline", [0.30, 0.80, 0.80, 0.80, 0.80], 10.0)
        balanced = candidate_summary("h64_balanced", 64, "baseline", [0.45, 0.55, 0.55, 0.55, 0.55], 12.0)

        selected = select_validation_candidate([weak_mean, balanced])

        self.assertEqual(selected["candidate_id"], "h64_balanced")

    def test_candidate_selection_requires_minimum_delta_for_128_units(self) -> None:
        baseline = candidate_summary("h64_baseline", 64, "baseline", [0.50, 0.50, 0.50, 0.50, 0.50], 10.0)
        too_small = candidate_summary("h128_baseline", 128, "baseline", [0.519, 0.52, 0.52, 0.52, 0.52], 9.0)

        selected = select_validation_candidate([baseline, too_small])

        self.assertEqual(selected["candidate_id"], "h64_baseline")

    def test_candidate_selection_allows_128_with_minimum_delta(self) -> None:
        baseline = candidate_summary("h64_baseline", 64, "baseline", [0.50, 0.50, 0.50, 0.50, 0.50], 10.0)
        improved = candidate_summary("h128_baseline", 128, "baseline", [0.52, 0.53, 0.53, 0.53, 0.53], 9.0)

        selected = select_validation_candidate([baseline, improved])

        self.assertEqual(selected["candidate_id"], "h128_baseline")

    def test_candidate_selection_requires_weighted_loss_delta(self) -> None:
        baseline = candidate_summary("h64_baseline", 64, "baseline", [0.60, 0.44, 0.60, 0.49, 0.60], 10.0)
        too_small = candidate_summary("h64_weighted", 64, "mild_failed_doa", [0.60, 0.445, 0.60, 0.50, 0.60], 9.0)

        selected = select_validation_candidate([baseline, too_small])

        self.assertEqual(selected["candidate_id"], "h64_baseline")

    def test_selection_summary_rejects_s1_test_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "adapted_test_metrics"):
            assert_selection_summary_is_uncontaminated(
                {
                    "contains_s1_test_metrics": False,
                    "candidates": [],
                    "adapted_test_metrics": {},
                }
            )

    def test_selection_summary_omits_test_action_coverage(self) -> None:
        summary = {
            "train": {"actions": [0]},
            "val": {"actions": [0]},
            "test": {"actions": [0]},
        }

        filtered = omit_split(summary, "test")

        self.assertEqual(set(filtered), {"train", "val"})

    def test_final_evaluation_requires_selection_artifact(self) -> None:
        with patch("sys.argv", ["run_doa5_subject_adaptation.py", "--device", "cpu"]):
            args = parse_doa5_args()

        with self.assertRaisesRegex(ValueError, "selection-artifact"):
            run_doa5_subject_adaptation(args)

    def test_final_evaluation_rejects_contaminated_selection_artifact(self) -> None:
        artifact = REPO_ROOT / "tests" / "_tmp_contaminated_selection.json"
        artifact.write_text(
            (
                "{"
                "\"contains_s1_test_metrics\": false,"
                "\"adapted_test_metrics\": {},"
                "\"selected_candidate\": {\"candidate_config\": {\"hidden_units\": 64, \"loss_profile\": \"baseline\"}},"
                "\"selected_candidate_id\": \"h64_baseline\""
                "}"
            ),
            encoding="utf-8",
        )
        try:
            with patch(
                "sys.argv",
                [
                    "run_doa5_subject_adaptation.py",
                    "--device",
                    "cpu",
                    "--selection-artifact",
                    str(artifact),
                ],
            ):
                args = parse_doa5_args()

            with self.assertRaisesRegex(ValueError, "adapted_test_metrics"):
                run_doa5_subject_adaptation(args)
        finally:
            if artifact.exists():
                artifact.unlink()

    def test_mu_law_target_normalizer_round_trips(self) -> None:
        targets = np.array([[-10.0], [0.0], [10.0], [25.0]], dtype=np.float32)

        stats = fit_target_normalizer(targets, method="mu_law", mu=255.0)
        normalized = apply_target_normalizer(targets, stats)
        recovered = inverse_target_normalizer(normalized, stats)

        self.assertEqual(stats["method"], "mu_law")
        self.assertLessEqual(float(np.max(np.abs(normalized))), 1.0 + 1e-6)
        np.testing.assert_allclose(recovered, targets, atol=1e-5)

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

    def test_action_stratified_split_puts_each_action_in_each_partition(self) -> None:
        n_windows = 90
        action_labels = np.repeat(np.array([1, 2, 3], dtype=np.int16), 30)
        repetition_labels = np.tile(np.repeat(np.arange(1, 7, dtype=np.int16), 5), 3)
        recording = RecordingFeatures(
            recording_id="demo.mat",
            x_windows=np.arange(n_windows * 2, dtype=np.float32).reshape(n_windows, 2),
            y_windows=np.arange(n_windows, dtype=np.float32).reshape(n_windows, 1),
            target_alignment_indices=np.arange(n_windows, dtype=np.int32) * 10,
            feature_names=["f1", "f2"],
            target_names=["angle_1"],
            fs=100.0,
            action_labels=action_labels,
            repetition_labels=repetition_labels,
        )

        splits = build_action_stratified_sequence_splits(
            [recording],
            seq_len=3,
            seq_stride=1,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            gap_windows=1,
        )

        for split in splits.values():
            self.assertIsNotNone(split.action_labels)
            np.testing.assert_array_equal(np.unique(split.action_labels), np.array([1, 2, 3], dtype=np.int16))

    def test_action_stratified_split_falls_back_when_actions_have_two_repetitions(self) -> None:
        n_windows = 80
        action_labels = np.repeat(np.array([1, 2], dtype=np.int16), 40)
        repetition_labels = np.tile(np.repeat(np.array([1, 2], dtype=np.int16), 20), 2)
        recording = RecordingFeatures(
            recording_id="two-rep.mat",
            x_windows=np.arange(n_windows * 2, dtype=np.float32).reshape(n_windows, 2),
            y_windows=np.arange(n_windows, dtype=np.float32).reshape(n_windows, 1),
            target_alignment_indices=np.arange(n_windows, dtype=np.int32) * 10,
            feature_names=["f1", "f2"],
            target_names=["angle_1"],
            fs=100.0,
            action_labels=action_labels,
            repetition_labels=repetition_labels,
        )

        splits = build_action_stratified_sequence_splits(
            [recording],
            seq_len=3,
            seq_stride=1,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            gap_windows=1,
        )

        for split in splits.values():
            self.assertIsNotNone(split.action_labels)
            np.testing.assert_array_equal(np.unique(split.action_labels), np.array([1, 2], dtype=np.int16))


if __name__ == "__main__":
    unittest.main()
