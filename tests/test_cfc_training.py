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
    DenseCfCLinearRegressor,
    DomainDiscriminator,
    GradientReversalFunction,
    build_cfc_regressor,
    RecordingFeatures,
    SequenceSplit,
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
from run_db2_paper_cfc_finetune import freeze_for_linear_head
from run_doa5_subject_adaptation import (
    apply_adaptation_mode,
    assert_selection_summary_is_uncontaminated,
    limit_split_by_support_seconds,
    omit_split,
    parse_adaptation_modes,
    parse_args as parse_doa5_args,
    run_doa5_subject_adaptation,
    select_validation_candidate,
    per_doa_pass,
    validate_support_action_coverage,
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

    def test_parse_adaptation_modes_rejects_unknown_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported adaptation modes"):
            parse_adaptation_modes("autoncp_full,bad_mode")

    def test_motor_only_adaptation_freezes_non_motor_layers(self) -> None:
        class DummyWiredModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cfc = torch.nn.Module()
                self.cfc.rnn_cell = torch.nn.Module()
                self.cfc.rnn_cell.num_layers = 3
                self.cfc.rnn_cell.layer_0 = torch.nn.Linear(2, 3)
                self.cfc.rnn_cell.layer_1 = torch.nn.Linear(3, 4)
                self.cfc.rnn_cell.layer_2 = torch.nn.Linear(4, 5)
                self.cfc.rnn_cell.layer_2.sparsity_mask = torch.nn.Parameter(torch.ones(4, 5), requires_grad=False)

        model = DummyWiredModel()

        audit = apply_adaptation_mode(model, "autoncp_motor_only")

        self.assertTrue(audit["parameter_mask_valid"])
        self.assertEqual(audit["matched_trainable_prefixes"], ["cfc.rnn_cell.layer_2."])
        self.assertTrue(all(name.startswith("cfc.rnn_cell.layer_2.") for name in audit["trainable_param_names"]))
        self.assertFalse(any(name.endswith("sparsity_mask") for name in audit["trainable_param_names"]))
        for name, parameter in model.named_parameters():
            should_train = name.startswith("cfc.rnn_cell.layer_2.") and not name.endswith("sparsity_mask")
            self.assertEqual(parameter.requires_grad, should_train)

    def test_dense_cfc_linear_head_freeze_only_updates_head(self) -> None:
        model = build_cfc_regressor(
            input_dim=12,
            output_dim=10,
            hidden_units=32,
            model_family="dense_cfc_linear",
        )

        audit = freeze_for_linear_head(model)

        self.assertGreater(audit["trainable_param_count"], 0)
        self.assertTrue(all(name.startswith("head.") for name in audit["trainable_param_names"]))
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("head."))

    def test_support_seconds_budget_balances_actions_without_exceeding_total(self) -> None:
        labels = np.repeat(np.array([1, 2, 3], dtype=np.int16), 10)
        split = SequenceSplit(
            x=np.zeros((30, 2, 2), dtype=np.float32),
            y=np.zeros((30, 1), dtype=np.float32),
            time_s=np.arange(30, dtype=np.float32) * 0.1,
            alignment_indices=np.arange(30, dtype=np.int32),
            recording_ids=np.array(["demo"] * 30),
            feature_names=["a", "b"],
            target_names=["angle"],
            action_labels=labels,
        )

        limited, budget = limit_split_by_support_seconds(
            split,
            support_seconds_total=0.9,
            stride_ms=100,
            seq_stride=1,
        )

        self.assertLessEqual(budget["actual_total_seconds"], 0.9 + 1e-9)
        self.assertEqual(limited.x.shape[0], 9)
        self.assertEqual(budget["per_action_sequence_counts"], {"1": 3, "2": 3, "3": 3})

    def test_support_coverage_rejects_missing_evaluated_action(self) -> None:
        def make_split(labels: np.ndarray) -> SequenceSplit:
            return SequenceSplit(
                x=np.zeros((labels.size, 1, 1), dtype=np.float32),
                y=np.zeros((labels.size, 1), dtype=np.float32),
                time_s=np.arange(labels.size, dtype=np.float32),
                alignment_indices=np.arange(labels.size, dtype=np.int32),
                recording_ids=np.array(["demo"] * labels.size),
                feature_names=["a"],
                target_names=["angle"],
                action_labels=labels.astype(np.int16),
            )

        with self.assertRaisesRegex(ValueError, "zero coverage"):
            validate_support_action_coverage(
                {
                    "train": make_split(np.array([1, 1])),
                    "val": make_split(np.array([1, 2])),
                    "test": make_split(np.array([1])),
                }
            )

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

    # --- GRL / DD / ATL smoke tests -------------------------------------------------

    def test_grl_backwards_sign(self) -> None:
        """GradientReversalFunction negates gradients by -lambda."""
        x = torch.randn(4, 8, requires_grad=True)
        lambda_val = 0.5

        y = GradientReversalFunction.apply(x, lambda_val)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        torch.testing.assert_close(
            x.grad,
            -lambda_val * torch.ones_like(x),
            msg="GRL backward should multiply gradient by -lambda",
        )

    def test_grl_preserves_forward_output(self) -> None:
        """GradientReversalFunction forward is identity."""
        x = torch.randn(3, 5)
        lambda_val = 0.3
        y = GradientReversalFunction.apply(x, lambda_val)
        torch.testing.assert_close(y, x, msg="GRL forward must be identity")

    def test_dd_construction(self) -> None:
        """DomainDiscriminator outputs probabilities in [0, 1] with correct shape."""
        batch_size = 16
        in_dim = 128
        hidden = 128

        dd = DomainDiscriminator(in_dim=in_dim, hidden=hidden)
        x = torch.randn(batch_size, in_dim)
        out = dd(x)

        self.assertEqual(out.shape, (batch_size, 1))
        self.assertTrue(torch.all(out >= 0.0).item(), msg="All DD outputs should be >= 0")
        self.assertTrue(torch.all(out <= 1.0).item(), msg="All DD outputs should be <= 1")

    def test_forward_with_features(self) -> None:
        """forward_with_features() returns (prediction, pre_dropout_state) with correct shapes."""
        batch_size = 4
        seq_len = 8
        input_dim = 60
        output_dim = 5
        hidden_units = 32

        model = DenseCfCLinearRegressor(input_dim, output_dim, hidden_units, dropout=0.1)
        model.eval()
        x = torch.randn(batch_size, seq_len, input_dim)

        pred, pre_dropout = model.forward_with_features(x)

        self.assertEqual(pred.shape, (batch_size, output_dim))
        self.assertEqual(pre_dropout.shape, (batch_size, hidden_units))

        # Verify pre_dropout is the CfC output BEFORE dropout by comparing norms
        # Dropout zeroes some elements so ||pre_dropout|| >= ||post_dropout|| in expectation
        with torch.no_grad():
            y_seq, _ = model.cfc(x)
            expected_pre = y_seq[:, -1, :]
        torch.testing.assert_close(
            pre_dropout, expected_pre,
            msg="pre_dropout_state should equal the raw CfC final timestep output",
        )

    def test_atl_smoke(self) -> None:
        """One ATL training iteration on synthetic data completes without error."""
        n_source = 10
        n_target = 5
        input_dim = 60
        output_dim = 5
        hidden_units = 32
        seq_len = 8

        source_x = torch.randn(n_source, seq_len, input_dim)
        source_y = torch.randn(n_source, output_dim)
        target_x = torch.randn(n_target, seq_len, input_dim)
        target_y = torch.randn(n_target, output_dim)

        model = DenseCfCLinearRegressor(input_dim, output_dim, hidden_units, dropout=0.1)
        model.train()

        dd = DomainDiscriminator(in_dim=hidden_units, hidden=128)

        cfc_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        dd_optimizer = torch.optim.AdamW(dd.parameters(), lr=1e-3)

        loss_fn = torch.nn.MSELoss()
        bce_fn = torch.nn.BCELoss()

        lambda_val = 0.5

        # Single ATL iteration following the same logic as _atl_training_epoch
        combined_x = torch.cat([source_x, target_x], dim=0)
        combined_y = torch.cat([source_y, target_y], dim=0)

        cfc_optimizer.zero_grad(set_to_none=True)
        dd_optimizer.zero_grad(set_to_none=True)

        pred, features = model.forward_with_features(combined_x)

        mse_loss = loss_fn(pred, combined_y)

        n_src = source_x.shape[0]
        src_feat = features[:n_src]
        tgt_feat = features[n_src:]

        src_rev = GradientReversalFunction.apply(src_feat, lambda_val)
        tgt_rev = GradientReversalFunction.apply(tgt_feat, lambda_val)

        src_domain_pred = dd(src_rev)
        tgt_domain_pred = dd(tgt_rev)

        src_domain_loss = bce_fn(src_domain_pred, torch.zeros(n_src, 1))
        tgt_domain_loss = bce_fn(tgt_domain_pred, torch.ones(n_target, 1))
        domain_loss = src_domain_loss + tgt_domain_loss

        total_loss = mse_loss + lambda_val * domain_loss
        total_loss.backward()

        cfc_optimizer.step()
        dd_optimizer.step()

        # Verify all losses are finite
        self.assertTrue(torch.isfinite(mse_loss).all(), msg="MSE loss should be finite")
        self.assertTrue(torch.isfinite(domain_loss).all(), msg="Domain loss should be finite")
        self.assertTrue(torch.isfinite(total_loss).all(), msg="Total loss should be finite")

        # Verify gradients flowed to model parameters
        model_grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertGreater(len(model_grads), 0, msg="CfC model should have non-None gradients")
        self.assertTrue(
            any(torch.isfinite(g).all() and torch.any(g != 0.0) for g in model_grads),
            msg="At least one CfC parameter should have non-zero finite gradient",
        )

        # Verify gradients flowed to DD parameters
        dd_grads = [p.grad for p in dd.parameters() if p.grad is not None]
        self.assertGreater(len(dd_grads), 0, msg="DD should have non-None gradients")
        self.assertTrue(
            any(torch.isfinite(g).all() and torch.any(g != 0.0) for g in dd_grads),
            msg="At least one DD parameter should have non-zero finite gradient",
        )

        # Verify DD outputs are valid probabilities
        with torch.no_grad():
            model.eval()
            dd.eval()
            _, test_feat = model.forward_with_features(target_x)
            test_dd_out = dd(GradientReversalFunction.apply(test_feat, lambda_val))
        self.assertTrue(torch.all(test_dd_out >= 0.0).item())
        self.assertTrue(torch.all(test_dd_out <= 1.0).item())


if __name__ == "__main__":
    unittest.main()
