import copy
import inspect
import pathlib
import sys
import tempfile
import unittest
import warnings
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "src" / "deep learning"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import run_db2_paper_cfc_finetune as paper_protocol
import train as training_module
import feature_extraction as feature_module
from feature_extraction import (
    DEFAULT_MU_LAW_MU,
    apply_feature_normalizer,
    fit_feature_normalizer,
)
from train import (
    DenseCfCLinearRegressor,
    DomainDiscriminator,
    build_best_cfc_config,
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
from run_db2_paper_cfc_finetune import (
    freeze_for_linear_head,
    save_feature_normalization_stats,
)


class CfCTrainingHelpersTests(unittest.TestCase):

    def test_paper_protocol_uses_rms_and_shared_mu_defaults(self) -> None:
        with patch.object(sys, "argv", ["run_db2_paper_cfc_finetune.py"]):
            args = paper_protocol.parse_args()

        self.assertEqual(args.feature_order, "rms")
        self.assertEqual(args.emg_channels, "1,2,3,4,5,6,7,8,9,10,11,12")
        self.assertEqual(DEFAULT_MU_LAW_MU, 255.0)
        self.assertEqual(args.mu_law_mu, DEFAULT_MU_LAW_MU)
        self.assertEqual(build_best_cfc_config().mu_law_mu, DEFAULT_MU_LAW_MU)

    def test_parse_emg_channels_accepts_one_based_physical_channels(self) -> None:
        channels = paper_protocol.parse_emg_channels("1,2,3,4,5,6,7,8")

        self.assertEqual(channels, (1, 2, 3, 4, 5, 6, 7, 8))

    def test_parse_emg_channels_rejects_invalid_lists(self) -> None:
        invalid_values = ("", ",", "1,,2", "1,1", "0,1", "-1,2", "1,13", "one,2")

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    paper_protocol.parse_emg_channels(value)

    def test_emg_channels_are_selected_before_preprocessing(self) -> None:
        raw_emg = np.arange(40 * 12, dtype=np.float32).reshape(40, 12)
        synthetic_data = {
            "emg": raw_emg,
            "glove": np.zeros((40, 22), dtype=np.float32),
            "restimulus": np.ones(40, dtype=np.int16),
            "rerepetition": np.ones(40, dtype=np.int16),
        }
        config = build_best_cfc_config(
            db2_dir=pathlib.Path("synthetic-db2"),
            emg_channels=(1, 3, 8),
            feature_order=("rms",),
        )

        with (
            patch.object(paper_protocol, "list_db2_files", return_value=[pathlib.Path("S1_E1_A1.mat")]),
            patch.object(paper_protocol, "load_data", return_value=synthetic_data),
            patch.object(
                paper_protocol,
                "preprocess_emg",
                side_effect=RuntimeError("stop after channel-selection boundary"),
            ) as preprocess,
        ):
            with self.assertRaisesRegex(RuntimeError, "channel-selection boundary"):
                paper_protocol.load_subject_filtered_split(
                    "S1",
                    ("E1",),
                    config,
                    actions=(1,),
                )

        expected = raw_emg[:, [0, 2, 7]]  # physical channels 1, 3, 8 -> zero-based columns 0, 2, 7
        np.testing.assert_array_equal(preprocess.call_args.args[0], expected)

    def test_emg_channel_selection_rejects_recording_width_mismatch(self) -> None:
        raw_emg = np.zeros((20, 8), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "channel 9"):
            paper_protocol.select_emg_channels(
                raw_emg,
                (1, 9),
                recording_id="S1_E1_A1.mat",
            )

    def test_recording_feature_names_keep_physical_channel_numbers(self) -> None:
        raw_emg = np.arange(40 * 12, dtype=np.float32).reshape(40, 12)
        synthetic_data = {
            "emg": raw_emg,
            "glove": np.zeros((40, 22), dtype=np.float32),
            "restimulus": np.ones(40, dtype=np.int16),
            "rerepetition": np.ones(40, dtype=np.int16),
        }
        config = build_best_cfc_config(
            db2_dir=pathlib.Path("synthetic-db2"),
            emg_channels=(1, 3, 8),
            feature_order=("rms", "zc"),
            seq_len=1,
        )
        feature_set = {
            "feature_matrix": np.zeros((1, 6), dtype=np.float32),
            "target_values": np.zeros((1, 5), dtype=np.float32),
            "target_alignment_indices": np.array([20], dtype=np.int32),
            "channel_feature_names": [
                "ch1_rms", "ch1_zc", "ch2_rms", "ch2_zc", "ch3_rms", "ch3_zc",
            ],
            "target_names": list(paper_protocol.DOA5_NAMES),
            "fs": 2000.0,
            "n_windows": 1,
        }

        with (
            patch.object(paper_protocol, "list_db2_files", return_value=[pathlib.Path("S1_E1_A1.mat")]),
            patch.object(paper_protocol, "load_data", return_value=synthetic_data),
            patch.object(paper_protocol, "preprocess_emg", side_effect=lambda emg, fs: emg),
            patch.object(
                paper_protocol,
                "_compute_rest_thresholds",
                return_value=np.zeros(3, dtype=np.float32),
            ),
            patch.object(paper_protocol, "sliding_window", return_value={"synthetic": True}),
            patch.object(paper_protocol, "extract_emg_features", return_value=feature_set),
            patch.object(
                paper_protocol,
                "build_sequence_split",
                side_effect=lambda recordings, seq_len, seq_stride: recordings[0],
            ),
        ):
            recording = paper_protocol.load_subject_filtered_split(
                "S1",
                ("E1",),
                config,
                actions=(1,),
            )

        self.assertEqual(
            recording.feature_names,
            ["ch1_rms", "ch1_zc", "ch3_rms", "ch3_zc", "ch8_rms", "ch8_zc"],
        )

    def test_training_config_serializes_selected_emg_channels(self) -> None:
        config = build_best_cfc_config(emg_channels=(1, 2, 3, 4, 5, 6, 7, 8))

        self.assertEqual(config.emg_channels, (1, 2, 3, 4, 5, 6, 7, 8))
        self.assertEqual(asdict(config)["emg_channels"], (1, 2, 3, 4, 5, 6, 7, 8))

    def test_resume_input_dim_comes_from_checkpoint_channels_and_features(self) -> None:
        config = build_best_cfc_config(
            emg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
            feature_order=("rms", "zc"),
        )
        checkpoint_config = {
            "emg_channels": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature_order": ["rms", "zc"],
        }

        input_dim = paper_protocol.resolve_resume_input_dim(
            checkpoint_config,
            config,
            actual_input_dim=16,
        )

        self.assertEqual(input_dim, 16)  # 8 selected channels * 2 features

    def test_resume_rejects_different_channel_selection(self) -> None:
        config = build_best_cfc_config(
            emg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
            feature_order=("rms", "zc"),
        )
        checkpoint_config = {
            "emg_channels": [1, 2, 3, 4, 5, 6, 7, 9],
            "feature_order": ["rms", "zc"],
        }

        with self.assertRaisesRegex(ValueError, "EMG channels"):
            paper_protocol.resolve_resume_input_dim(
                checkpoint_config,
                config,
                actual_input_dim=16,
            )

    def test_legacy_resume_is_explicitly_all_twelve_channels(self) -> None:
        config = build_best_cfc_config(
            emg_channels=tuple(range(1, 13)),
            feature_order=("rms",),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            input_dim = paper_protocol.resolve_resume_input_dim(
                {"feature_order": ["rms"]},
                config,
                actual_input_dim=12,
            )

        self.assertEqual(input_dim, 12)
        self.assertTrue(any("legacy checkpoint" in str(item.message).lower() for item in caught))

    def test_legacy_resume_rejects_current_eight_channel_selection(self) -> None:
        config = build_best_cfc_config(
            emg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
            feature_order=("rms",),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "EMG channels"):
                paper_protocol.resolve_resume_input_dim(
                    {"feature_order": ["rms"]},
                    config,
                    actual_input_dim=8,
                )

    def test_train_module_is_library_only(self) -> None:
        self.assertFalse(hasattr(training_module, "main"))

    def test_paper_protocol_omits_obsolete_loading_helpers(self) -> None:
        self.assertFalse(hasattr(paper_protocol, "_resolve_actions"))
        self.assertFalse(hasattr(paper_protocol, "load_subject_exercise_split"))

    def test_paper_protocol_keeps_imports_at_module_level(self) -> None:
        source_lines = inspect.getsource(paper_protocol).splitlines()
        local_imports = [
            line
            for line in source_lines
            if line.startswith(("    import ", "    from "))
        ]
        self.assertEqual(local_imports, [])

    def test_paper_protocol_metadata_does_not_claim_grl_atl(self) -> None:
        source = inspect.getsource(paper_protocol)
        self.assertNotIn("ATL domain adaptation with GRL+DD", source)
        self.assertIn("GAN-style alternating DD and target-network training", source)

    # ── Sequence construction ────────────────────────────────────────────────

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

    # ── Regression metrics ───────────────────────────────────────────────────

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

    # ── Freeze / head-only fine-tune ─────────────────────────────────────────

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

    # ── Normalization ────────────────────────────────────────────────────────

    def test_feature_and_target_normalizers_share_array_implementation(self) -> None:
        self.assertIs(
            training_module._fit_array_normalizer,
            feature_module._fit_array_normalizer,
        )
        self.assertIs(
            training_module._apply_array_normalizer,
            feature_module._apply_array_normalizer,
        )

    def test_feature_and_target_normalizers_are_numerically_identical(self) -> None:
        values = np.array(
            [[-10.0, 2.0], [0.0, 4.0], [10.0, 8.0], [25.0, 16.0]],
            dtype=np.float32,
        )

        for method in ("zscore", "mu_law"):
            with self.subTest(method=method):
                feature_stats = fit_feature_normalizer(values, method=method, mu=255.0)
                target_stats = fit_target_normalizer(values, method=method, mu=255.0)
                self.assertEqual(feature_stats.keys(), target_stats.keys())
                for key in feature_stats:
                    if isinstance(feature_stats[key], np.ndarray):
                        np.testing.assert_allclose(feature_stats[key], target_stats[key])
                    else:
                        self.assertEqual(feature_stats[key], target_stats[key])
                np.testing.assert_allclose(
                    apply_feature_normalizer(values, feature_stats),
                    apply_target_normalizer(values, target_stats),
                )

    def test_zscore_target_normalizer_round_trips(self) -> None:
        targets = np.array([[-10.0], [0.0], [10.0], [25.0]], dtype=np.float32)

        stats = fit_target_normalizer(targets, method="zscore")
        normalized = apply_target_normalizer(targets, stats)
        recovered = inverse_target_normalizer(normalized, stats)

        np.testing.assert_allclose(recovered, targets, atol=1e-5)

    def test_feature_normalization_stats_are_saved_for_header_export(self) -> None:
        stats = {
            "method": "mu_law",
            "center": np.arange(1.0, 7.0, dtype=np.float32),
            "scale": np.arange(11.0, 17.0, dtype=np.float32),
            "mu": 255.0,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = save_feature_normalization_stats(
                pathlib.Path(tmp_dir),
                stats,
                emg_channels=(1, 3, 8),
                feature_order=("rms", "zc"),
            )
            with np.load(path) as saved:
                self.assertEqual(path.name, "feature_normalization.npz")
                np.testing.assert_allclose(saved["center"], stats["center"])
                np.testing.assert_allclose(saved["scale"], stats["scale"])
                self.assertEqual(float(saved["mu"]), stats["mu"])
                np.testing.assert_array_equal(saved["emg_channels"], np.array([1, 3, 8]))
                np.testing.assert_array_equal(saved["feature_order"], np.array(["rms", "zc"]))

    def test_mu_law_target_normalizer_round_trips(self) -> None:
        targets = np.array([[-10.0], [0.0], [10.0], [25.0]], dtype=np.float32)

        stats = fit_target_normalizer(targets, method="mu_law", mu=255.0)
        normalized = apply_target_normalizer(targets, stats)
        recovered = inverse_target_normalizer(normalized, stats)

        self.assertEqual(stats["method"], "mu_law")
        self.assertLessEqual(float(np.max(np.abs(normalized))), 1.0 + 1e-6)
        np.testing.assert_allclose(recovered, targets, atol=1e-5)

    # ── Data splits ──────────────────────────────────────────────────────────

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

    # ── DD / GAN ATL ────────────────────────────────────────────────────────

    def test_dd_construction(self) -> None:
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

        with torch.no_grad():
            y_seq, _ = model.cfc(x)
            expected_pre = y_seq[:, -1, :]
        torch.testing.assert_close(
            pre_dropout, expected_pre,
            msg="pre_dropout_state should equal the raw CfC final timestep output",
        )

    def test_atl_smoke(self) -> None:
        """GAN-style ATL: frozen source model + trainable target model + DD."""
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

        pretrained = DenseCfCLinearRegressor(input_dim, output_dim, hidden_units, dropout=0.1)

        # Multi-s-net: frozen
        source_model = copy.deepcopy(pretrained)
        for p in source_model.parameters():
            p.requires_grad = False
        source_model.eval()

        # New-t-net: trainable, warm-start
        target_model = copy.deepcopy(pretrained)
        target_model.train()

        dd = DomainDiscriminator(in_dim=hidden_units, hidden=128)
        dd.train()

        target_opt = torch.optim.AdamW(target_model.parameters(), lr=1e-4)
        dd_opt = torch.optim.AdamW(dd.parameters(), lr=1e-4)

        loss_fn = torch.nn.MSELoss()
        subject_weight = 1.0

        # Record initial weights to verify frozen source
        source_params_before = [p.clone().detach() for p in source_model.parameters()]

        # --- One batch of GAN training ---
        # Step 1: Train DD
        with torch.no_grad():
            _, F_s = source_model.forward_with_features(source_x)

        pred_t, F_t = target_model.forward_with_features(target_x)

        dd_opt.zero_grad()
        dd_src = dd(F_s)
        dd_tgt = dd(F_t)
        eps = 1e-8
        L_DD = -(torch.log(dd_src + eps).mean() + torch.log(1.0 - dd_tgt + eps).mean())
        L_DD.backward()
        dd_opt.step()

        # Step 2: Train New-t-net — fresh forward
        pred_t, F_t = target_model.forward_with_features(target_x)

        target_opt.zero_grad()
        L_mapping = -torch.log(dd(F_t) + eps).mean()
        L_subject = subject_weight * loss_fn(pred_t, target_y)
        (L_mapping + L_subject).backward()
        target_opt.step()

        # --- Assertions ---
        # Source model must be unchanged
        for before, after in zip(source_params_before, source_model.parameters()):
            torch.testing.assert_close(after, before,
                msg="Multi-s-net weights must not change (frozen)")

        # Losses must be finite
        self.assertTrue(torch.isfinite(L_DD).all(), msg="L_DD must be finite")
        self.assertTrue(torch.isfinite(L_mapping).all(), msg="L_mapping must be finite")
        self.assertTrue(torch.isfinite(L_subject).all(), msg="L_subject must be finite")

        # Target model must have gradients
        target_grads = [p.grad for p in target_model.parameters()
                        if p.requires_grad and p.grad is not None]
        self.assertGreater(len(target_grads), 0,
            msg="New-t-net must have non-None gradients")

        # DD must have gradients
        dd_grads = [p.grad for p in dd.parameters() if p.grad is not None]
        self.assertGreater(len(dd_grads), 0,
            msg="DD must have non-None gradients")


if __name__ == "__main__":
    unittest.main()
