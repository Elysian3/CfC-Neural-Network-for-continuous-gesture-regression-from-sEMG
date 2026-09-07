import copy
import inspect
import pathlib
import sys
import tempfile
import unittest
import warnings
from contextlib import ExitStack
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "src" / "deep learning"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
DATAFLOW_DIR = REPO_ROOT / "src" / "dataflow"
if str(DATAFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(DATAFLOW_DIR))

import feature_extraction as feature_module
import run_db2_paper_cfc_finetune as paper_protocol
import train as training_module
from feature_extraction import (
    DEFAULT_MU_LAW_MU,
    apply_feature_normalizer,
    fit_feature_normalizer,
)
from run_db2_paper_cfc_finetune import (
    freeze_for_linear_head,
    save_feature_normalization_stats,
)
from train import (
    CfCTrainingConfig,
    DenseCfCLinearRegressor,
    DomainDiscriminator,
    GpuResidentStatefulBatches,
    RecordingFeatures,
    SegmentIndex,
    SequenceSplit,
    WeightedSmoothL1Loss,
    apply_target_normalizer,
    build_action_stratified_sequence_splits,
    build_best_cfc_config,
    build_blocked_sequence_splits,
    build_cfc_regressor,
    build_sequence_split,
    compute_grouped_regression_metrics,
    compute_regression_metrics,
    evaluate_chain,
    fit_target_normalizer,
    inverse_target_normalizer,
)


class CfCTrainingHelpersTests(unittest.TestCase):

    def _small_sequence_split(self) -> SequenceSplit:
        return SequenceSplit(
            x=np.zeros((2, 3, 4), dtype=np.float32),
            y=np.zeros((2, 1), dtype=np.float32),
            time_s=np.array([0.0, 0.05], dtype=np.float32),
            alignment_indices=np.array([0, 1], dtype=np.int64),
            recording_ids=np.array(["synthetic.mat", "synthetic.mat"]),
            feature_names=["ch1_rms", "ch2_rms", "ch3_rms", "ch4_rms"],
            target_names=["target"],
        )

    def test_paper_protocol_uses_rms_and_shared_mu_defaults(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run_db2_paper_cfc_finetune.py", "--target-offset-samples", "0"],
        ):
            args = paper_protocol.parse_args()

        self.assertEqual(args.feature_order, "rms")
        self.assertEqual(args.emg_channels, "1,2,3,4,5,6,7,8,9,10,11,12")
        self.assertEqual(args.target_mapping, "joint_angles10")
        self.assertEqual(args.target_offset_samples, 0)
        self.assertEqual(args.hidden_units, 256)
        self.assertEqual(DEFAULT_MU_LAW_MU, 255.0)
        self.assertEqual(args.mu_law_mu, DEFAULT_MU_LAW_MU)
        self.assertEqual(build_best_cfc_config().mu_law_mu, DEFAULT_MU_LAW_MU)

    def test_parse_emg_channels_accepts_one_based_physical_channels(self) -> None:
        channels = paper_protocol.parse_emg_channels("1,2,3,4,5,6,7,8")

        self.assertEqual(channels, (1, 2, 3, 4, 5, 6, 7, 8))

    def test_parse_emg_channels_rejects_invalid_lists(self) -> None:
        invalid_values = ("", ",", "1,,2", "1,1", "0,1", "-1,2", "1,13", "one,2")

        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                paper_protocol.parse_emg_channels(value)

    def test_checkpoint_output_dim_is_inferred_from_head_weights(self) -> None:
        state_dict = DenseCfCLinearRegressor(4, 10, 8, dropout=0.0).state_dict()

        self.assertEqual(paper_protocol.infer_checkpoint_output_dim(state_dict), 10)

    def test_checkpoint_output_dim_requires_a_valid_head_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "head.weight"):
            paper_protocol.infer_checkpoint_output_dim({})

    def test_checkpoint_target_contract_accepts_legacy_gap_and_rejects_semantic_mismatch(self) -> None:
        current = paper_protocol.resolve_target_contract("joint_angles10")

        paper_protocol.validate_checkpoint_target_contract(None, current)
        paper_protocol.validate_checkpoint_target_contract(copy.deepcopy(current), current)
        mismatched = copy.deepcopy(current)
        mismatched["source_column_indices_zero_based"] = list(range(10))
        with self.assertRaisesRegex(ValueError, "source_column_indices_zero_based"):
            paper_protocol.validate_checkpoint_target_contract(mismatched, current)

    def test_resume_integrity_binds_checkpoint_stats_to_source_split_and_protocol(self) -> None:
        split = self._small_sequence_split()
        x_stats = {
            "method": "mu_law",
            "center": np.zeros(4, dtype=np.float32),
            "scale": np.ones(4, dtype=np.float32),
            "mu": 255.0,
        }
        y_stats = {
            "method": "mu_law",
            "center": np.zeros(1, dtype=np.float32),
            "scale": np.ones(1, dtype=np.float32),
            "mu": 255.0,
        }
        kwargs = {
            "x_stats": x_stats,
            "y_stats": y_stats,
            "source_supervised": split,
            "source_stream": split,
            "source_stream_score_mask": np.asarray([True, True]),
            "protocol": {"actions": [1], "random_seed": 42},
        }
        metadata = paper_protocol.build_resume_integrity_metadata(**kwargs)
        checkpoint = {
            "feature_normalization_stats": x_stats,
            "target_normalization_stats": y_stats,
            "resume_integrity": metadata,
        }

        resumed_x_stats, resumed_y_stats = paper_protocol.validate_resume_integrity_metadata(
            checkpoint,
            metadata,
            allow_legacy_resume_without_integrity_binding=False,
        )

        self.assertIs(resumed_x_stats, x_stats)
        self.assertIs(resumed_y_stats, y_stats)

        changed_split = copy.deepcopy(split)
        changed_split.x[0, 0, 0] = 1.0
        changed_metadata = paper_protocol.build_resume_integrity_metadata(
            **{**kwargs, "source_supervised": changed_split}
        )
        with self.assertRaisesRegex(ValueError, "source_split_identities"):
            paper_protocol.validate_resume_integrity_metadata(
                checkpoint,
                changed_metadata,
                allow_legacy_resume_without_integrity_binding=False,
            )

        changed_protocol_metadata = paper_protocol.build_resume_integrity_metadata(
            **{**kwargs, "protocol": {"actions": [2], "random_seed": 42}}
        )
        with self.assertRaisesRegex(ValueError, "protocol_identity"):
            paper_protocol.validate_resume_integrity_metadata(
                checkpoint,
                changed_protocol_metadata,
                allow_legacy_resume_without_integrity_binding=False,
            )

        checkpoint["feature_normalization_stats"] = {
            **x_stats,
            "center": np.ones(4, dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "normalization statistics"):
            paper_protocol.validate_resume_integrity_metadata(
                checkpoint,
                metadata,
                allow_legacy_resume_without_integrity_binding=False,
            )

    def test_resumed_legacy_history_keeps_only_epoch_and_training_loss(self) -> None:
        history = [
            {"epoch": 1.0, "train_loss": 0.3, "val_mae": 20.0, "early_stopped": 1.0},
            {"epoch": 2.0, "train_loss": 0.2, "val_chain_mae": 19.0},
            {"epoch": 3.0, "val_mae": 18.0},
        ]

        projected = paper_protocol.project_pretrain_history(history)

        self.assertEqual(
            projected,
            [{"epoch": 1.0, "train_loss": 0.3}, {"epoch": 2.0, "train_loss": 0.2}],
        )
        self.assertEqual(paper_protocol.project_pretrain_history(None), [])

    def test_resume_architecture_rejects_hidden_width_mismatch(self) -> None:
        current = SimpleNamespace(
            hidden_units=256,
            model_family="dense_cfc_linear",
            cfc_dropout=0.1,
        )
        checkpoint = {
            "hidden_units": 128,
            "model_family": "dense_cfc_linear",
            "cfc_dropout": 0.1,
        }

        with self.assertRaisesRegex(ValueError, "--hidden-units"):
            paper_protocol.validate_resume_architecture(checkpoint, current)

        paper_protocol.validate_resume_architecture({**checkpoint, "hidden_units": 256}, current)

    def test_legacy_pretrain_provenance_marks_unknown_execution_settings(self) -> None:
        with self.assertWarnsRegex(UserWarning, "lacks pretrain_provenance"):
            provenance = paper_protocol.build_legacy_pretrain_provenance(
                {"hidden_units": 256, "model_family": "dense_cfc_linear", "cfc_dropout": 0.1}
            )

        self.assertEqual(provenance["status"], "legacy_unknown")
        self.assertIsNone(provenance["augment_prob_requested"])
        self.assertIsNone(provenance["augment_prob_effective"])
        self.assertIsNone(provenance["gpu_resident_effective"])

    def test_adaptation_provenance_excludes_query_labels_from_stream_identity(self) -> None:
        stream = self._small_sequence_split()
        support = paper_protocol.copy_split_by_indices(stream, np.asarray([0]))
        provenance = paper_protocol.build_adaptation_provenance(
            target_support_split=support,
            target_stream_split=stream,
            support_score_mask=np.asarray([True, False]),
            mode="atl_stateful",
            epochs=10,
            learning_rate=1e-4,
            atl_subject_weight=1.0,
        )

        self.assertEqual(provenance["mode"], "atl_stateful")
        self.assertEqual(provenance["epochs"], 10)
        self.assertEqual(provenance["learning_rate"], 1e-4)
        self.assertEqual(provenance["atl_subject_weight"], 1.0)
        self.assertIn("target_support_split_identity", provenance)
        self.assertIn("target_stream_identity", provenance)
        self.assertIn("support_score_mask", provenance)

        stream.y[1, 0] += 1.0
        query_label_changed = paper_protocol.build_adaptation_provenance(
            target_support_split=support,
            target_stream_split=stream,
            support_score_mask=np.asarray([True, False]),
            mode="atl_stateful",
            epochs=10,
            learning_rate=1e-4,
            atl_subject_weight=1.0,
        )
        self.assertEqual(
            provenance["target_stream_identity"],
            query_label_changed["target_stream_identity"],
        )

        support.y[0, 0] += 1.0
        support_label_changed = paper_protocol.build_adaptation_provenance(
            target_support_split=support,
            target_stream_split=stream,
            support_score_mask=np.asarray([True, False]),
            mode="atl_stateful",
            epochs=10,
            learning_rate=1e-4,
            atl_subject_weight=1.0,
        )
        self.assertNotEqual(
            provenance["target_support_split_identity"],
            support_label_changed["target_support_split_identity"],
        )

    def test_source_supervision_metadata_records_all_repetitions_without_train_test_labels(self) -> None:
        split = SequenceSplit(
            x=np.zeros((4, 1, 1), dtype=np.float32),
            y=np.zeros((4, 1), dtype=np.float32),
            time_s=np.arange(4, dtype=np.float32),
            alignment_indices=np.arange(4, dtype=np.int64),
            recording_ids=np.asarray(["r"] * 4),
            feature_names=["f"],
            target_names=["t"],
            action_labels=np.asarray([1, 1, 2, 2], dtype=np.int16),
            repetition_labels=np.asarray([1, 2, 1, 2], dtype=np.int16),
        )

        metadata = paper_protocol.describe_all_selected_repetitions(split, actions=(1, 2))

        self.assertEqual(metadata["supervision"], "all_selected_repetitions")
        self.assertEqual(metadata["selected_repetitions_by_action"], {"1": [1, 2], "2": [1, 2]})
        self.assertEqual(metadata["n_selected_sequences"], 4)

    def test_resume_integrity_rejects_legacy_checkpoint_without_named_opt_in(self) -> None:
        stats = {
            "method": "mu_law",
            "center": np.zeros(1, dtype=np.float32),
            "scale": np.ones(1, dtype=np.float32),
            "mu": 255.0,
        }
        checkpoint = {
            "feature_normalization_stats": stats,
            "target_normalization_stats": stats,
        }

        with self.assertRaisesRegex(ValueError, "allow-legacy-resume-without-integrity-binding"):
            paper_protocol.validate_resume_integrity_metadata(
                checkpoint,
                {},
                allow_legacy_resume_without_integrity_binding=False,
            )
        with self.assertWarnsRegex(UserWarning, "allow-legacy-resume-without-integrity-binding"):
            paper_protocol.validate_resume_integrity_metadata(
                checkpoint,
                {},
                allow_legacy_resume_without_integrity_binding=True,
            )

    def test_legacy_resume_escape_hatch_defaults_to_disabled(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run_db2_paper_cfc_finetune.py", "--target-offset-samples", "0"],
        ):
            args = paper_protocol.parse_args()
        self.assertFalse(args.allow_legacy_resume_without_integrity_binding)

    def test_paper_protocol_requires_explicit_target_offset(self) -> None:
        with (
            patch.object(sys, "argv", ["run_db2_paper_cfc_finetune.py"]),
            self.assertRaises(SystemExit),
        ):
            paper_protocol.parse_args()

    def test_subject_loader_keeps_a_full_replay_stream_separate_from_training_segments(self) -> None:
        n_samples = 20
        emg = np.arange(1, n_samples + 1, dtype=np.float32)[:, None]
        glove = np.arange(n_samples, dtype=np.float32)[:, None]
        stimulus = np.asarray([1] * 10 + [2] * 10, dtype=np.int16)
        repetition = np.ones(n_samples, dtype=np.int16)
        config = SimpleNamespace(
            db2_dir=pathlib.Path("unused"),
            target_columns=(0,),
            target_mapping=None,
            emg_channels=(1,),
            window_ms=2.0,
            stride_ms=1.0,
            target_offset_samples=0,
            feature_order=("rms",),
            seq_len=2,
            seq_stride=1,
        )
        file_path = pathlib.Path("S1_E1_A1.mat")

        with (
            patch.object(paper_protocol, "list_db2_files", return_value=[file_path]),
            patch.object(
                paper_protocol,
                "load_data",
                return_value={
                    "emg": emg,
                    "glove": glove,
                    "restimulus": stimulus,
                    "rerepetition": repetition,
                },
            ),
            patch.object(paper_protocol, "select_emg_channels", side_effect=lambda values, *_args, **_kwargs: values),
            patch.object(paper_protocol, "preprocess_emg", side_effect=lambda values, **_kwargs: values),
            patch.object(paper_protocol, "_compute_rest_thresholds", return_value=np.zeros(1, dtype=np.float32)),
        ):
            selected_split, stream_split = paper_protocol.load_subject_splits(
                "S1",
                ("E1",),
                config,
                actions=(1, 2),
            )

        self.assertEqual(np.unique(stream_split.recording_ids).tolist(), ["S1_E1_A1.mat"])
        self.assertTrue(np.all(np.diff(stream_split.alignment_indices) > 0))
        self.assertGreater(np.count_nonzero(np.diff(stream_split.action_labels)), 0)
        self.assertEqual(SegmentIndex(stream_split).segments, [(0, stream_split.x.shape[0])])
        self.assertEqual(len(SegmentIndex(selected_split)), 2)
        self.assertEqual(selected_split.x.shape[0], 6)
        score_mask = paper_protocol.exact_query_score_mask(stream_split, selected_split)
        self.assertEqual(int(score_mask.sum()), selected_split.x.shape[0])

    def test_selected_windows_reject_label_excursions_before_delayed_target_anchor(self) -> None:
        n_samples = 16
        emg = np.arange(1, n_samples + 1, dtype=np.float32)[:, None]
        glove = np.arange(n_samples, dtype=np.float32)[:, None]
        stimulus = np.ones(n_samples, dtype=np.int16)
        stimulus[4:6] = 2
        repetition = np.ones(n_samples, dtype=np.int16)
        config = SimpleNamespace(
            db2_dir=pathlib.Path("unused"),
            target_columns=(0,),
            target_mapping=None,
            emg_channels=(1,),
            window_ms=2.0,
            stride_ms=1.0,
            target_offset_samples=4,
            feature_order=("rms",),
            seq_len=1,
            seq_stride=1,
        )
        file_path = pathlib.Path("S1_E1_A1.mat")

        with (
            patch.object(paper_protocol, "list_db2_files", return_value=[file_path]),
            patch.object(
                paper_protocol,
                "load_data",
                return_value={
                    "emg": emg,
                    "glove": glove,
                    "restimulus": stimulus,
                    "rerepetition": repetition,
                },
            ),
            patch.object(paper_protocol, "select_emg_channels", side_effect=lambda values, *_args, **_kwargs: values),
            patch.object(paper_protocol, "preprocess_emg", side_effect=lambda values, **_kwargs: values),
            patch.object(paper_protocol, "_compute_rest_thresholds", return_value=np.zeros(1, dtype=np.float32)),
        ):
            selected_split, _stream_split = paper_protocol.load_subject_splits(
                "S1",
                ("E1",),
                config,
                actions=(1,),
            )

        self.assertNotIn(7, selected_split.alignment_indices.tolist())

    def test_copy_split_marks_rows_selected_across_a_gap_as_a_new_stream(self) -> None:
        split = SequenceSplit(
            x=np.arange(4, dtype=np.float32).reshape(4, 1, 1),
            y=np.zeros((4, 1), dtype=np.float32),
            time_s=np.arange(4, dtype=np.float32) * 0.05,
            alignment_indices=np.asarray([100, 200, 300, 400], dtype=np.int64),
            recording_ids=np.asarray(["f.mat"] * 4),
            feature_names=["f"],
            target_names=["t"],
            stream_start_flags=np.asarray([True, False, False, False]),
            expected_alignment_step=100,
        )

        selected = paper_protocol.copy_split_by_indices(split, np.asarray([0, 2, 3]))

        np.testing.assert_array_equal(selected.stream_start_flags, [True, True, False])
        self.assertEqual(SegmentIndex(selected).segments, [(0, 1), (1, 3)])

    def test_exact_query_score_mask_uses_row_provenance_not_labels_alone(self) -> None:
        stream_split = SequenceSplit(
            x=np.zeros((3, 1, 1), dtype=np.float32),
            y=np.zeros((3, 1), dtype=np.float32),
            time_s=np.arange(3, dtype=np.float32),
            alignment_indices=np.asarray([100, 200, 300], dtype=np.int64),
            recording_ids=np.asarray(["f.mat"] * 3),
            feature_names=["f"],
            target_names=["t"],
            action_labels=np.asarray([1, 1, 1], dtype=np.int16),
            repetition_labels=np.asarray([2, 2, 2], dtype=np.int16),
            source_recording_ids=np.asarray(["f.mat"] * 3),
        )
        query_split = SequenceSplit(
            x=np.zeros((2, 1, 1), dtype=np.float32),
            y=np.zeros((2, 1), dtype=np.float32),
            time_s=np.asarray([0.0, 2.0], dtype=np.float32),
            alignment_indices=np.asarray([100, 300], dtype=np.int64),
            recording_ids=np.asarray(["f.mat:A1:R2:W0"] * 2),
            feature_names=["f"],
            target_names=["t"],
            action_labels=np.asarray([1, 1], dtype=np.int16),
            repetition_labels=np.asarray([2, 2], dtype=np.int16),
            source_recording_ids=np.asarray(["f.mat"] * 2),
        )

        mask = paper_protocol.exact_query_score_mask(stream_split, query_split)

        np.testing.assert_array_equal(mask, [True, False, True])

    def test_full_source_supervision_mask_includes_former_train_and_val_rows(self) -> None:
        stream_split = SequenceSplit(
            x=np.zeros((4, 1, 1), dtype=np.float32),
            y=np.zeros((4, 1), dtype=np.float32),
            time_s=np.arange(4, dtype=np.float32),
            alignment_indices=np.arange(4, dtype=np.int64),
            recording_ids=np.asarray(["r"] * 4),
            feature_names=["f"],
            target_names=["t"],
            source_recording_ids=np.asarray(["f.mat"] * 4),
        )
        former_train = paper_protocol.copy_split_by_indices(stream_split, np.asarray([0, 2]))
        former_val = paper_protocol.copy_split_by_indices(stream_split, np.asarray([1, 3]))
        all_source_labels = paper_protocol.concat_splits([former_train, former_val])

        mask = paper_protocol.provenance_score_mask(stream_split, all_source_labels)

        np.testing.assert_array_equal(mask, [True, True, True, True])

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
            self.assertRaisesRegex(RuntimeError, "channel-selection boundary"),
        ):
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
            recording = paper_protocol.load_subject_stream_split(
                "S1",
                ("E1",),
                config,
            )

        self.assertEqual(
            recording.feature_names,
            ["ch1_rms", "ch1_zc", "ch3_rms", "ch3_zc", "ch8_rms", "ch8_zc"],
        )

    def test_training_config_serializes_selected_emg_channels(self) -> None:
        config = build_best_cfc_config(emg_channels=(1, 2, 3, 4, 5, 6, 7, 8))

        self.assertEqual(config.emg_channels, (1, 2, 3, 4, 5, 6, 7, 8))
        self.assertEqual(asdict(config)["emg_channels"], (1, 2, 3, 4, 5, 6, 7, 8))

    def test_train_loader_pins_host_memory_when_cuda_is_selected(self) -> None:
        config = build_best_cfc_config(device="cuda")

        with patch.object(training_module, "resolve_device", return_value=torch.device("cuda")):
            loader = training_module.make_train_loader(self._small_sequence_split(), config)

        self.assertTrue(loader.pin_memory)  # CUDA async copies require pinned host tensors.

    def test_train_loader_leaves_host_memory_unpinned_for_cpu(self) -> None:
        loader = training_module.make_train_loader(
            self._small_sequence_split(),
            build_best_cfc_config(device="cpu"),
        )

        self.assertFalse(loader.pin_memory)  # Pinning helps CUDA transfers only and costs CPU resources.

    def test_train_one_epoch_returns_sample_weighted_loss(self) -> None:
        split = SequenceSplit(
            x=np.array([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32),
            y=np.zeros((3, 1, 1), dtype=np.float32),
            time_s=np.array([0.0, 0.05, 0.10], dtype=np.float32),
            alignment_indices=np.array([0, 1, 2], dtype=np.int64),
            recording_ids=np.array(["synthetic.mat"] * 3),
            feature_names=["ch1_rms"],
            target_names=["target"],
        )
        loader = training_module.make_train_loader(
            split,
            build_best_cfc_config(device="cpu", batch_size=2),
        )
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        epoch_loss = training_module.train_one_epoch(
            model,
            loader,
            optimizer,
            torch.nn.MSELoss(),
            device=torch.device("cpu"),
            gradient_clip_norm=None,
        )

        self.assertAlmostEqual(epoch_loss, 14.0 / 3.0, places=6)  # (1^2 + 2^2 + 3^2) / 3 samples

    def test_predict_sequences_preserves_batch_order(self) -> None:
        sequences = np.arange(12, dtype=np.float32).reshape(3, 2, 2)

        predicted = training_module.predict_sequences(
            torch.nn.Identity(),
            sequences,
            batch_size=2,
            device=torch.device("cpu"),
        )

        np.testing.assert_array_equal(predicted, sequences)

    def test_per_sequence_roll_matches_hand_computed_cyclic_indices(self) -> None:
        sequences = torch.tensor(
            [[[0.0], [1.0], [2.0]], [[10.0], [11.0], [12.0]]],
        )
        shifts = torch.tensor([1, -1])

        rolled = paper_protocol._roll_sequences_by_shift(sequences, shifts)

        expected = torch.tensor(
            [[[2.0], [0.0], [1.0]], [[11.0], [12.0], [10.0]]],
        )  # A positive shift moves the final timestep to index zero for that sequence only.
        torch.testing.assert_close(rolled, expected)

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
        self.assertIn(
            "causal stateful TBPTT with alternating DD and target-network training",
            source,
        )

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
        self.assertEqual(split.expected_alignment_step, 100)

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

    def test_target_normalization_stats_are_saved_for_firmware_inverse(self) -> None:
        feature_stats = {
            "method": "mu_law",
            "center": np.zeros(2, dtype=np.float32),
            "scale": np.ones(2, dtype=np.float32),
            "mu": 255.0,
        }
        target_stats = {
            "method": "mu_law",
            "center": np.asarray([10.0, 20.0], dtype=np.float32),
            "scale": np.asarray([30.0, 40.0], dtype=np.float32),
            "mu": 255.0,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = save_feature_normalization_stats(
                pathlib.Path(tmp_dir),
                feature_stats,
                emg_channels=(1, 2),
                feature_order=("rms",),
                target_stats=target_stats,
                target_names=("joint_a", "joint_b"),
            )
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["target_center"], [10.0, 20.0])
                np.testing.assert_array_equal(saved["target_scale"], [30.0, 40.0])
                self.assertEqual(float(saved["target_mu"]), 255.0)
                np.testing.assert_array_equal(saved["target_names"], ["joint_a", "joint_b"])

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


class RunProtocolOrchestrationTests(unittest.TestCase):
    """Synthetic contract tests for the no-validation paper orchestrator."""

    @staticmethod
    def _split(subject: str, rows: int = 4) -> SequenceSplit:
        alignment = np.arange(rows, dtype=np.int64)
        return SequenceSplit(
            x=np.arange(rows, dtype=np.float32).reshape(rows, 1, 1),
            y=np.arange(rows, dtype=np.float32).reshape(rows, 1),
            time_s=alignment.astype(np.float32),
            alignment_indices=alignment,
            recording_ids=np.asarray([f"{subject}.mat"] * rows),
            feature_names=["rms"],
            target_names=["target"],
            action_labels=np.ones(rows, dtype=np.int16),
            repetition_labels=np.arange(1, rows + 1, dtype=np.int16),
            stream_start_flags=np.asarray([True] + [False] * (rows - 1)),
            source_recording_ids=np.asarray([f"{subject}.mat"] * rows),
        )

    @staticmethod
    def _args(output_dir: pathlib.Path, *, enable_atl: bool) -> SimpleNamespace:
        return SimpleNamespace(
            output_dir=output_dir,
            db2_dir=output_dir / "DB2",
            subjects="S1,S2",
            exercise="E1",
            actions="1",
            target_subject="S1",
            target_mapping="joint_angles10",
            glove_columns="1",
            emg_channels="1",
            feature_order="rms",
            window_ms=200.0,
            stride_ms=50.0,
            target_offset_samples=0,
            mu_law_mu=255.0,
            seq_len=1,
            hidden_units=8,
            model_family="dense_cfc_linear",
            batch_size=2,
            learning_rate=1e-4,
            weight_decay=1e-4,
            cfc_dropout=0.1,
            max_epochs=2,
            random_seed=42,
            device="cpu",
            cuda_graph=False,
            gpu_resident=False,
            stateful=True,
            train_repetitions_per_action=2,
            resume_pretrain=None,
            allow_legacy_resume_without_integrity_binding=False,
            augment_prob=1.0,
            skip_fine_tune=False,
            enable_atl=enable_atl,
            fine_tune_learning_rate=3e-4,
            fine_tune_epochs=3,
            atl_subject_weight=0.7,
        )

    def _run(self, *, enable_atl: bool) -> tuple[dict, dict, list[str], dict]:
        source_selected = self._split("S2")
        source_stream = self._split("S2")
        target_all = self._split("S1")
        target_support = paper_protocol.copy_split_by_indices(target_all, np.asarray([0, 1]))
        target_query = paper_protocol.copy_split_by_indices(target_all, np.asarray([2, 3]))
        calls: dict[str, object] = {}
        events: list[str] = []
        model = torch.nn.Linear(1, 1)

        def fake_train_model(**kwargs):
            events.append("pretrain")
            calls["train_model"] = kwargs
            return {"model": model, "history": [{"epoch": 1.0, "train_loss": 0.5}]}

        def fake_fine_tune_head(**kwargs):
            events.append("fine_tune")
            calls["fine_tune_head"] = kwargs
            if kwargs.get("enable_atl", False):
                score_mask = kwargs["support_score_mask"]
                self.assertTrue(np.all(np.isfinite(kwargs["support_split"].y[score_mask])))
                self.assertTrue(np.all(np.isnan(kwargs["support_split"].y[~score_mask])))
                np.testing.assert_array_equal(kwargs["support_split"].x, target_all.x)
                np.testing.assert_array_equal(kwargs["support_split"].time_s, target_all.time_s)
                np.testing.assert_array_equal(
                    kwargs["support_split"].alignment_indices,
                    target_all.alignment_indices,
                )
                np.testing.assert_array_equal(
                    kwargs["support_split"].source_recording_ids,
                    target_all.source_recording_ids,
                )
                history = [{"epoch": 1.0, "L_DD": 1.0, "L_mapping": 0.5, "L_subject": 0.25}]
            else:
                history = [{"epoch": 1.0, "train_loss": 0.25}]
            return model, history, {"mode": "synthetic"}

        def fake_evaluate(_model, split, **_kwargs):
            events.append("zero_shot" if len([e for e in events if e.endswith("shot")]) == 0 else "adapted")
            self.assertIs(split, target_query)
            return {"metrics": {}, "per_action_metrics": {}, "y_true": split.y, "y_pred": split.y}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir)
            args = self._args(output_dir, enable_atl=enable_atl)
            with (
                patch.object(paper_protocol, "discover_target_files", return_value={"S1": [], "S2": []}),
                patch.object(
                    paper_protocol,
                    "load_subject_splits",
                    side_effect=lambda subject, *_args, **_kwargs: (
                        (target_all, target_all) if subject == "S1" else (source_selected, source_stream)
                    ),
                ),
                patch.object(
                    paper_protocol,
                    "select_repetition_split",
                    return_value=(target_support, target_query, {"1": {"support": [1, 2], "query": [3, 4]}}),
                ) as split_subject,
                patch.object(paper_protocol, "train_model", side_effect=fake_train_model),
                patch.object(paper_protocol, "fine_tune_head", side_effect=fake_fine_tune_head),
                patch.object(paper_protocol, "normalize_sequence_inputs", side_effect=lambda split, **_kwargs: split),
                patch.object(paper_protocol, "evaluate_split", side_effect=fake_evaluate),
                patch.object(paper_protocol, "evaluate_chain", return_value={"metrics": {}, "h_norm_stats": {}}),
            ):
                summary = paper_protocol.run_protocol(args)
            self.assertEqual(split_subject.call_count, 1, "only the target subject may be repetition-split")
            self.assertEqual(events, ["pretrain", "zero_shot", "fine_tune", "adapted"])
            checkpoint = torch.load(
                output_dir / "checkpoints" / "S1_dense_cfc_head_ft.pt",
                map_location="cpu",
                weights_only=False,
            )
            return calls, summary, events, checkpoint

    def test_run_protocol_supervises_all_source_rows_and_standard_ft_only_support(self) -> None:
        calls, summary, _events, checkpoint = self._run(enable_atl=False)
        train = calls["train_model"]
        self.assertEqual(train["supervised_split"].x.shape[0], 4)
        self.assertTrue(np.all(train["stateful_stream_score_mask"]))
        fine_tune = calls["fine_tune_head"]
        self.assertFalse(fine_tune.get("enable_atl", False))
        self.assertEqual(fine_tune["support_split"].x.shape[0], 2)
        self.assertEqual(summary["pretrain_history"], [{"epoch": 1.0, "train_loss": 0.5}])
        self.assertEqual(summary["fine_tune_history"], [{"epoch": 1.0, "train_loss": 0.25}])
        self.assertEqual(set(checkpoint["history"][0]), {"epoch", "train_loss"})
        self.assertEqual(checkpoint["adaptation_provenance"]["mode"], "head_only")
        self.assertEqual(checkpoint["adaptation_provenance"]["learning_rate"], 3e-4)
        self.assertIsNone(checkpoint["adaptation_provenance"]["atl_subject_weight"])

    def test_run_protocol_atl_uses_full_target_stream_and_support_only_mask(self) -> None:
        calls, summary, _events, checkpoint = self._run(enable_atl=True)
        fine_tune = calls["fine_tune_head"]
        self.assertTrue(fine_tune["enable_atl"])
        self.assertEqual(fine_tune["support_split"].x.shape[0], 4)
        np.testing.assert_array_equal(fine_tune["support_score_mask"], [True, True, False, False])
        self.assertEqual(fine_tune["source_split"].x.shape[0], 4)
        self.assertTrue(np.all(fine_tune["source_score_mask"]))
        self.assertEqual(summary["fine_tune_history"], [{"epoch": 1.0, "L_DD": 1.0, "L_mapping": 0.5, "L_subject": 0.25}])
        self.assertEqual(set(checkpoint["history"][0]), {"epoch", "L_DD", "L_mapping", "L_subject"})
        provenance = checkpoint["adaptation_provenance"]
        self.assertEqual(provenance["mode"], "atl_stateful")
        self.assertEqual(provenance["epochs"], 3)
        self.assertEqual(provenance["learning_rate"], 1e-4)
        self.assertEqual(provenance["atl_subject_weight"], 0.7)

    def test_run_protocol_save_then_resume_preserves_training_history_and_rejects_architecture_change(self) -> None:
        source_selected = self._split("S2")
        source_stream = self._split("S2")
        target_all = self._split("S1")
        target_support = paper_protocol.copy_split_by_indices(target_all, np.asarray([0, 1]))
        target_query = paper_protocol.copy_split_by_indices(target_all, np.asarray([2, 3]))

        def fake_evaluate(_model, split, **_kwargs):
            self.assertIs(split, target_query)
            return {"metrics": {}, "per_action_metrics": {}, "y_true": split.y, "y_pred": split.y}

        def enter_common_patches(stack: ExitStack) -> None:
            stack.enter_context(
                patch.object(paper_protocol, "discover_target_files", return_value={"S1": [], "S2": []})
            )
            stack.enter_context(
                patch.object(
                    paper_protocol,
                    "load_subject_splits",
                    side_effect=lambda subject, *_args, **_kwargs: (
                        (target_all, target_all) if subject == "S1" else (source_selected, source_stream)
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    paper_protocol,
                    "select_repetition_split",
                    return_value=(target_support, target_query, {"1": {"support": [1, 2], "query": [3, 4]}}),
                )
            )
            stack.enter_context(
                patch.object(paper_protocol, "normalize_sequence_inputs", side_effect=lambda split, **_kwargs: split)
            )
            stack.enter_context(patch.object(paper_protocol, "evaluate_split", side_effect=fake_evaluate))
            stack.enter_context(
                patch.object(paper_protocol, "evaluate_chain", return_value={"metrics": {}, "h_norm_stats": {}})
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir)
            first_args = self._args(output_dir, enable_atl=False)
            first_args.skip_fine_tune = True
            initial_model = torch.nn.Linear(1, 1)
            with ExitStack() as stack:
                enter_common_patches(stack)
                stack.enter_context(patch.object(
                    paper_protocol,
                    "train_model",
                    return_value={"model": initial_model, "history": [{"epoch": 1.0, "train_loss": 0.5}]},
                ))
                paper_protocol.run_protocol(first_args)

            checkpoint_path = output_dir / "checkpoints" / "S1_dense_cfc_pretrain.pt"
            initial_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(initial_checkpoint["history"], [{"epoch": 1.0, "train_loss": 0.5}])
            initial_provenance = initial_checkpoint["pretrain_provenance"]
            self.assertEqual(initial_provenance["status"], "originating_pretrain")
            self.assertEqual(initial_provenance["augment_prob_requested"], 1.0)
            self.assertEqual(initial_provenance["augment_prob_effective"], 0.0)
            self.assertTrue(initial_provenance["gpu_resident_effective"])

            resume_args = self._args(output_dir, enable_atl=False)
            resume_args.skip_fine_tune = True
            resume_args.resume_pretrain = checkpoint_path
            resume_args.max_epochs = 99
            resume_args.learning_rate = 7e-4
            resume_args.weight_decay = 2e-3
            resume_args.batch_size = 3
            resume_args.augment_prob = 0.25
            with ExitStack() as stack:
                enter_common_patches(stack)
                stack.enter_context(
                    patch.object(paper_protocol, "train_model", side_effect=AssertionError("resume must not retrain"))
                )
                stack.enter_context(patch.object(paper_protocol, "infer_checkpoint_output_dim", return_value=1))
                stack.enter_context(
                    patch.object(paper_protocol, "build_cfc_regressor", return_value=torch.nn.Linear(1, 1))
                )
                resumed_summary = paper_protocol.run_protocol(resume_args)

            resumed_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(resumed_summary["pretrain_history"], [{"epoch": 1.0, "train_loss": 0.5}])
            self.assertEqual(resumed_checkpoint["history"], [{"epoch": 1.0, "train_loss": 0.5}])
            self.assertEqual(resumed_checkpoint["config"], initial_checkpoint["config"])
            self.assertEqual(
                resumed_summary["config"],
                paper_protocol.make_jsonable(initial_checkpoint["config"]),
            )
            self.assertEqual(resumed_summary["invocation_config"]["max_epochs"], 99)
            self.assertEqual(resumed_summary["invocation_config"]["learning_rate"], 7e-4)
            self.assertEqual(resumed_checkpoint["pretrain_provenance"], initial_provenance)
            self.assertEqual(resumed_summary["pretrain_provenance"], initial_provenance)
            self.assertEqual(resumed_checkpoint["resume_invocation"]["current_config"]["max_epochs"], 99)
            self.assertEqual(resumed_checkpoint["resume_invocation"]["current_config"]["learning_rate"], 7e-4)
            self.assertEqual(resumed_checkpoint["resume_invocation"]["current_config"]["weight_decay"], 2e-3)
            self.assertEqual(resumed_checkpoint["resume_invocation"]["current_config"]["batch_size"], 3)
            self.assertEqual(resumed_checkpoint["resume_invocation"]["augment_prob_requested"], 0.25)
            self.assertEqual(resumed_checkpoint["resume_invocation"]["status"], "resumed")

            legacy_checkpoint_path = output_dir / "checkpoints" / "legacy_without_pretrain_provenance.pt"
            legacy_checkpoint = copy.deepcopy(initial_checkpoint)
            legacy_checkpoint.pop("pretrain_provenance")
            torch.save(legacy_checkpoint, legacy_checkpoint_path)
            legacy_args = self._args(output_dir, enable_atl=False)
            legacy_args.skip_fine_tune = True
            legacy_args.resume_pretrain = legacy_checkpoint_path
            with self.assertWarnsRegex(UserWarning, "lacks pretrain_provenance"):
                with ExitStack() as stack:
                    enter_common_patches(stack)
                    stack.enter_context(patch.object(paper_protocol, "infer_checkpoint_output_dim", return_value=1))
                    stack.enter_context(
                        patch.object(paper_protocol, "build_cfc_regressor", return_value=torch.nn.Linear(1, 1))
                    )
                    paper_protocol.run_protocol(legacy_args)
            rewritten_legacy_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(rewritten_legacy_checkpoint["pretrain_provenance"]["status"], "legacy_unknown")
            self.assertIsNone(rewritten_legacy_checkpoint["pretrain_provenance"]["augment_prob_requested"])

            incompatible_args = self._args(output_dir, enable_atl=False)
            incompatible_args.skip_fine_tune = True
            incompatible_args.resume_pretrain = checkpoint_path
            incompatible_args.hidden_units = 9
            with ExitStack() as stack:
                enter_common_patches(stack)
                stack.enter_context(patch.object(paper_protocol, "infer_checkpoint_output_dim", return_value=1))
                with self.assertRaisesRegex(ValueError, "--hidden-units"):
                    paper_protocol.run_protocol(incompatible_args)


class SegmentIndexTests(unittest.TestCase):

    def _split_with_labels(
        self,
        recordings: list[str],
        actions: list[int] | None,
        repetitions: list[int] | None,
        *,
        stream_start_flags: list[bool] | None = None,
    ) -> SequenceSplit:
        n = len(recordings)
        raw = np.arange(n + 2, dtype=np.float32)
        x = (
            np.stack([raw[index : index + 3] for index in range(n)], axis=0)[..., None]
            if n
            else np.empty((0, 3, 1), dtype=np.float32)
        )
        return SequenceSplit(
            x=x,
            y=np.zeros((n, 1), dtype=np.float32),
            time_s=np.arange(n, dtype=np.float32) * 0.05,
            alignment_indices=np.arange(n, dtype=np.int64),
            recording_ids=np.asarray(recordings),
            feature_names=["f"],
            target_names=["t"],
            action_labels=None if actions is None else np.asarray(actions, dtype=np.int16),
            repetition_labels=None if repetitions is None else np.asarray(repetitions, dtype=np.int16),
            stream_start_flags=(
                None if stream_start_flags is None else np.asarray(stream_start_flags, dtype=bool)
            ),
        )

    def test_action_and_repetition_changes_do_not_reset_a_continuous_stream(self) -> None:
        split = self._split_with_labels(
            recordings=["f1.mat"] * 6,
            actions=[18, 18, 18, 19, 19, 19],
            repetitions=[1, 1, 1, 2, 2, 2],
        )
        index = SegmentIndex(split)
        self.assertEqual(index.segments, [(0, 6)])

    def test_explicit_stream_start_resets_within_one_recording(self) -> None:
        split = self._split_with_labels(
            recordings=["f1.mat"] * 6,
            actions=[18, 18, 18, 19, 19, 19],
            repetitions=[1, 1, 1, 1, 1, 1],
            stream_start_flags=[True, False, False, True, False, False],
        )
        index = SegmentIndex(split)
        self.assertEqual(index.segments, [(0, 3), (3, 6)])

    def test_recording_change_splits_segment(self) -> None:
        split = self._split_with_labels(
            recordings=["f1.mat"] * 3 + ["f2.mat"] * 3,
            actions=[18] * 6,
            repetitions=[1] * 6,
        )
        index = SegmentIndex(split)
        self.assertEqual(index.segments, [(0, 3), (3, 6)])

    def test_broken_sequence_overlap_splits_a_stream(self) -> None:
        split = self._split_with_labels(
            recordings=["f1.mat"] * 4,
            actions=None,
            repetitions=None,
        )
        split.x[2] += 100.0
        index = SegmentIndex(split)
        self.assertEqual(index.segments, [(0, 2), (2, 3), (3, 4)])

    def test_seq_len_one_uses_the_declared_alignment_step(self) -> None:
        split = SequenceSplit(
            x=np.ones((4, 1, 1), dtype=np.float32),
            y=np.zeros((4, 1), dtype=np.float32),
            time_s=np.arange(4, dtype=np.float32) * 0.05,
            alignment_indices=np.asarray([100, 200, 400, 500], dtype=np.int64),
            recording_ids=np.asarray(["f1.mat"] * 4),
            feature_names=["f"],
            target_names=["t"],
            expected_alignment_step=100,
        )

        self.assertEqual(SegmentIndex(split).segments, [(0, 2), (2, 4)])

    def test_seq_len_one_preserves_a_contiguous_nonunit_alignment_step(self) -> None:
        split = SequenceSplit(
            x=np.ones((3, 1, 1), dtype=np.float32),
            y=np.zeros((3, 1), dtype=np.float32),
            time_s=np.arange(3, dtype=np.float32) * 0.05,
            alignment_indices=np.asarray([100, 200, 300], dtype=np.int64),
            recording_ids=np.asarray(["f1.mat"] * 3),
            feature_names=["f"],
            target_names=["t"],
            expected_alignment_step=100,
        )

        self.assertEqual(SegmentIndex(split).segments, [(0, 3)])

    def test_seq_len_one_without_alignment_provenance_fails_closed(self) -> None:
        split = SequenceSplit(
            x=np.ones((3, 1, 1), dtype=np.float32),
            y=np.zeros((3, 1), dtype=np.float32),
            time_s=np.arange(3, dtype=np.float32) * 0.05,
            alignment_indices=np.asarray([100, 200, 300], dtype=np.int64),
            recording_ids=np.asarray(["f1.mat"] * 3),
            feature_names=["f"],
            target_names=["t"],
        )

        self.assertEqual(SegmentIndex(split).segments, [(0, 1), (1, 2), (2, 3)])

    def test_empty_split_yields_no_segments(self) -> None:
        split = self._split_with_labels(recordings=[], actions=[], repetitions=[])
        index = SegmentIndex(split)
        self.assertEqual(len(index), 0)
        self.assertEqual(index.segments, [])

    def test_segments_cover_all_rows_without_overlap(self) -> None:
        split = self._split_with_labels(
            recordings=["f1.mat"] * 7 + ["f2.mat"] * 5,
            actions=[1, 1, 2, 2, 2, 3, 3] + [1, 1, 1, 2, 2],
            repetitions=[1] * 12,
        )
        index = SegmentIndex(split)
        covered = [pos for start, end in index.segments for pos in range(start, end)]
        self.assertEqual(covered, list(range(split.x.shape[0])))
        for (s1, e1), (s2, e2) in zip(index.segments, index.segments[1:]):
            self.assertLessEqual(e1, s2, msg="segments must not overlap")


class ForwardWithStateTests(unittest.TestCase):

    def _model(self) -> DenseCfCLinearRegressor:
        return build_cfc_regressor(
            input_dim=16, output_dim=5, hidden_units=64,
            model_family="dense_cfc_linear", cfc_dropout=0.1,
        )

    def test_hx_none_is_bitwise_identical_to_forward(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = self._model().to(device)
        x = torch.randn(4, 8, 16, device=device)
        model.eval()
        with torch.no_grad():
            pred_forward = model(x)
            pred_state, _ = model.forward_with_state(x, None)
        torch.testing.assert_close(pred_state, pred_forward, rtol=0, atol=0)

    def test_carry_matches_unrolled_sequence(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(0)
        model = self._model().to(device)
        model.eval()
        x1 = torch.randn(3, 8, 16, device=device)
        x2 = torch.randn(3, 8, 16, device=device)
        with torch.no_grad():
            _, h1 = model.forward_with_state(x1, None)
            _, h2 = model.forward_with_state(x2, h1)
            y_full, _ = model.cfc(torch.cat([x1, x2], dim=1), None)
        # h1 == state after frame 8 of the 16-frame run (index 7)
        torch.testing.assert_close(h1, y_full[:, 7, :], rtol=0, atol=1e-6)
        # carry: x2 processed from h1 ends at the 16-frame run's final state
        torch.testing.assert_close(h2, y_full[:, -1, :], rtol=0, atol=1e-6)

    def test_hx_shape_validation(self) -> None:
        device = torch.device("cpu")
        model = self._model().to(device)
        x = torch.randn(4, 8, 16, device=device)
        with self.assertRaises(ValueError):
            model.forward_with_state(x, torch.zeros(4, 128, device=device))

    def test_h_final_is_detached(self) -> None:
        device = torch.device("cpu")
        model = self._model().to(device)
        x = torch.randn(2, 8, 16, device=device)
        _, h = model.forward_with_state(x, None)
        self.assertFalse(h.requires_grad, msg="carried h must not carry gradients")


class GpuResidentStatefulBatchesTests(unittest.TestCase):

    def _split(self) -> SequenceSplit:
        streams = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        ]
        sequences = [streams[0][index : index + 3] for index in range(3)]
        sequences.extend(streams[1][index : index + 3] for index in range(2))
        return SequenceSplit(
            x=np.asarray(sequences, dtype=np.float32)[..., None],
            y=np.asarray([[6.0], [10.0], [15.0], [60.0], [100.0]], dtype=np.float32),
            time_s=np.arange(5, dtype=np.float32) * 0.05,
            alignment_indices=np.arange(5, dtype=np.int64),
            recording_ids=np.asarray(["A"] * 3 + ["B"] * 2),
            feature_names=["f"],
            target_names=["sum"],
            action_labels=np.asarray([18, 18, 19, 20, 20], dtype=np.int16),
            repetition_labels=np.asarray([1, 1, 2, 1, 1], dtype=np.int16),
        )

    def _batches(self, split, batch_size, seed=42):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        index = SegmentIndex(split)
        batches = GpuResidentStatefulBatches(
            split.x, split.y, index, batch_size=batch_size,
            device=torch.device("cpu"), generator=generator,
        )
        return index, batches, list(batches)

    def _collect_scored_pairs(self, batch_size: int) -> list[tuple[float, float]]:
        split = self._split()
        _index, _batches, collected = self._batches(split, batch_size=batch_size)
        pairs = []
        for x_chunk, y_chunk, score_mask, _reset in collected:
            pairs.extend(
                (float(x_chunk[lane, step, 0]), float(y_chunk[lane, step, 0]))
                for lane, step in score_mask.nonzero(as_tuple=False).tolist()
            )
        return sorted(pairs)

    def test_reconstructs_one_new_frame_per_step_and_scores_each_target_once(self) -> None:
        split = self._split()
        _index, _batches, collected = self._batches(split, batch_size=2)

        reconstructed = {}
        for chunk_index, (x_chunk, _y_chunk, score_mask, reset_before) in enumerate(collected):
            self.assertEqual(x_chunk.shape, (2, 3, 1))
            self.assertEqual(score_mask.shape, (2, 3))
            if chunk_index == 0:
                self.assertTrue(bool(reset_before.all()))
                for lane in range(2):
                    key = int(x_chunk[lane, 0, 0].item())
                    reconstructed[key] = x_chunk[lane, :, 0].tolist()
            else:
                self.assertFalse(bool(reset_before.any()))
                for lane in range(2):
                    first = int(collected[0][0][lane, 0, 0].item())
                    valid_steps = score_mask[lane].nonzero(as_tuple=False).flatten().tolist()
                    reconstructed[first].extend(float(x_chunk[lane, step, 0]) for step in valid_steps)

        self.assertEqual(reconstructed[1], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(reconstructed[10], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(
            self._collect_scored_pairs(batch_size=2),
            [(3.0, 6.0), (4.0, 10.0), (5.0, 15.0), (30.0, 60.0), (40.0, 100.0)],
        )

    def test_temporal_target_pairing_is_batch_size_invariant(self) -> None:
        self.assertEqual(
            self._collect_scored_pairs(batch_size=1),
            self._collect_scored_pairs(batch_size=2),
        )

    def test_batch_size_is_a_cap_on_independent_stream_lanes(self) -> None:
        split = self._split()
        batches = GpuResidentStatefulBatches(
            split.x,
            split.y,
            SegmentIndex(split),
            batch_size=1024,
            device=torch.device("cpu"),
        )

        self.assertEqual(batches.requested_batch_size, 1024)
        self.assertEqual(batches.batch_size, 2)

    def test_external_score_mask_keeps_context_frames_but_excludes_labels(self) -> None:
        split = self._split()
        selected = np.asarray([False, True, False, True, False], dtype=bool)
        batches = GpuResidentStatefulBatches(
            split.x,
            split.y,
            SegmentIndex(split),
            batch_size=2,
            device=torch.device("cpu"),
            generator=torch.Generator(device="cpu").manual_seed(42),
            sequence_score_mask=selected,
        )

        scored_pairs = []
        observed_frames = []
        for x_chunk, y_chunk, score_mask, _reset in batches:
            observed_frames.extend(x_chunk[..., 0].reshape(-1).tolist())
            scored_pairs.extend(
                (float(x_chunk[lane, step, 0]), float(y_chunk[lane, step, 0]))
                for lane, step in score_mask.nonzero(as_tuple=False).tolist()
            )

        self.assertIn(3.0, observed_frames)
        self.assertEqual(sorted(scored_pairs), [(4.0, 10.0), (30.0, 60.0)])

    def test_training_metadata_matches_chunk_tensors(self) -> None:
        split = self._split()
        batches = GpuResidentStatefulBatches(
            split.x,
            split.y,
            SegmentIndex(split),
            batch_size=2,
            device=torch.device("cpu"),
            generator=torch.Generator(device="cpu").manual_seed(42),
            sequence_score_mask=np.asarray([False, True, False, True, False]),
        )

        for _x_chunk, _y_chunk, score_mask, reset_before, requires_reset, score_count in batches.iter_training_batches():
            self.assertEqual(requires_reset, bool(reset_before.any()))
            self.assertEqual(score_count, int(score_mask.sum()))


class _CumulativeCfC(torch.nn.Module):
    state_size = 1

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.received_hx: list[torch.Tensor] = []

    def forward(self, x, hx=None):
        if hx is None:
            hx = torch.zeros(x.shape[0], 1, device=x.device)
        self.received_hx.append(hx.detach().clone())
        outputs = hx[:, None, :] + torch.cumsum(x[..., :1] * self.scale, dim=1)
        return outputs, outputs[:, -1, :]


class _CumulativeRegressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfc = _CumulativeCfC()
        self.dropout = torch.nn.Identity()
        self.head = torch.nn.Identity()


class StatefulResetTimingTests(unittest.TestCase):

    def test_stateful_epoch_uses_cpu_batch_metadata_for_control_flow(self) -> None:
        source = inspect.getsource(training_module.train_one_epoch_stateful)

        self.assertIn("iter_training_batches", source)
        self.assertNotIn("reset_before.any()", source)
        self.assertNotIn("score_mask.sum()", source)

    def test_causal_stateful_loss_is_zero_for_any_segment_batch_size(self) -> None:
        split = GpuResidentStatefulBatchesTests()._split()

        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                model = _CumulativeRegressor()
                batches = GpuResidentStatefulBatches(
                    split.x,
                    split.y,
                    SegmentIndex(split),
                    batch_size=batch_size,
                    device=torch.device("cpu"),
                    generator=torch.Generator(device="cpu").manual_seed(42),
                )
                optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

                loss = training_module.train_one_epoch_stateful(
                    model,
                    batches,
                    optimizer,
                    torch.nn.MSELoss(),
                    device=torch.device("cpu"),
                    gradient_clip_norm=None,
                )

                self.assertEqual(loss, 0.0)

    def test_state_is_reset_before_the_first_frame_of_each_stream(self) -> None:
        split = SequenceSplit(
            x=np.asarray([[[1.0], [2.0]], [[10.0], [20.0]]], dtype=np.float32),
            y=np.asarray([[3.0], [30.0]], dtype=np.float32),
            time_s=np.asarray([0.05, 0.10], dtype=np.float32),
            alignment_indices=np.asarray([1, 2], dtype=np.int64),
            recording_ids=np.asarray(["A", "B"]),
            feature_names=["f"],
            target_names=["sum"],
        )
        batches = GpuResidentStatefulBatches(
            split.x,
            split.y,
            SegmentIndex(split),
            batch_size=1,
            device=torch.device("cpu"),
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
        model = _CumulativeRegressor()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        training_module.train_one_epoch_stateful(
            model,
            batches,
            optimizer,
            torch.nn.MSELoss(),
            device=torch.device("cpu"),
            gradient_clip_norm=None,
        )

        self.assertEqual(len(model.cfc.received_hx), 2)
        for received_hx in model.cfc.received_hx:
            torch.testing.assert_close(received_hx, torch.zeros_like(received_hx))

    def test_unscored_context_advances_hidden_state_without_contributing_loss(self) -> None:
        split = GpuResidentStatefulBatchesTests()._split()
        batches = GpuResidentStatefulBatches(
            split.x,
            split.y,
            SegmentIndex(split),
            batch_size=2,
            device=torch.device("cpu"),
            generator=torch.Generator(device="cpu").manual_seed(42),
            sequence_score_mask=np.asarray([False, True, False, True, False]),
        )
        model = _CumulativeRegressor()

        loss = training_module.train_one_epoch_stateful(
            model,
            batches,
            torch.optim.SGD(model.parameters(), lr=0.0),
            torch.nn.MSELoss(),
            device=torch.device("cpu"),
            gradient_clip_norm=None,
        )

        self.assertEqual(loss, 0.0)


class StatefulTrainingIntegrationTests(unittest.TestCase):
    """End-to-end stateful pretraining and ATL isolation."""

    def _config(self) -> CfCTrainingConfig:
        return build_best_cfc_config(
            emg_channels=(1, 2, 3, 4, 5, 6, 7, 8),
            target_source="glove", target_columns=(),
            target_mapping="doa5", target_mapping_source="glove",
            window_ms=200.0, stride_ms=50.0, target_offset_samples=200,
            feature_order=("rms", "zc"), feature_normalization="mu_law",
            target_normalization="mu_law", mu_law_mu=DEFAULT_MU_LAW_MU,
            seq_len=8, seq_stride=1, hidden_units=32,
            model_family="dense_cfc_linear", batch_size=32,
            learning_rate=1e-3, weight_decay=1e-4, cfc_dropout=0.0,
            max_epochs=3, early_stopping_patience=30,
            random_seed=42, device="cpu", num_workers=0,
        )

    def _split(self, n: int) -> SequenceSplit:
        if n % 2:
            raise ValueError("synthetic split requires an even sequence count")
        rng = np.random.default_rng(0)
        sequences_per_recording = n // 2
        streams = rng.standard_normal((2, sequences_per_recording + 7, 16)).astype(np.float32)
        x = np.concatenate(
            [
                np.stack(
                    [stream[index : index + 8] for index in range(sequences_per_recording)],
                    axis=0,
                )
                for stream in streams
            ],
            axis=0,
        )
        y = rng.standard_normal((n, 5)).astype(np.float32)
        rec = np.asarray([f"r{i // sequences_per_recording}" for i in range(n)])
        act = np.asarray([18 + (i % 256) // 128 for i in range(n)], dtype=np.int16)
        rep = np.asarray([1 + (i % 128) // 64 for i in range(n)], dtype=np.int16)
        return SequenceSplit(
            x=x, y=y, time_s=np.arange(n, dtype=np.float32) * 0.05,
            alignment_indices=np.arange(n, dtype=np.int64),
            recording_ids=rec, feature_names=[], target_names=[],
            action_labels=act, repetition_labels=rep,
        )

    def test_stateful_pretrain_loss_decreases(self) -> None:
        from run_db2_paper_cfc_finetune import train_model
        n = 256
        split = self._split(n)
        config = self._config()
        x_stats = fit_feature_normalizer(
            split.x.reshape(-1, 16), method="mu_law", mu=DEFAULT_MU_LAW_MU)
        y_stats = fit_target_normalizer(split.y, method="mu_law", mu=DEFAULT_MU_LAW_MU)
        with (
            patch.object(paper_protocol, "evaluate_chain", side_effect=AssertionError("no validation")),
            patch.object(paper_protocol, "evaluate_split", side_effect=AssertionError("no validation")),
        ):
            result = train_model(
            supervised_split=split, config=config,
            x_stats=x_stats, y_stats=y_stats, device=torch.device("cpu"),
            augment_prob=0.0, cuda_graph=False, gpu_resident=True,
            stateful=True,
            )
        history = result["history"]
        self.assertEqual(len(history), 3)
        self.assertTrue(all(set(entry) == {"epoch", "train_loss"} for entry in history))
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"],
                        msg="stateful training loss must decrease")

    def test_pretrain_returns_the_final_epoch_without_state_rewind(self) -> None:
        from run_db2_paper_cfc_finetune import train_model

        class MarkerModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(()))

            def forward(self, x):
                return self.marker.expand(x.shape[0], 5)

        split = self._split(8)
        config = self._config()
        marker_model = MarkerModel()
        x_stats = fit_feature_normalizer(
            split.x.reshape(-1, 16), method="mu_law", mu=DEFAULT_MU_LAW_MU
        )
        y_stats = fit_target_normalizer(
            split.y, method="mu_law", mu=DEFAULT_MU_LAW_MU
        )

        class IncrementOptimizer:
            def __init__(self, params, **_kwargs) -> None:
                self.params = list(params)

            def zero_grad(self, **_kwargs) -> None:
                for parameter in self.params:
                    parameter.grad = None

            def step(self) -> None:
                with torch.no_grad():
                    self.params[0].add_(1.0)

        with (
            patch.object(paper_protocol, "build_cfc_regressor", return_value=marker_model),
            patch.object(paper_protocol.torch.optim, "AdamW", IncrementOptimizer),
            patch.object(paper_protocol, "evaluate_split", side_effect=AssertionError("no validation")),
        ):
            result = train_model(
                supervised_split=split,
                config=config,
                x_stats=x_stats,
                y_stats=y_stats,
                device=torch.device("cpu"),
                augment_prob=0.0,
                cuda_graph=False,
                gpu_resident=False,
                stateful=False,
            )

        self.assertEqual(float(result["model"].marker.item()), 3.0)
        self.assertEqual(len(result["history"]), 3)

    def test_chain_evaluation_preserves_exact_pairs_across_recordings(self) -> None:
        from run_db2_paper_cfc_finetune import _chain_metrics_dict
        split = SequenceSplit(
            x=np.asarray(
                [
                    [[1.0], [2.0]],
                    [[2.0], [3.0]],
                    [[3.0], [4.0]],
                    [[100.0], [200.0]],
                    [[200.0], [300.0]],
                    [[300.0], [400.0]],
                ],
                dtype=np.float32,
            ),
            y=np.asarray([[3.0], [6.0], [10.0], [300.0], [600.0], [1000.0]], dtype=np.float32),
            time_s=np.arange(6, dtype=np.float32) * 0.05,
            alignment_indices=np.arange(6, dtype=np.int64),
            recording_ids=np.asarray(["A"] * 3 + ["B"] * 3),
            feature_names=["f"],
            target_names=["sum"],
            action_labels=np.asarray([18, 18, 19, 20, 20, 21], dtype=np.int16),
            repetition_labels=np.asarray([1, 1, 2, 1, 1, 2], dtype=np.int16),
        )
        model = _CumulativeRegressor()
        y_stats = {
            "method": "zscore",
            "mean": np.zeros(1, dtype=np.float32),
            "std": np.ones(1, dtype=np.float32),
        }
        out = evaluate_chain(model, split, target_stats=y_stats, device=torch.device("cpu"))

        np.testing.assert_array_equal(out["y_true"].reshape(-1), split.y.reshape(-1))
        np.testing.assert_allclose(out["y_pred"].reshape(-1), split.y.reshape(-1), rtol=0, atol=0)
        self.assertEqual(out["metrics"]["mae_mean"], 0.0)
        self.assertEqual(out["h_norm_stats"]["n_chains"], 2)
        d = _chain_metrics_dict(out)
        self.assertIn("metrics", d)
        self.assertIn("h_norm_stats", d)

    def test_chain_evaluation_batches_unequal_recordings_without_changing_order(self) -> None:
        split = SequenceSplit(
            x=np.asarray(
                [
                    [[1.0], [2.0]],
                    [[2.0], [3.0]],
                    [[3.0], [4.0]],
                    [[10.0], [20.0]],
                ],
                dtype=np.float32,
            ),
            y=np.asarray([[3.0], [6.0], [10.0], [30.0]], dtype=np.float32),
            time_s=np.arange(4, dtype=np.float32) * 0.05,
            alignment_indices=np.asarray([0, 1, 2, 0], dtype=np.int64),
            recording_ids=np.asarray(["A", "A", "A", "B"]),
            feature_names=["f"],
            target_names=["sum"],
        )
        model = _CumulativeRegressor()
        y_stats = {
            "method": "zscore",
            "mean": np.zeros(1, dtype=np.float32),
            "std": np.ones(1, dtype=np.float32),
        }

        out = evaluate_chain(
            model,
            split,
            target_stats=y_stats,
            device=torch.device("cpu"),
            score_mask=np.asarray([True, False, True, True]),
        )

        np.testing.assert_array_equal(out["y_true"].reshape(-1), [3.0, 10.0, 30.0])
        np.testing.assert_allclose(out["y_pred"].reshape(-1), [3.0, 10.0, 30.0], rtol=0, atol=0)
        self.assertEqual(out["h_norm_stats"]["mean_step30"], 20.0)
        self.assertEqual(out["h_norm_stats"]["mean_step_last"], 20.0)
        self.assertEqual(len(model.cfc.received_hx), 2)
        self.assertTrue(all(tuple(hx.shape) == (2, 1) for hx in model.cfc.received_hx))

    def test_chain_evaluation_does_not_reset_on_action_or_repetition_labels(self) -> None:
        split = SequenceSplit(
            x=np.asarray(
                [[[1.0], [2.0]], [[2.0], [3.0]], [[3.0], [4.0]]],
                dtype=np.float32,
            ),
            y=np.asarray([[3.0], [6.0], [10.0]], dtype=np.float32),
            time_s=np.arange(3, dtype=np.float32) * 0.05,
            alignment_indices=np.arange(3, dtype=np.int64),
            recording_ids=np.asarray(["A"] * 3),
            feature_names=["f"],
            target_names=["sum"],
            action_labels=np.asarray([18, 19, 19], dtype=np.int16),
            repetition_labels=np.asarray([1, 1, 2], dtype=np.int16),
        )
        y_stats = {
            "method": "zscore",
            "mean": np.zeros(1, dtype=np.float32),
            "std": np.ones(1, dtype=np.float32),
        }

        out = evaluate_chain(
            _CumulativeRegressor(),
            split,
            target_stats=y_stats,
            device=torch.device("cpu"),
        )

        np.testing.assert_allclose(out["y_pred"].reshape(-1), [3.0, 6.0, 10.0], rtol=0, atol=0)
        self.assertEqual(out["h_norm_stats"]["n_chains"], 1)

    def test_chain_evaluation_carries_through_unscored_context_frames(self) -> None:
        split = SequenceSplit(
            x=np.asarray(
                [
                    [[1.0], [2.0]],
                    [[2.0], [3.0]],
                    [[3.0], [4.0]],
                    [[4.0], [5.0]],
                ],
                dtype=np.float32,
            ),
            y=np.asarray([[3.0], [6.0], [10.0], [15.0]], dtype=np.float32),
            time_s=np.arange(4, dtype=np.float32) * 0.05,
            alignment_indices=np.arange(4, dtype=np.int64),
            recording_ids=np.asarray(["A"] * 4),
            feature_names=["f"],
            target_names=["sum"],
            action_labels=np.asarray([18, 0, 19, 19], dtype=np.int16),
            repetition_labels=np.asarray([1, 0, 2, 2], dtype=np.int16),
        )
        y_stats = {
            "method": "zscore",
            "mean": np.zeros(1, dtype=np.float32),
            "std": np.ones(1, dtype=np.float32),
        }

        out = evaluate_chain(
            _CumulativeRegressor(),
            split,
            target_stats=y_stats,
            device=torch.device("cpu"),
            score_mask=np.asarray([True, False, True, True]),
        )

        np.testing.assert_array_equal(out["y_true"].reshape(-1), [3.0, 10.0, 15.0])
        np.testing.assert_allclose(out["y_pred"].reshape(-1), [3.0, 10.0, 15.0], rtol=0, atol=0)
        self.assertEqual(out["h_norm_stats"]["n_chains"], 1)

    def test_chain_protocol_metadata_declares_full_stream_context_and_query_scoring(self) -> None:
        metadata = paper_protocol.chain_evaluation_protocol_metadata()

        self.assertEqual(metadata["scope"], "chronological_feature_frame_recurrent_replay")
        self.assertIn("exact", metadata["score_rows"])
        self.assertIn("rest", metadata["input_context"])
        self.assertIn("model", metadata["session_initialization"])
        self.assertIn("action", metadata["state_policy"])
        self.assertIn("zero-phase", metadata["preprocessing_scope"])
        self.assertIn("not end-to-end causal", metadata["preprocessing_scope"])
        self.assertIn("no source or support labels", metadata["selection_role"])

        training_metadata = paper_protocol.stateful_training_protocol_metadata()
        self.assertIn("all chronological", training_metadata["input_context"])
        self.assertIn("every selected", training_metadata["label_usage"])
        self.assertIn("input-transductive", training_metadata["held_out_input_scope"])
        self.assertIn("query-context", training_metadata["gradient_scope"])

    def test_atl_uses_all_support_labels_without_validation_or_checkpoint_selection(self) -> None:
        from run_db2_paper_cfc_finetune import fine_tune_head

        split = self._split(8)
        config = self._config()
        model = DenseCfCLinearRegressor(16, 5, 16, dropout=0.0)
        y_stats = {
            "method": "zscore",
            "mean": np.zeros(5, dtype=np.float32),
            "std": np.ones(5, dtype=np.float32),
        }
        epoch_metrics = {
            "L_DD": 1.0,
            "L_mapping": 0.5,
            "L_subject": 0.25,
            "dd_source_acc": 0.5,
            "dd_target_acc": 0.5,
            "n_source_domain_frames": 8.0,
            "n_target_domain_frames": 8.0,
            "n_supervised_frames": 8.0,
        }
        score_mask = np.asarray([True, False, True, False, True, False, True, False])

        with (
            patch.object(paper_protocol, "_atl_training_epoch", return_value=epoch_metrics) as train_epoch,
            patch.object(paper_protocol, "evaluate_chain", side_effect=AssertionError("no validation")),
            patch.object(
                paper_protocol,
                "evaluate_split",
                side_effect=AssertionError("ATL validation must be causal/stateful"),
            ),
        ):
            _adapted, history, audit = fine_tune_head(
                model=model,
                support_split=split,
                config=config,
                y_stats=y_stats,
                device=torch.device("cpu"),
                learning_rate=1e-3,
                epochs=1,
                enable_atl=True,
                source_split=split,
                source_score_mask=score_mask,
                support_score_mask=score_mask,
            )

        source_loader = train_epoch.call_args.kwargs["source_loader"]
        target_loader = train_epoch.call_args.kwargs["target_loader"]
        self.assertIsInstance(source_loader, GpuResidentStatefulBatches)
        self.assertIsInstance(target_loader, GpuResidentStatefulBatches)
        np.testing.assert_array_equal(source_loader.sequence_score_mask.cpu(), score_mask)
        np.testing.assert_array_equal(target_loader.sequence_score_mask.cpu(), score_mask)
        self.assertEqual(train_epoch.call_args.kwargs["dd"].net[0].in_features, 16)
        self.assertEqual(set(history[0]), {"epoch", "L_DD", "L_mapping", "L_subject"})
        self.assertEqual(audit["mode"], "atl_gan_stateful_tbptt")
        self.assertEqual(audit["hidden_units"], 16)

    def test_atl_carries_and_resets_independent_domain_states_with_masked_losses(self) -> None:
        class Chunks:
            batch_size = 2

            def __init__(self, chunks):
                self.chunks = chunks

            def __iter__(self):
                return iter(self.chunks)

            def iter_training_batches(self):
                for chunk in self.chunks:
                    yield (*chunk, bool(chunk[3].any()), int(chunk[2].sum()))

        torch.manual_seed(7)
        source_model = DenseCfCLinearRegressor(1, 1, 8, dropout=0.0)
        target_model = copy.deepcopy(source_model)
        for parameter in source_model.parameters():
            parameter.requires_grad = False
        dd = DomainDiscriminator(in_dim=8, hidden=8)
        target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=1e-3)
        dd_optimizer = torch.optim.AdamW(dd.parameters(), lr=1e-3)

        score_mask = torch.tensor([[False, True], [False, True]])
        zeros_y = torch.zeros(2, 2, 1)
        source_loader = Chunks(
            [
                (torch.ones(2, 2, 1), zeros_y, score_mask, torch.tensor([True, True])),
                (torch.full((2, 2, 1), 2.0), zeros_y, score_mask, torch.tensor([False, True])),
            ]
        )
        target_loader = Chunks(
            [
                (torch.full((2, 2, 1), 3.0), zeros_y, score_mask, torch.tensor([True, True])),
                (torch.full((2, 2, 1), 4.0), zeros_y, score_mask, torch.tensor([True, False])),
            ]
        )

        source_hx_inputs = []
        target_hx_inputs = []
        source_forward = source_model.cfc.forward
        target_forward = target_model.cfc.forward

        def record_source(x, hx=None, *args, **kwargs):
            source_hx_inputs.append(hx.detach().clone())
            return source_forward(x, *args, hx=hx, **kwargs)

        def record_target(x, hx=None, *args, **kwargs):
            target_hx_inputs.append(hx.detach().clone())
            return target_forward(x, *args, hx=hx, **kwargs)

        source_model.cfc.forward = record_source
        target_model.cfc.forward = record_target

        metrics = paper_protocol._atl_training_epoch(
            source_model=source_model,
            target_model=target_model,
            dd=dd,
            source_loader=source_loader,
            target_loader=target_loader,
            target_optimizer=target_optimizer,
            dd_optimizer=dd_optimizer,
            loss_fn=torch.nn.MSELoss(),
            subject_weight=1.0,
            device=torch.device("cpu"),
            gradient_clip_norm=1.0,
        )

        self.assertEqual(len(source_hx_inputs), 2)
        self.assertEqual(len(target_hx_inputs), 2)
        self.assertTrue(torch.count_nonzero(source_hx_inputs[1][0]).item() > 0)
        self.assertEqual(torch.count_nonzero(source_hx_inputs[1][1]).item(), 0)
        self.assertEqual(torch.count_nonzero(target_hx_inputs[1][0]).item(), 0)
        self.assertTrue(torch.count_nonzero(target_hx_inputs[1][1]).item() > 0)
        self.assertFalse(source_hx_inputs[1].requires_grad)
        self.assertFalse(target_hx_inputs[1].requires_grad)
        self.assertEqual(metrics["n_source_domain_frames"], 4.0)
        self.assertEqual(metrics["n_target_domain_frames"], 4.0)
        self.assertEqual(metrics["n_supervised_frames"], 4.0)


if __name__ == "__main__":
    unittest.main()
