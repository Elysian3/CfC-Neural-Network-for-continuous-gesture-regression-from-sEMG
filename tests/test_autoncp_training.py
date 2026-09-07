"""CPU contracts for the AutoNCP CfC regression family."""

import copy
import io
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

import run_db2_paper_cfc_finetune as paper_protocol
from train import (
    AutoNCPCfCLinearRegressor,
    CfCTrainingConfig,
    SequenceSplit,
    build_best_cfc_config,
    build_cfc_regressor,
    fit_feature_normalizer,
    fit_target_normalizer,
)


class AutoNCPCfCTrainingTests(unittest.TestCase):
    """Keep AutoNCP behavior covered without requiring DB2 recordings or a GPU."""

    input_dim = 3
    output_dim = 10
    hidden_units = 16
    seq_len = 3

    def _split(self, rows: int = 6, *, query_nan: bool = False) -> SequenceSplit:
        rng = np.random.default_rng(22222)
        frames = rng.normal(size=(rows + self.seq_len - 1, self.input_dim)).astype(np.float32)
        x = np.stack([frames[index : index + self.seq_len] for index in range(rows)])
        y = rng.normal(size=(rows, self.output_dim)).astype(np.float32)
        if query_nan:
            y[1::2] = np.nan  # Query labels must never enter the masked support loss.
        indices = np.arange(rows, dtype=np.int64)
        return SequenceSplit(
            x=x,
            y=y,
            time_s=indices.astype(np.float32),
            alignment_indices=indices,
            recording_ids=np.asarray(["synthetic.mat"] * rows),
            feature_names=["f1", "f2", "f3"],
            target_names=[f"joint_{index}" for index in range(self.output_dim)],
            action_labels=np.asarray([1, 1, 2, 2, 3, 3][:rows], dtype=np.int16),
            repetition_labels=np.asarray([1, 2, 1, 2, 1, 2][:rows], dtype=np.int16),
            stream_start_flags=np.asarray([True] + [False] * (rows - 1)),
            expected_alignment_step=1,
        )

    def _config(self, **overrides) -> CfCTrainingConfig:
        values = {
            "emg_channels": (1, 2, 3), "feature_order": ("rms",),
            "hidden_units": self.hidden_units, "model_family": "autoncp_cfc_linear",
            "seq_len": self.seq_len, "batch_size": 2, "max_epochs": 2,
            "learning_rate": 1e-3, "weight_decay": 0.0, "cfc_dropout": 0.0,
            "gradient_clip_norm": 1.0, "random_seed": 7, "device": "cpu",
        }
        values.update(overrides)
        return build_best_cfc_config(**values)

    def test_cli_default_is_dense_and_opt_in_selects_autoncp(self) -> None:
        with patch.object(sys, "argv", ["paper.py", "--target-offset-samples", "0"]):
            self.assertEqual(paper_protocol.parse_args().model_family, "dense_cfc_linear")
        with patch.object(
            sys,
            "argv",
            ["paper.py", "--target-offset-samples", "0", "--model-family", "autoncp_cfc_linear"],
        ):
            self.assertEqual(paper_protocol.parse_args().model_family, "autoncp_cfc_linear")

    def test_motor_output_is_ten_wide_while_carried_state_is_full_wiring_state(self) -> None:
        model = build_cfc_regressor(
            input_dim=self.input_dim, output_dim=self.output_dim,
            hidden_units=self.hidden_units, model_family="autoncp_cfc_linear",
        ).eval()
        self.assertIsInstance(model, AutoNCPCfCLinearRegressor)
        self.assertEqual(model.cfc.output_size, self.output_dim)  # AutoNCP motor neurons.
        self.assertGreater(model.cfc.state_size, model.cfc.output_size)
        self.assertLess(sum(p.numel() for p in model.parameters()), 100_000)
        x = torch.randn(2, self.seq_len, self.input_dim)
        prediction, features = model.forward_with_features(x)
        carried_prediction, carried = model.forward_with_state(x)
        with torch.no_grad():
            _, expected_carried = model.cfc(x)
        self.assertEqual(prediction.shape, (2, self.output_dim))
        self.assertEqual(features.shape, (2, self.output_dim))
        self.assertEqual(carried.shape, (2, model.cfc.state_size))
        self.assertFalse(carried.requires_grad)
        torch.testing.assert_close(carried_prediction, prediction)
        torch.testing.assert_close(carried, expected_carried)

    def test_chunked_recurrence_matches_full_sequence_and_fixed_masks_are_not_trainable(self) -> None:
        torch.manual_seed(11)
        model = build_cfc_regressor(
            input_dim=self.input_dim, output_dim=self.output_dim,
            hidden_units=self.hidden_units, model_family="autoncp_cfc_linear",
        ).eval()
        x = torch.randn(2, 6, self.input_dim)
        with torch.no_grad():
            full_sequence, full_state = model.cfc(x)
            first_sequence, state = model.cfc(x[:, :3])
            second_sequence, chunked_state = model.cfc(x[:, 3:], state)
        torch.testing.assert_close(torch.cat([first_sequence, second_sequence], dim=1), full_sequence)
        torch.testing.assert_close(chunked_state, full_state)
        full_prediction = model(x)
        _, helper_state = model.forward_with_state(x[:, :3])
        chunked_prediction, helper_state = model.forward_with_state(x[:, 3:], helper_state)
        torch.testing.assert_close(chunked_prediction, full_prediction)
        torch.testing.assert_close(helper_state, full_state)

        prediction = model(x).sum()
        prediction.backward()
        gate_gradients = [
            parameter.grad for name, parameter in model.named_parameters()
            if ("ff" in name.lower() or "time_" in name.lower()) and parameter.requires_grad
        ]
        self.assertTrue(gate_gradients, "CfC candidate/time gates must remain trainable")
        self.assertTrue(all(gradient is not None for gradient in gate_gradients))
        masks = [(name, parameter) for name, parameter in model.named_parameters() if "mask" in name.lower()]
        self.assertTrue(masks, "AutoNCP wiring must retain fixed connectivity masks")
        self.assertTrue(all(not parameter.requires_grad for _, parameter in masks))

    def test_stateful_pretrain_and_atl_ignore_nan_query_labels_then_head_only_freezes_backbone(self) -> None:
        source = self._split()
        config = self._config()
        x_stats = fit_feature_normalizer(source.x.reshape(-1, self.input_dim), method="zscore")
        y_stats = fit_target_normalizer(source.y, method="zscore")
        pretrain = paper_protocol.train_model(
            supervised_split=source, config=config, x_stats=x_stats, y_stats=y_stats,
            device=torch.device("cpu"), augment_prob=0.0, stateful=True, gpu_resident=True,
        )
        self.assertEqual(len(pretrain["history"]), config.max_epochs)
        self.assertTrue(all(np.isfinite(item["train_loss"]) for item in pretrain["history"]))

        support = self._split(query_nan=True)
        source_before_atl = copy.deepcopy(pretrain["model"].state_dict())
        support_mask = np.asarray([True, False, True, False, True, False])
        adapted, atl_history, audit = paper_protocol.fine_tune_head(
            model=pretrain["model"], support_split=support, source_split=source,
            support_score_mask=support_mask, source_score_mask=np.ones(len(source.x), dtype=bool),
            config=config, y_stats=y_stats, device=torch.device("cpu"), learning_rate=1e-3,
            epochs=1, enable_atl=True, cfc_atl_lr=1e-3, dd_lr=1e-3,
        )
        self.assertEqual(len(atl_history), 1)
        self.assertEqual(audit["feature_dim"], self.output_dim)  # DD consumes motor output, not state_size.
        self.assertEqual(audit["hidden_units"], adapted.cfc.state_size)
        self.assertTrue(all(np.isfinite(value) for value in atl_history[0].values()))
        self.assertTrue(any(
            not torch.equal(source_before_atl[name], value)
            for name, value in adapted.state_dict().items() if name.startswith("cfc.")
        ))
        for name, value in adapted.state_dict().items():
            if "mask" in name.lower():
                torch.testing.assert_close(value, source_before_atl[name])

        cfc_before_head_only = {name: value.clone() for name, value in adapted.state_dict().items() if name.startswith("cfc.")}
        head_only, history, head_audit = paper_protocol.fine_tune_head(
            model=adapted, support_split=source, config=config, y_stats=y_stats,
            device=torch.device("cpu"), learning_rate=1e-3, epochs=1,
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(head_audit["trainable_param_names"], ["head.weight", "head.bias"])
        self.assertEqual(head_audit["trainable_param_count"], sum(p.numel() for p in head_only.head.parameters()))
        for name, before in cfc_before_head_only.items():
            torch.testing.assert_close(head_only.state_dict()[name], before)
        self.assertFalse(any("mask" in name.lower() for name in head_audit["trainable_param_names"]))

    def test_state_dict_roundtrip_and_family_mismatch_resume_are_rejected(self) -> None:
        model = build_cfc_regressor(
            input_dim=self.input_dim, output_dim=self.output_dim,
            hidden_units=self.hidden_units, model_family="autoncp_cfc_linear",
        ).eval()
        restored = build_cfc_regressor(
            input_dim=self.input_dim, output_dim=self.output_dim,
            hidden_units=self.hidden_units, model_family="autoncp_cfc_linear",
        ).eval()
        checkpoint = {"config": {"model_family": "autoncp_cfc_linear"}, "state_dict": model.state_dict()}
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=False)
        self.assertEqual(loaded["config"]["model_family"], "autoncp_cfc_linear")
        restored.load_state_dict(loaded["state_dict"])
        x = torch.randn(2, self.seq_len, self.input_dim)
        torch.testing.assert_close(restored(x), model(x))

        checkpoint = {"hidden_units": self.hidden_units, "model_family": "dense_cfc_linear", "cfc_dropout": 0.0}
        with self.assertRaisesRegex(ValueError, "model_family"):
            paper_protocol.validate_resume_architecture(checkpoint, self._config())
