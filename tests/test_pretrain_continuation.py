"""Continuation keeps source supervision, normalizers and parent checkpoint intact."""

import copy
from contextlib import ExitStack
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "deep learning"))
import run_db2_paper_cfc_finetune as paper
from train import build_best_cfc_config, build_cfc_regressor
from tests import test_cfc_training as fixtures


class PretrainContinuationTests(unittest.TestCase):
    def test_cli_continuation_is_explicit_and_weights_follow_j10_order(self):
        with patch.object(sys, "argv", ["paper.py", "--target-offset-samples", "0"]):
            args = paper.parse_args()
        self.assertEqual(args.additional_pretrain_epochs, 0)
        self.assertEqual(args.pretrain_target_weights, ())
        with patch.object(sys, "argv", [
            "paper.py", "--target-offset-samples", "0", "--additional-pretrain-epochs", "10",
            "--pretrain-target-weights", "2,2,1,1,1,1,1,1,1,1",
        ]):
            args = paper.parse_args()
        self.assertEqual(args.additional_pretrain_epochs, 10)
        self.assertEqual(args.pretrain_target_weights, (2, 2, 1, 1, 1, 1, 1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, "requires --resume-pretrain"):
            paper.run_protocol(args)

    def test_weighted_stateful_training_updates_loaded_model_and_ignores_context_labels(self):
        split = fixtures.RunProtocolOrchestrationTests._split("S2", rows=4)
        split.y = np.repeat(split.y, 2, axis=1)
        split.target_names = ["a", "b"]
        mask = np.array([True, False, True, False])
        supervised = paper.copy_split_by_indices(split, np.flatnonzero(mask))
        split.y[~mask] = np.nan
        model = build_cfc_regressor(input_dim=1, output_dim=2, hidden_units=8, model_family="autoncp_cfc_linear")
        model.eval()
        before = copy.deepcopy(model.state_dict())
        config = build_best_cfc_config(
            seq_len=1, max_epochs=2, hidden_units=8, model_family="autoncp_cfc_linear",
            batch_size=2, learning_rate=1e-3, pretrain_target_weights=(2.0, 1.0),
        )
        original_optimizer = torch.optim.AdamW

        def training_optimizer(*args, **kwargs):
            self.assertTrue(model.cfc.training, "enter training mode before optimizer/graph setup")
            return original_optimizer(*args, **kwargs)

        with (
            patch.object(paper, "build_cfc_regressor", side_effect=AssertionError("must reuse loaded model")),
            patch.object(torch.optim, "AdamW", side_effect=training_optimizer),
        ):
            result = paper.train_model(
                supervised_split=supervised, config=config,
                x_stats={"method": "zscore", "mean": np.zeros(1), "std": np.ones(1)},
                y_stats={"method": "zscore", "mean": np.zeros(2), "std": np.ones(2)},
                device=torch.device("cpu"), stateful=True, gpu_resident=True,
                stateful_stream_split=split, stateful_stream_score_mask=mask,
                initial_model=model, epoch_offset=600,
            )
        self.assertIs(result["model"], model)
        self.assertEqual([entry["epoch"] for entry in result["history"]], [601, 602])
        self.assertTrue(all(np.isfinite(entry["train_loss"]) for entry in result["history"]))
        self.assertTrue(any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()))
        for name, value in model.state_dict().items():
            if "mask" in name:
                torch.testing.assert_close(value, before[name])

    def test_continuation_preserves_parent_history_and_records_new_objective(self):
        source = fixtures.RunProtocolOrchestrationTests._split("S2")
        target = fixtures.RunProtocolOrchestrationTests._split("S1")
        for split in (source, target):
            split.y = np.repeat(split.y, 10, axis=1)
            split.target_names = [f"joint{i}" for i in range(10)]
        support = paper.copy_split_by_indices(target, np.array([0, 1]))
        query = paper.copy_split_by_indices(target, np.array([2, 3]))
        source_model = torch.nn.Linear(1, 10)
        with torch.no_grad():
            source_model.weight.fill_(0.25)

        def enter_common(stack):
            stack.enter_context(patch.object(paper, "discover_target_files", return_value={"S1": [], "S2": []}))
            stack.enter_context(patch.object(paper, "load_subject_splits", side_effect=lambda subject, *_a, **_k:
                (target, target) if subject == "S1" else (source, source)))
            stack.enter_context(patch.object(paper, "select_repetition_split", return_value=(support, query, {})))
            stack.enter_context(patch.object(paper, "normalize_sequence_inputs", side_effect=lambda split, **_k: split))
            stack.enter_context(patch.object(paper, "evaluate_split", return_value={"metrics": {}, "per_action_metrics": {}}))
            stack.enter_context(patch.object(paper, "evaluate_chain", return_value={"metrics": {}, "h_norm_stats": {}}))
            stack.enter_context(patch.object(paper, "fine_tune_head", side_effect=AssertionError("source only")))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = fixtures.RunProtocolOrchestrationTests._args(root / "base", enable_atl=False)
            args.skip_fine_tune = True
            with ExitStack() as stack:
                enter_common(stack)
                stack.enter_context(patch.object(paper, "train_model", return_value={
                    "model": source_model, "history": [{"epoch": 600., "train_loss": .1}],
                }))
                paper.run_protocol(args)
            checkpoint = args.output_dir / "checkpoints/S1_dense_cfc_pretrain.pt"
            parent_bytes = checkpoint.read_bytes()
            args.output_dir = root / "weighted"
            args.resume_pretrain = checkpoint
            args.additional_pretrain_epochs = 10
            args.pretrain_target_weights = (2., 2., 1., 1., 1., 1., 1., 1., 1., 1.)

            def continue_training(**kwargs):
                self.assertEqual(kwargs["config"].max_epochs, 10)
                self.assertEqual(kwargs["config"].pretrain_target_weights, args.pretrain_target_weights)
                self.assertEqual(kwargs["epoch_offset"], 600)
                torch.testing.assert_close(kwargs["initial_model"].weight, source_model.weight)
                self.assertTrue(all(row == "S2.mat" for row in kwargs["supervised_split"].recording_ids))
                np.testing.assert_array_equal(kwargs["stateful_stream_score_mask"], np.ones(4, dtype=bool))
                with torch.no_grad():
                    kwargs["initial_model"].weight.add_(0.1)
                return {"model": kwargs["initial_model"], "history": [
                    {"epoch": float(i), "train_loss": .09} for i in range(601, 611)
                ]}

            with ExitStack() as stack:
                enter_common(stack)
                stack.enter_context(patch.object(paper, "build_cfc_regressor", side_effect=lambda **_k: torch.nn.Linear(1, 10)))
                stack.enter_context(patch.object(paper, "infer_checkpoint_output_dim", return_value=10))
                stack.enter_context(patch.object(paper, "train_model", side_effect=continue_training))
                summary = paper.run_protocol(args)
            self.assertEqual(checkpoint.read_bytes(), parent_bytes)
            self.assertEqual(len(summary["pretrain_history"]), 11)
            self.assertEqual(summary["config"]["max_epochs"], 610)
            self.assertEqual(summary["config"]["pretrain_target_weights"], list(args.pretrain_target_weights))
            stage = summary["pretrain_provenance"]["continuations"][-1]
            self.assertEqual((stage["epoch_start"], stage["epoch_end"]), (601, 610))
            self.assertEqual(stage["optimizer_state"], "reinitialized_adamw")
            self.assertEqual(stage["loss"], "mean_normalized_weighted_mse")
            saved = torch.load(args.output_dir / "checkpoints/S1_dense_cfc_pretrain.pt", weights_only=False)
            original = torch.load(io.BytesIO(parent_bytes), weights_only=False)
            self.assertEqual(saved["resume_integrity"], original["resume_integrity"])
            torch.testing.assert_close(saved["model_state_dict"]["weight"], source_model.weight + 0.1)


if __name__ == "__main__":
    unittest.main()
