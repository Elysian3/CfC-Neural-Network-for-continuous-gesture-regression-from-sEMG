"""ATL update parity; opt into the CUDA integration check with ANTIKYTHERA_TEST_CUDA=1."""

import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "deep learning"))
import run_db2_paper_cfc_finetune as paper
from train import DomainDiscriminator, SequenceSplit, build_best_cfc_config, build_cfc_regressor


class Chunks:
    def __init__(self, chunks):
        self.chunks = chunks
        self.batch_size = chunks[0][0].shape[0]

    def iter_training_batches(self):
        for chunk in self.chunks:
            yield (*chunk, bool(chunk[3].any()), int(chunk[2].sum()))


def synthetic_split():
    rng = np.random.default_rng(4)
    frames = rng.normal(size=(20, 3)).astype(np.float32)
    rows = 17
    indices = np.arange(rows)
    return SequenceSplit(
        x=np.stack([frames[i:i + 3] for i in indices]),
        y=rng.normal(size=(rows, 2)).astype(np.float32),
        time_s=indices.astype(np.float32), alignment_indices=indices,
        recording_ids=np.asarray(["a"] * 9 + ["b"] * 8),
        feature_names=["a", "b", "c"], target_names=["j1", "j2"],
        stream_start_flags=np.asarray([i in (0, 9) for i in indices]),
        expected_alignment_step=1,
    )


class ATLAccelerationTests(unittest.TestCase):
    def test_reused_forward_matches_original_alternating_parameter_updates(self):
        for family in ("dense_cfc_linear", "autoncp_cfc_linear"):
            with self.subTest(family=family):
                torch.manual_seed(8)
                source = build_cfc_regressor(
                    input_dim=3, output_dim=2, hidden_units=16,
                    model_family=family, cfc_dropout=0.2,
                ).eval().requires_grad_(False)
                reference = copy.deepcopy(source).train()
                # Keep AutoNCP masks fixed, restoring only the trainable weights.
                for name, parameter in reference.named_parameters():
                    parameter.requires_grad_("mask" not in name)
                actual = copy.deepcopy(reference)
                ref_dd = DomainDiscriminator(reference.cfc.output_size, hidden=8)
                actual_dd = copy.deepcopy(ref_dd)
                x = torch.randn(2, 3, 3)
                y = torch.randn(2, 3, 2)
                mask = torch.tensor([[False, True, True], [False, True, False]])
                y[~mask] = float("nan")
                reset = torch.ones(2, dtype=torch.bool)
                ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3)
                ref_dd_opt = torch.optim.AdamW(ref_dd.parameters(), lr=1e-3)
                actual_opt = torch.optim.AdamW(actual.parameters(), lr=1e-3)
                actual_dd_opt = torch.optim.AdamW(actual_dd.parameters(), lr=1e-3)
                rng_state = torch.get_rng_state()

                # Original algorithm: frozen source, separate detached target DD
                # pass, then a second target pass against the updated discriminator.
                with torch.no_grad():
                    fs = source.cfc(x)[0][mask]
                    ft = reference.cfc(x * 2)[0][mask]
                domains = ref_dd(torch.cat((fs, ft)))
                dd_loss = -(torch.log(domains[:3] + 1e-8).mean()
                            + torch.log(1 - domains[3:] + 1e-8).mean())
                dd_loss.backward()
                torch.nn.utils.clip_grad_norm_(ref_dd.parameters(), 1.0)
                ref_dd_opt.step()
                sequence = reference.cfc(x * 2)[0]
                pred = reference.head(reference.dropout(sequence))[mask]
                ref_dd.eval().requires_grad_(False)
                map_loss = -torch.log(ref_dd(sequence[mask]) + 1e-8).mean()
                reg_loss = 0.7 * torch.nn.functional.mse_loss(pred, y[mask])
                (map_loss + reg_loss).backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
                ref_opt.step()

                torch.set_rng_state(rng_state)
                with patch.object(actual.cfc, "forward", wraps=actual.cfc.forward) as forward:
                    metrics = paper._atl_training_epoch(
                        source_model=source, target_model=actual, dd=actual_dd,
                        source_loader=Chunks([(x, y, mask, reset)]),
                        target_loader=Chunks([(x * 2, y, mask, reset)]),
                        target_optimizer=actual_opt, dd_optimizer=actual_dd_opt,
                        loss_fn=torch.nn.MSELoss(), subject_weight=0.7,
                        device=torch.device("cpu"), gradient_clip_norm=1.0,
                    )
                self.assertEqual(forward.call_count, 1)
                for name, expected in reference.state_dict().items():
                    torch.testing.assert_close(actual.state_dict()[name], expected)
                for name, expected in ref_dd.state_dict().items():
                    torch.testing.assert_close(actual_dd.state_dict()[name], expected)
                for key, expected in zip(("L_DD", "L_mapping", "L_subject"), (dd_loss, map_loss, reg_loss)):
                    self.assertAlmostEqual(metrics[key], float(expected.detach()), places=6)

    def test_atl_batch_control_uses_cpu_metadata(self):
        split = synthetic_split()
        model = build_cfc_regressor(input_dim=3, output_dim=2, hidden_units=16)
        source = copy.deepcopy(model).eval().requires_grad_(False)
        from train import GpuResidentStatefulBatches, SegmentIndex
        loader = GpuResidentStatefulBatches(
            split.x, split.y, SegmentIndex(split), batch_size=2, device=torch.device("cpu"),
        )
        dd = DomainDiscriminator(16, hidden=8)
        scalar_item = torch.Tensor.item
        calls = []

        def record_item(tensor, *args):
            calls.append(tensor.dtype)
            return scalar_item(tensor, *args)

        # SGD has no optimizer scalar reads; only the five final metric reads remain.
        with patch.object(torch.Tensor, "item", record_item):
            paper._atl_training_epoch(
                source_model=source, target_model=model, dd=dd,
                source_loader=loader, target_loader=loader,
                target_optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
                dd_optimizer=torch.optim.SGD(dd.parameters(), lr=1e-3),
                loss_fn=torch.nn.MSELoss(), subject_weight=1.0,
                device=torch.device("cpu"), gradient_clip_norm=None,
            )
        self.assertEqual(len(calls), 5)

    @unittest.skipUnless(os.environ.get("ANTIKYTHERA_TEST_CUDA") == "1", "opt-in CUDA check")
    def test_cuda_graph_matches_eager_with_context_padding_and_multiple_epochs(self):
        self.assertTrue(torch.cuda.is_available())
        split = synthetic_split()
        mask = np.zeros(len(split.x), dtype=bool)
        mask[[0, 8, 9, 16]] = True
        support = copy.deepcopy(split)
        support.y[~mask] = np.nan
        for family in ("dense_cfc_linear", "autoncp_cfc_linear"):
            with self.subTest(family=family):
                torch.manual_seed(10)
                model = build_cfc_regressor(
                    input_dim=3, output_dim=2, hidden_units=16, model_family=family,
                ).cuda()
                results = []
                for graph in (False, True):
                    torch.manual_seed(21)
                    config = build_best_cfc_config(
                        seq_len=3, batch_size=2, hidden_units=16,
                        model_family=family, cuda_graph=graph,
                    )
                    results.append(paper.fine_tune_head(
                        model=model, support_split=support, source_split=split,
                        support_score_mask=mask, config=config, y_stats={},
                        device=torch.device("cuda"), learning_rate=1e-3,
                        epochs=2, enable_atl=True,
                    ))
                eager, eager_history, _ = results[0]
                graphed, graph_history, audit = results[1]
                self.assertTrue(audit["cuda_graph_enabled"])
                for name, expected in eager.state_dict().items():
                    torch.testing.assert_close(graphed.state_dict()[name], expected, atol=2e-5, rtol=2e-4)
                for actual, expected in zip(graph_history, eager_history):
                    np.testing.assert_allclose(list(actual.values()), list(expected.values()), atol=2e-5, rtol=2e-4)
                # Returned model must support arbitrary evaluation batch shapes.
                self.assertEqual(graphed(torch.zeros(1, 5, 3, device="cuda")).shape, (1, 2))

        with (
            patch.object(torch.cuda, "make_graphed_callables", side_effect=RuntimeError("capture unavailable")),
            self.assertWarnsRegex(UserWarning, "falling back to eager"),
        ):
            fallback, _, audit = paper.fine_tune_head(
                model=model, support_split=support, source_split=split,
                support_score_mask=mask, config=config, y_stats={},
                device=torch.device("cuda"), learning_rate=1e-3,
                epochs=1, enable_atl=True,
            )
        self.assertFalse(audit["cuda_graph_enabled"])
        self.assertEqual(fallback(torch.zeros(1, 5, 3, device="cuda")).shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
