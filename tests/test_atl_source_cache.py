import pathlib
import sys
import unittest

import numpy as np
import torch
from torch import nn


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "src" / "deep learning"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from train import GpuResidentStatefulBatches, SegmentIndex, SequenceSplit


class _CountingCfC(nn.Module):
    state_size = 2
    output_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, frames: torch.Tensor, hx: torch.Tensor):
        self.calls += 1
        self.batch_sizes.append(int(frames.shape[0]))
        outputs = []
        for frame in frames.unbind(dim=1):
            hx = torch.tanh(hx + frame[:, :2])
            outputs.append(hx)
        return torch.stack(outputs, dim=1), hx


def _split() -> SequenceSplit:
    sequence_count, seq_len, features = 9, 3, 3
    frames = np.arange((sequence_count + seq_len - 1) * features, dtype=np.float32).reshape(
        sequence_count + seq_len - 1, features
    ) / 10.0
    x = np.stack([frames[index : index + seq_len] for index in range(sequence_count)])
    y = np.arange(sequence_count, dtype=np.float32).reshape(-1, 1)
    y[[1, 6]] = np.nan  # Context-only sequences retain their unscored NaN labels.
    return SequenceSplit(
        x=x,
        y=y,
        time_s=np.arange(sequence_count, dtype=np.float32),
        alignment_indices=np.arange(sequence_count, dtype=np.int64),
        recording_ids=np.array(["a"] * 3 + ["b"] * 2 + ["c"] * 4),
        feature_names=["f0", "f1", "f2"],
        target_names=["target"],
    )


def _loader(seed: int) -> GpuResidentStatefulBatches:
    split = _split()
    return GpuResidentStatefulBatches(
        split.x,
        split.y,
        SegmentIndex(split),
        batch_size=2,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(seed),
        sequence_score_mask=np.array([True, False, True, True, True, True, False, True, True]),
    )


class FrozenSourceFeatureCacheTests(unittest.TestCase):
    def test_cached_source_frames_match_eager_scored_outputs_and_rng(self) -> None:
        eager = _loader(77)
        cached = _loader(77)
        cfc = _CountingCfC().eval()
        generator_before = cached.generator.get_state().clone()
        torch.manual_seed(123)
        global_rng_before = torch.get_rng_state().clone()

        cached.cache_encoder_outputs(cfc)

        self.assertTrue(cached.encoder_outputs_cached)
        self.assertEqual(cfc.calls, 4)  # Two padded chunks for each of two segment groups.
        self.assertEqual(max(cfc.batch_sizes), cached.batch_size)
        self.assertEqual(
            sum(outputs.shape[0] for outputs in cached._cached_encoder_outputs.values()),
            15,  # Segment streams contain 5, 4, and 6 real frames; padding is not retained.
        )
        self.assertTrue(torch.equal(generator_before, cached.generator.get_state()))
        self.assertTrue(torch.equal(global_rng_before, torch.get_rng_state()))

        for _ in range(3):
            eager_hx = torch.zeros(eager.batch_size, cfc.state_size)
            eager_batches = list(eager.iter_training_batches())
            cached_batches = list(cached.iter_training_batches())
            self.assertEqual(len(eager_batches), len(cached_batches))
            for eager_batch, cached_batch in zip(eager_batches, cached_batches):
                eager_x, eager_y, eager_mask, eager_reset, eager_first, eager_count = eager_batch
                cached_x, cached_y, cached_mask, cached_reset, cached_first, cached_count = cached_batch
                if eager_first:
                    eager_hx.zero_()
                eager_hx[eager_reset] = 0
                eager_outputs, eager_hx = cfc(eager_x, eager_hx)
                eager_hx = eager_hx.detach()
                torch.testing.assert_close(cached_x[cached_mask], eager_outputs[eager_mask])
                torch.testing.assert_close(cached_y, eager_y, equal_nan=True)
                self.assertTrue(torch.equal(cached_mask, eager_mask))
                self.assertTrue(torch.equal(cached_reset, eager_reset))
                self.assertEqual(cached_first, eager_first)
                self.assertEqual(cached_count, eager_count)

        calls_after_cache = cfc.calls
        list(cached.iter_training_batches())
        list(eager.iter_training_batches())
        self.assertEqual(cfc.calls, calls_after_cache)
        self.assertTrue(torch.equal(eager.generator.get_state(), cached.generator.get_state()))

    def test_cache_requires_frozen_eval_encoder(self) -> None:
        loader = _loader(1)
        encoder = _CountingCfC()
        with self.assertRaisesRegex(ValueError, "eval"):
            loader.cache_encoder_outputs(encoder)
        encoder.eval()
        encoder.register_parameter("trainable", nn.Parameter(torch.ones(1)))
        with self.assertRaisesRegex(ValueError, "frozen"):
            loader.cache_encoder_outputs(encoder)


if __name__ == "__main__":
    unittest.main()
