"""Synthetic-data tests for the mislabeled-target ATL control experiment."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "src" / "deep learning"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from atl_mislabel_ab import (
    _phase_median,
    _segment_runs,
    build_index_labels,
    build_template_labels,
)
from train import SequenceSplit


def _split(x, y, actions, reps, recording="f.mat"):
    n = len(y)
    return SequenceSplit(
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        time_s=np.arange(n, dtype=np.float32) * 0.05,
        alignment_indices=np.arange(n, dtype=np.int64),
        recording_ids=np.asarray([recording] * n),
        feature_names=["f"],
        target_names=["t1", "t2"],
        action_labels=np.asarray(actions, dtype=np.int16),
        repetition_labels=np.asarray(reps, dtype=np.int16),
    )


class SegmentRunsTests(unittest.TestCase):
    def test_runs_split_on_action_and_repetition_change(self) -> None:
        labels = np.array([1, 1, 2, 2, 2, 1], dtype=np.int16)
        reps = np.array([1, 1, 1, 1, 1, 2], dtype=np.int16)
        runs = _segment_runs(labels, reps)
        self.assertEqual(runs, [(0, 2, 1, 1), (2, 5, 2, 1), (5, 6, 1, 2)])

    def test_empty_input_yields_no_runs(self) -> None:
        self.assertEqual(_segment_runs(np.array([], dtype=np.int16), np.array([], dtype=np.int16)), [])


class PhaseMedianTests(unittest.TestCase):
    def test_single_trajectory_is_resampled_exactly(self) -> None:
        # A 2x2 straight line resampled to 3 points keeps its values exactly.
        traj = np.array([[0.0, 10.0], [4.0, 6.0]], dtype=np.float64)
        out = _phase_median([traj], 3)
        np.testing.assert_allclose(out, [[0.0, 10.0], [2.0, 8.0], [4.0, 6.0]])

    def test_median_across_two_subjects_ignores_extreme_subject(self) -> None:
        a = np.zeros((5, 1), dtype=np.float64)
        b = np.full((5, 1), 10.0, dtype=np.float64)
        c = np.full((5, 1), 100.0, dtype=np.float64)
        out = _phase_median([a, b, c], 5)
        np.testing.assert_allclose(out, np.full((5, 1), 10.0))

    def test_out_len_one_uses_mean_then_median(self) -> None:
        out = _phase_median([np.array([[2.0], [4.0]]), np.array([[10.0], [20.0]])], 1)
        np.testing.assert_allclose(out, [[9.0]])  # means 3 and 15, median 9


class TemplateLabelTests(unittest.TestCase):
    def test_template_has_target_shape_and_uses_source_subjects_only(self) -> None:
        # Source: two subjects, same action 6, one segment each. Target: one
        # subject with the same action but a different trajectory length.
        src = _split(
            x=np.zeros((6, 2, 1)),
            y=np.concatenate([np.arange(6).reshape(-1, 2), np.arange(6).reshape(-1, 2) * 10.0]),
            actions=[6] * 3 + [6] * 3,
            reps=[1] * 3 + [2] * 3,
        )
        tgt = _split(x=np.zeros((5, 2, 1)), y=np.zeros((5, 2)), actions=[6] * 5, reps=[1] * 5)
        out = build_template_labels(tgt, src)
        self.assertEqual(out.shape, (5, 2))
        # Phase-aligned median of the two straight lines ([0,1]->[4,5] and
        # [0,10]->[40,50]), resampled to 5 phases, per-dimension median.
        np.testing.assert_allclose(out, [[0.0, 5.5], [5.5, 11.0], [11.0, 16.5],
                                          [16.5, 22.0], [22.0, 27.5]])

    def test_template_requires_action_labels(self) -> None:
        split = _split(x=np.zeros((3, 2, 1)), y=np.zeros((3, 2)), actions=[1] * 3, reps=[1] * 3)
        unlabeled = SequenceSplit(
            x=split.x, y=split.y, time_s=split.time_s, alignment_indices=split.alignment_indices,
            recording_ids=split.recording_ids, feature_names=split.feature_names,
            target_names=split.target_names, action_labels=None, repetition_labels=None,
        )
        with self.assertRaises(ValueError):
            build_template_labels(unlabeled, split)


class IndexLabelTests(unittest.TestCase):
    def test_index_labels_tile_donor_in_row_order(self) -> None:
        tgt = _split(x=np.zeros((5, 2, 1)), y=np.zeros((5, 3)), actions=[1] * 5, reps=[1] * 5)
        donor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        out = build_index_labels(tgt, donor)
        np.testing.assert_allclose(out, [[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6], [1, 2, 3]])

    def test_index_labels_reject_empty_donor(self) -> None:
        tgt = _split(x=np.zeros((2, 2, 1)), y=np.zeros((2, 2)), actions=[1] * 2, reps=[1] * 2)
        with self.assertRaises(ValueError):
            build_index_labels(tgt, np.empty((0, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
