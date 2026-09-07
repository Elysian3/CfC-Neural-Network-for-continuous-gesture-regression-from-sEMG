"""Synthetic contracts for the DB2 pre-normalization recording cache."""

import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "src" / "deep learning"
DATAFLOW_DIR = REPO_ROOT / "src" / "dataflow"
for directory in (TRAINING_DIR, DATAFLOW_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import datapreprocess
import run_db2_paper_cfc_finetune as paper_protocol


class RecordingCacheTests(unittest.TestCase):
    def _config(self, root: pathlib.Path, **overrides):
        values = dict(
            db2_dir=root, target_columns=(0,), target_mapping=None,
            emg_channels=(1,), window_ms=2.0, stride_ms=1.0,
            target_offset_samples=0, feature_order=("rms",), seq_len=2, seq_stride=1,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _data(samples: int = 40) -> dict:
        return {
            "emg": np.arange(samples, dtype=np.float32)[:, None],
            "glove": (np.arange(samples, dtype=np.float32) * 2)[:, None],
            "restimulus": np.asarray([1] * (samples // 2) + [2] * (samples - samples // 2)),
            "rerepetition": np.ones(samples, dtype=np.int16),
        }

    def test_load_data_selects_requested_mat_variables_and_default_contract(self) -> None:
        raw = {"emg": np.zeros((2, 1)), "glove": np.ones((2, 1))}
        with patch.object(datapreprocess.sio, "loadmat", return_value=raw) as loadmat:
            selected = datapreprocess.load_data("synthetic.mat", variable_names=("emg", "glove"))
        self.assertEqual(set(selected), {"emg", "glove"})
        self.assertEqual(loadmat.call_args.kwargs["variable_names"], ("emg", "glove"))

        with patch.object(datapreprocess.sio, "loadmat", return_value=raw) as loadmat:
            datapreprocess.load_data("synthetic.mat")
        self.assertIn("restimulus", loadmat.call_args.kwargs["variable_names"])

    def test_cache_hit_skips_preprocessing_and_preserves_full_and_selected_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            file_path = root / "S1_E1_A1.mat"
            file_path.write_bytes(b"source")
            cache_dir = root / "cache"
            config = self._config(root)
            patches = (
                patch.object(paper_protocol, "list_db2_files", return_value=[file_path]),
                patch.object(paper_protocol, "load_data", return_value=self._data()),
                patch.object(paper_protocol, "preprocess_emg", side_effect=lambda values, **_: values),
                patch.object(paper_protocol, "_compute_rest_thresholds", return_value=np.zeros(1, dtype=np.float32)),
            )
            with patches[0], patches[1], patches[2] as preprocess, patches[3]:
                uncached_selected, uncached_stream = paper_protocol.load_subject_splits(
                    "S1", ("E1",), config, actions=(1, 2), cache_dir=cache_dir,
                )
            self.assertEqual(preprocess.call_count, 1)
            self.assertEqual(len(list(cache_dir.glob("*.npz"))), 1)

            with (
                patch.object(paper_protocol, "list_db2_files", return_value=[file_path]),
                patch.object(paper_protocol, "load_data", side_effect=AssertionError("cache miss")),
                patch.object(paper_protocol, "preprocess_emg", side_effect=AssertionError("cache miss")),
            ):
                cached_selected, cached_stream = paper_protocol.load_subject_splits(
                    "S1", ("E1",), config, actions=(1, 2), cache_dir=cache_dir,
                )
            for before, after in ((uncached_selected, cached_selected), (uncached_stream, cached_stream)):
                np.testing.assert_array_equal(after.x, before.x)
                np.testing.assert_array_equal(after.y, before.y)
                np.testing.assert_array_equal(after.alignment_indices, before.alignment_indices)
                np.testing.assert_array_equal(after.action_labels, before.action_labels)
                np.testing.assert_array_equal(after.repetition_labels, before.repetition_labels)

            with (
                patch.object(paper_protocol, "list_db2_files", return_value=[file_path]),
                patch.object(paper_protocol, "load_data", return_value=self._data()),
                patch.object(paper_protocol, "preprocess_emg", side_effect=lambda values, **_: values),
                patch.object(paper_protocol, "_compute_rest_thresholds", return_value=np.zeros(1, dtype=np.float32)),
            ):
                no_cache_selected, no_cache_stream = paper_protocol.load_subject_splits(
                    "S1", ("E1",), config, actions=(1, 2), cache_dir=None,
                )
            for before, after in ((no_cache_selected, cached_selected), (no_cache_stream, cached_stream)):
                np.testing.assert_array_equal(after.x, before.x)
                np.testing.assert_array_equal(after.y, before.y)

    def test_cache_key_changes_for_source_and_feature_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "S1_E1_A1.mat"
            source.write_bytes(b"one")
            cache_dir = root / "cache"
            first = paper_protocol._recording_cache_path(source, self._config(root), cache_dir)
            changed_offset = paper_protocol._recording_cache_path(
                source, self._config(root, target_offset_samples=1), cache_dir,
            )
            source.write_bytes(b"changed source")
            changed_source = paper_protocol._recording_cache_path(source, self._config(root), cache_dir)
        self.assertNotEqual(first, changed_offset)
        self.assertNotEqual(first, changed_source)

    def test_corrupt_cache_warns_and_recomputes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            file_path = root / "S1_E1_A1.mat"
            file_path.write_bytes(b"source")
            cache_dir = root / "cache"
            config = self._config(root)
            cache_path = paper_protocol._recording_cache_path(file_path, config, cache_dir)
            cache_dir.mkdir()
            cache_path.write_bytes(b"not an npz")
            with (
                patch.object(paper_protocol, "list_db2_files", return_value=[file_path]),
                patch.object(paper_protocol, "load_data", return_value=self._data()),
                patch.object(paper_protocol, "preprocess_emg", side_effect=lambda values, **_: values) as preprocess,
                patch.object(paper_protocol, "_compute_rest_thresholds", return_value=np.zeros(1, dtype=np.float32)),
                self.assertWarnsRegex(UserWarning, "Ignoring unusable recording feature cache"),
            ):
                paper_protocol.load_subject_splits("S1", ("E1",), config, actions=(1, 2), cache_dir=cache_dir)
            self.assertEqual(preprocess.call_count, 1)

    def test_readable_cache_with_inconsistent_arrays_is_rejected(self) -> None:
        recording = paper_protocol.RecordingFeatures(
            recording_id="S1_E1_A1.mat", source_recording_id="S1_E1_A1.mat",
            x_windows=np.ones((4, 2), dtype=np.float32),
            y_windows=np.ones((4, 1), dtype=np.float32),
            target_alignment_indices=np.arange(4), feature_names=["a", "b"],
            target_names=["joint"], fs=2000.0,
            action_labels=np.ones(4, dtype=np.int16),
            repetition_labels=np.ones(4, dtype=np.int16),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "cache.npz"
            paper_protocol._save_recording_cache(path, recording, np.ones(4, dtype=bool))
            with np.load(path, allow_pickle=False) as cached:
                valid = dict(cached)
            malformed = {
                "x_windows": np.ones(4),
                "y_windows": np.ones((3, 1)),
                "alignment_indices": np.arange(3),
                "action_labels": np.ones(3),
                "repetition_labels": np.ones(3),
                "feature_names": np.asarray('["a"]'),
                "target_names": np.asarray('[]'),
            }
            for field, value in malformed.items():
                with self.subTest(field=field):
                    np.savez(path, **{**valid, field: value})
                    with self.assertWarnsRegex(UserWarning, "Ignoring unusable recording feature cache"):
                        self.assertIsNone(paper_protocol._load_recording_cache(path))

    def test_cache_writes_use_distinct_sibling_temp_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_path = pathlib.Path(temporary) / "recording-key.npz"
            recording = paper_protocol.RecordingFeatures(
                recording_id="synthetic.mat",
                x_windows=np.zeros((2, 1), dtype=np.float32),
                y_windows=np.zeros((2, 1), dtype=np.float32),
                target_alignment_indices=np.asarray([1, 2], dtype=np.int32),
                feature_names=["f"], target_names=["y"], fs=2000.0,
                action_labels=np.asarray([1, 1], dtype=np.int16),
                repetition_labels=np.asarray([1, 1], dtype=np.int16),
                source_recording_id="synthetic.mat",
            )
            original_replace = paper_protocol.os.replace
            with patch.object(paper_protocol.os, "replace", wraps=original_replace) as replace:
                paper_protocol._save_recording_cache(cache_path, recording, np.ones(2, dtype=bool))
                paper_protocol._save_recording_cache(cache_path, recording, np.ones(2, dtype=bool))
            temporary_sources = [pathlib.Path(call.args[0]) for call in replace.call_args_list]
            self.assertEqual(len(temporary_sources), 2)
            self.assertNotEqual(temporary_sources[0], temporary_sources[1])
            self.assertTrue(all(path.parent == cache_path.parent for path in temporary_sources))
            self.assertFalse(list(cache_path.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
