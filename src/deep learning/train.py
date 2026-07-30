from __future__ import annotations

import copy
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.utils.data as data
from ncps.torch import CfC


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATAFLOW_DIR = SCRIPT_DIR.parent / "dataflow"
if str(DATAFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(DATAFLOW_DIR))

from feature_extraction import (
    DEFAULT_MU_LAW_MU,
    STRIDE_MS,
    WIN_MS,
    _apply_array_normalizer,
    _fit_array_normalizer,
    _inverse_array_normalizer,
    apply_feature_normalizer,
    fit_feature_normalizer,
    run_feature_pipeline,
)


DEFAULT_TARGET_COLUMN = 10
DEFAULT_TARGET_OFFSET_SAMPLES = 200
DEFAULT_DB2_EMG_CHANNELS = tuple(range(1, 13))


@dataclass(frozen=True)
class CfCTrainingConfig:
    """
    Configuration for one traditional-feature CfC regression experiment.

    The defaults follow the current best whole-dataset configuration:
    - glove angle 10 as the target
    - blocked-time splitting across every available recording
    - short feature sequences (`seq_len=8`)
    - conservative CPU-friendly optimizer settings
    """

    db2_dir: Path = REPO_ROOT / "src" / "data" / "DB2"
    emg_channels: tuple[int, ...] = DEFAULT_DB2_EMG_CHANNELS
    split_strategy: str = "blocked_time"  #The mode of split of data: recor ding -> file-level; blocked_time -> continguous time block 
    train_files: tuple[str, ...] = ()
    val_files: tuple[str, ...] = ()
    test_files: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()
    target_source: str = "glove"
    target_columns: tuple[int, ...] = (DEFAULT_TARGET_COLUMN,)
    target_mapping: str | None = None
    target_mapping_source: str = "glove"
    target_mapping_version: str | None = None
    window_ms: float = WIN_MS
    stride_ms: float = STRIDE_MS
    feature_order: tuple[str, ...] = ("mav", "mavs", "wl", "zc", "ssc")
    target_offset_samples: int = DEFAULT_TARGET_OFFSET_SAMPLES
    zc_threshold: float | None = None
    ssc_threshold: float | None = None
    feature_normalization: str = "mu_law"
    target_normalization: str = "mu_law"
    mu_law_mu: float = DEFAULT_MU_LAW_MU
    blocked_train_fraction: float = 0.7
    blocked_val_fraction: float = 0.15
    blocked_test_fraction: float = 0.15
    blocked_gap_windows: int = 16 # buffer windows, for the division of data and the prevention of leakage
    seq_len: int = 8
    seq_stride: int = 1
    hidden_units: int = 64
    model_family: str = "dense_cfc_linear"
    cfc_dropout: float = 0.0
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 20
    early_stopping_patience: int = 5
    gradient_clip_norm: float = 1.0
    random_seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    plot_target_index: int = 0
    plot_max_points: int = 500 # Maximum amount of points to be printed
    max_windows_per_file: int | None = None


def build_best_cfc_config(**overrides) -> CfCTrainingConfig:
    """
    Build the current best known lightweight CfC regression configuration.

    Keeping this helper explicit prevents later scripts from scattering the
    same hyperparameter literals in multiple places.
    """

    config_values = {
        "db2_dir": REPO_ROOT / "src" / "data" / "DB2",
        "emg_channels": DEFAULT_DB2_EMG_CHANNELS,
        "split_strategy": "blocked_time",
        "target_source": "glove",
        "target_columns": (DEFAULT_TARGET_COLUMN,),
        "target_mapping": None,
        "target_mapping_source": "glove",
        "target_mapping_version": None,
        "window_ms": WIN_MS,
        "stride_ms": STRIDE_MS,
        "feature_order": ("mav", "mavs", "wl", "zc", "ssc"),
        "target_offset_samples": DEFAULT_TARGET_OFFSET_SAMPLES,
        "zc_threshold": None,
        "ssc_threshold": None,
        "feature_normalization": "mu_law",
        "target_normalization": "mu_law",
        "mu_law_mu": DEFAULT_MU_LAW_MU,
        "blocked_train_fraction": 0.7,
        "blocked_val_fraction": 0.15,
        "blocked_test_fraction": 0.15,
        "blocked_gap_windows": 16,
        "seq_len": 8,
        "seq_stride": 1,
        "hidden_units": 64,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "max_epochs": 20,
        "early_stopping_patience": 5,
        "gradient_clip_norm": 1.0,
        "random_seed": 42,
        "num_workers": 0,
        "device": "auto",
        "plot_target_index": 0,
        "plot_max_points": 500,
    }
    config_values.update(overrides)
    return CfCTrainingConfig(**config_values)


@dataclass
class RecordingFeatures:
    """One recording converted into aligned window features."""

    recording_id: str
    x_windows: np.ndarray
    y_windows: np.ndarray
    target_alignment_indices: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    fs: float
    action_labels: np.ndarray | None = None
    repetition_labels: np.ndarray | None = None


@dataclass
class SequenceSplit:
    """
    A many-to-one sequence dataset.

    `x` contains multiple consecutive feature windows.
    `y` is the target attached to the final window of each sequence.
    """

    x: np.ndarray
    y: np.ndarray
    time_s: np.ndarray
    alignment_indices: np.ndarray
    recording_ids: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    action_labels: np.ndarray | None = None
    repetition_labels: np.ndarray | None = None


class SequenceRegressionDataset(data.Dataset):
    """Thin PyTorch wrapper around pre-built sequence arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class WeightedSmoothL1Loss(nn.Module):
    """SmoothL1 loss with fixed per-target weights."""

    def __init__(self, target_weights: list[float] | tuple[float, ...] | np.ndarray) -> None:
        super().__init__()
        weights = torch.as_tensor(target_weights, dtype=torch.float32)
        if weights.ndim != 1:
            raise ValueError("target_weights must be a 1-D sequence")
        if torch.any(weights <= 0):
            raise ValueError("target_weights must be positive")
        self.register_buffer("target_weights", weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        if pred.shape[-1] != self.target_weights.numel():
            raise ValueError(
                f"expected {self.target_weights.numel()} targets, got {pred.shape[-1]}"
            )
        per_target = nn.functional.smooth_l1_loss(pred, target, reduction="none")
        return (per_target * self.target_weights).mean()


class DomainDiscriminator(nn.Module):
    """Binary domain classifier for GAN-based adversarial domain adaptation."""

    def __init__(self, in_dim=128, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DenseCfCLinearRegressor(nn.Module):
    """Dense CfC encoder with an explicit linear DoA readout head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_family = "dense_cfc_linear"
        self.cfc = CfC(
            input_dim,
            hidden_units,
            batch_first=True,
            return_sequences=True,
            backbone_dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = nn.Linear(hidden_units, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_sequence, _ = self.cfc(x)
        final_state = self.dropout(y_sequence[:, -1, :])
        return self.head(final_state)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (prediction, pre_dropout_state) tuple.

        The pre-dropout state is the CfC output at the final timestep before
        dropout is applied, useful for auxiliary tasks such as domain
        adversarial training.
        """
        y_sequence, _ = self.cfc(x)
        pre_dropout_state = y_sequence[:, -1, :]
        final_state = self.dropout(pre_dropout_state)
        return self.head(final_state), pre_dropout_state


def build_cfc_regressor(
    *,
    input_dim: int,
    output_dim: int,
    hidden_units: int,
    model_family: str = "dense_cfc_linear",
    cfc_dropout: float = 0.0,
) -> nn.Module:
    if model_family == "dense_cfc_linear":
        return DenseCfCLinearRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            dropout=cfc_dropout,
        )
    raise ValueError(f"unsupported model_family: {model_family}")


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def resolve_device(requested_device: str) -> torch.device:
    """Choose the training device from a simple string flag."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def list_db2_files(db2_dir: Path) -> list[Path]:
    """List available DB2 recordings recursively in sorted order."""
    files = sorted(db2_dir.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"no .mat files found in {db2_dir}")
    return files


def file_contains_target(file_path: Path, target_source: str) -> bool:
    """
    Check whether a .mat recording exposes the requested regression target.

    `whosmat` is used here because it only inspects metadata and avoids loading
    the whole recording into memory just to test field presence.
    """

    variable_names = {name for name, _, _ in sio.whosmat(file_path)}
    return target_source in variable_names and "emg" in variable_names


def resolve_data_splits(config: CfCTrainingConfig) -> dict[str, list[Path]]:
    """
    Resolve train/val/test file lists.

    The default split is recording-level, not window-level, to avoid leakage
    from heavily overlapping windows. With the three DB2 sample files in this
    repo, that becomes 1 train / 1 val / 1 test by default.
    """

    available_files = {path.name: path for path in list_db2_files(config.db2_dir)}

    if config.train_files or config.val_files or config.test_files:
        def resolve_named_files(names: tuple[str, ...], split_name: str) -> list[Path]:
            resolved = []
            for name in names:
                if name not in available_files:
                    raise FileNotFoundError(f"{split_name} file '{name}' not found under {config.db2_dir}")
                path = available_files[name]
                if not file_contains_target(path, config.target_source):
                    raise KeyError(
                        f"{split_name} file '{name}' does not contain target source '{config.target_source}'"
                    )
                resolved.append(path)
            return resolved

        train_files = resolve_named_files(config.train_files, "train")
        val_files = resolve_named_files(config.val_files, "val")
        test_files = resolve_named_files(config.test_files, "test")
    else:
        files = [
            path
            for path in available_files.values()
            if file_contains_target(path, config.target_source)
        ]
        skipped_files = sorted(set(available_files.values()) - set(files))
        if skipped_files:
            print(
                "Skipping recordings without the requested target source "
                f"'{config.target_source}': {[path.name for path in skipped_files]}"
            )
        if len(files) == 1:
            raise ValueError("need at least two recordings for train/validation splitting")
        if len(files) == 2:
            train_files = [files[0]]
            val_files = [files[1]]
            test_files = [files[1]]
        else:
            train_files = files[:-2]
            val_files = [files[-2]]
            test_files = [files[-1]]

    if not train_files:
        raise ValueError("training split is empty")
    if not val_files:
        raise ValueError("validation split is empty")
    if not test_files:
        test_files = val_files

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def resolve_blocked_source_files(config: CfCTrainingConfig) -> list[Path]:
    """
    Resolve the recordings used by blocked temporal splitting.

    In this strategy, train/validation/test are all drawn from disjoint time
    ranges inside the same set of recordings. This is useful when only a small
    number of recordings expose the required target fields.
    """

    available_files = {path.name: path for path in list_db2_files(config.db2_dir)}
    if config.source_files:
        source_files = []
        for name in config.source_files:
            if name not in available_files:
                raise FileNotFoundError(f"source file '{name}' not found under {config.db2_dir}")
            path = available_files[name]
            if not file_contains_target(path, config.target_source):
                raise KeyError(f"source file '{name}' does not contain target source '{config.target_source}'")
            source_files.append(path)
        return source_files

    return [
        path
        for path in available_files.values()
        if file_contains_target(path, config.target_source)
    ]


def load_recording_features(file_path: Path, config: CfCTrainingConfig) -> RecordingFeatures:
    """Load one .mat recording and convert it into aligned feature windows."""
    pipeline = run_feature_pipeline(
        str(file_path),
        target_source=config.target_source,
        target_columns=list(config.target_columns),
        target_mapping=config.target_mapping,
        target_mapping_source=config.target_mapping_source,
        window_ms=config.window_ms,
        stride_ms=config.stride_ms,
        feature_order=config.feature_order,
        target_offset_samples=config.target_offset_samples,
        zc_threshold=config.zc_threshold,
        ssc_threshold=config.ssc_threshold,
    )
    feature_set = pipeline["feature_set"]
    data = pipeline["data"]

    x_windows = np.asarray(feature_set["feature_matrix"], dtype=np.float32)
    y_windows = np.asarray(feature_set["target_values"], dtype=np.float32)
    alignment_indices = np.asarray(feature_set["target_alignment_indices"], dtype=np.int32)
    action_source = "restimulus" if "restimulus" in data else "stimulus"
    repetition_source = "rerepetition" if "rerepetition" in data else "repetition"
    action_labels = np.asarray(data[action_source]).reshape(-1)[alignment_indices].astype(np.int16)
    repetition_labels = np.asarray(data[repetition_source]).reshape(-1)[alignment_indices].astype(np.int16)

    if config.max_windows_per_file is not None:
        max_windows = min(config.max_windows_per_file, x_windows.shape[0])
        x_windows = x_windows[:max_windows]
        y_windows = y_windows[:max_windows]
        alignment_indices = alignment_indices[:max_windows]
        action_labels = action_labels[:max_windows]
        repetition_labels = repetition_labels[:max_windows]

    return RecordingFeatures(
        recording_id=file_path.name,
        x_windows=x_windows,
        y_windows=y_windows,
        target_alignment_indices=alignment_indices,
        feature_names=list(feature_set["channel_feature_names"]),
        target_names=list(feature_set["target_names"] or []),
        fs=float(feature_set["fs"]),
        action_labels=action_labels,
        repetition_labels=repetition_labels,
    )


def build_sequence_split(
    recordings: list[RecordingFeatures],
    *,
    seq_len: int,
    seq_stride: int,
) -> SequenceSplit:
    """
    Convert multiple recordings into one many-to-one sequence split.

    Sequences never cross recording boundaries. This keeps temporal semantics
    correct and makes later visualization easier to interpret.
    """

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if seq_stride <= 0:
        raise ValueError("seq_stride must be positive")

    x_sequences: list[np.ndarray] = []
    y_sequences: list[np.ndarray] = []
    time_values: list[float] = []
    alignment_values: list[int] = []
    recording_ids: list[str] = []
    action_labels: list[int] = []
    repetition_labels: list[int] = []

    feature_names: list[str] | None = None
    target_names: list[str] | None = None
    sampling_rate: float | None = None

    for recording in recordings:
        if recording.x_windows.shape[0] < seq_len:
            continue

        if feature_names is None:
            feature_names = recording.feature_names
        elif feature_names != recording.feature_names:
            raise ValueError("feature names differ between recordings")

        if target_names is None:
            target_names = recording.target_names
        elif target_names != recording.target_names:
            raise ValueError("target names differ between recordings")

        if sampling_rate is None:
            sampling_rate = recording.fs
        elif not np.isclose(sampling_rate, recording.fs):
            raise ValueError("sampling rates differ between recordings")

        for start in range(0, recording.x_windows.shape[0] - seq_len + 1, seq_stride):
            end = start + seq_len
            last_index = end - 1
            x_sequences.append(recording.x_windows[start:end])
            y_sequences.append(recording.y_windows[last_index])
            alignment_values.append(int(recording.target_alignment_indices[last_index]))
            time_values.append(float(recording.target_alignment_indices[last_index] / recording.fs))
            recording_ids.append(recording.recording_id)
            if recording.action_labels is not None:
                action_labels.append(int(recording.action_labels[last_index]))
            if recording.repetition_labels is not None:
                repetition_labels.append(int(recording.repetition_labels[last_index]))

    if not x_sequences:
        raise ValueError("no sequences were created; lower seq_len or check the input recordings")

    return SequenceSplit(
        x=np.stack(x_sequences).astype(np.float32),
        y=np.stack(y_sequences).astype(np.float32),
        time_s=np.asarray(time_values, dtype=np.float32),
        alignment_indices=np.asarray(alignment_values, dtype=np.int32),
        recording_ids=np.asarray(recording_ids),
        feature_names=feature_names or [],
        target_names=target_names or [],
        action_labels=np.asarray(action_labels, dtype=np.int16) if action_labels else None,
        repetition_labels=np.asarray(repetition_labels, dtype=np.int16) if repetition_labels else None,
    )


def _append_sequences_from_window_range(
    recording: RecordingFeatures,
    *,
    start_window: int,
    end_window: int,
    seq_len: int,
    seq_stride: int,
    x_sequences: list[np.ndarray],
    y_sequences: list[np.ndarray],
    time_values: list[float],
    alignment_values: list[int],
    recording_ids: list[str],
    action_labels: list[int],
    repetition_labels: list[int],
) -> None:
    """
    Create many-to-one sequences from a contiguous window interval.

    `start_window` is inclusive and `end_window` is exclusive.
    """

    available_windows = end_window - start_window
    if available_windows < seq_len:
        return

    for start in range(start_window, end_window - seq_len + 1, seq_stride):
        end = start + seq_len
        last_index = end - 1
        x_sequences.append(recording.x_windows[start:end])
        y_sequences.append(recording.y_windows[last_index])
        alignment_values.append(int(recording.target_alignment_indices[last_index]))
        time_values.append(float(recording.target_alignment_indices[last_index] / recording.fs))
        recording_ids.append(recording.recording_id)
        if recording.action_labels is not None:
            action_labels.append(int(recording.action_labels[last_index]))
        if recording.repetition_labels is not None:
            repetition_labels.append(int(recording.repetition_labels[last_index]))


def _append_sequences_to_split(
    recording: RecordingFeatures,
    *,
    split_name: str,
    segment_start: int,
    segment_end: int,
    seq_len: int,
    seq_stride: int,
    split_buffers: dict[str, dict[str, list]],
) -> None:
    _append_sequences_from_window_range(
        recording,
        start_window=segment_start,
        end_window=segment_end,
        seq_len=seq_len,
        seq_stride=seq_stride,
        x_sequences=split_buffers[split_name]["x"],
        y_sequences=split_buffers[split_name]["y"],
        time_values=split_buffers[split_name]["time"],
        alignment_values=split_buffers[split_name]["alignment"],
        recording_ids=split_buffers[split_name]["recording"],
        action_labels=split_buffers[split_name]["action"],
        repetition_labels=split_buffers[split_name]["repetition"],
    )


def _validate_blocked_split_args(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    gap_windows: int,
) -> None:
    if train_fraction <= 0.0 or val_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("blocked split fractions must all be positive")
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0, atol=1e-6):
        raise ValueError("blocked split fractions must sum to 1.0")
    if gap_windows < 0:
        raise ValueError("gap_windows must be non-negative")


def _empty_sequence_buffers() -> dict[str, dict[str, list]]:
    return {
        "train": {"x": [], "y": [], "time": [], "alignment": [], "recording": [], "action": [], "repetition": []},
        "val": {"x": [], "y": [], "time": [], "alignment": [], "recording": [], "action": [], "repetition": []},
        "test": {"x": [], "y": [], "time": [], "alignment": [], "recording": [], "action": [], "repetition": []},
    }


def _append_blocked_ranges(
    recording: RecordingFeatures,
    *,
    segment_start: int,
    segment_end: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    gap_windows: int,
    seq_len: int,
    seq_stride: int,
    split_buffers: dict[str, dict[str, list]],
) -> None:
    n_windows = segment_end - segment_start
    if n_windows < (3 * seq_len) + (2 * gap_windows):
        return

    train_end = segment_start + int(np.floor(n_windows * train_fraction))
    val_end = segment_start + int(np.floor(n_windows * (train_fraction + val_fraction)))

    train_range = (segment_start, max(train_end - gap_windows, segment_start))
    val_range = (
        min(train_end + gap_windows, segment_end),
        max(min(val_end - gap_windows, segment_end), min(train_end + gap_windows, segment_end)),
    )
    test_range = (min(val_end + gap_windows, segment_end), segment_end)

    for split_name, (start_window, end_window) in {
        "train": train_range,
        "val": val_range,
        "test": test_range,
    }.items():
        _append_sequences_from_window_range(
            recording,
            start_window=start_window,
            end_window=end_window,
            seq_len=seq_len,
            seq_stride=seq_stride,
            x_sequences=split_buffers[split_name]["x"],
            y_sequences=split_buffers[split_name]["y"],
            time_values=split_buffers[split_name]["time"],
            alignment_values=split_buffers[split_name]["alignment"],
            recording_ids=split_buffers[split_name]["recording"],
            action_labels=split_buffers[split_name]["action"],
            repetition_labels=split_buffers[split_name]["repetition"],
        )


def _finalize_sequence_splits(
    split_buffers: dict[str, dict[str, list]],
    *,
    feature_names: list[str],
    target_names: list[str],
    split_kind: str,
) -> dict[str, SequenceSplit]:
    sequence_splits: dict[str, SequenceSplit] = {}
    for split_name, buffers in split_buffers.items():
        if not buffers["x"]:
            raise ValueError(
                f"{split_kind} split '{split_name}' is empty; reduce gap_windows or seq_len, "
                "or provide longer recordings"
            )
        sequence_splits[split_name] = SequenceSplit(
            x=np.stack(buffers["x"]).astype(np.float32),
            y=np.stack(buffers["y"]).astype(np.float32),
            time_s=np.asarray(buffers["time"], dtype=np.float32),
            alignment_indices=np.asarray(buffers["alignment"], dtype=np.int32),
            recording_ids=np.asarray(buffers["recording"]),
            feature_names=feature_names,
            target_names=target_names,
            action_labels=np.asarray(buffers["action"], dtype=np.int16) if buffers["action"] else None,
            repetition_labels=np.asarray(buffers["repetition"], dtype=np.int16) if buffers["repetition"] else None,
        )

    return sequence_splits


def build_blocked_sequence_splits(
    recordings: list[RecordingFeatures],
    *,
    seq_len: int,
    seq_stride: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    gap_windows: int,
) -> dict[str, SequenceSplit]:
    """
    Split each recording into train/val/test time blocks with guard gaps.

    This strategy is a compromise for the current repo state:
    - it lets training use both E1 and E2
    - it still keeps validation and test on unseen contiguous time regions
    - it avoids the worst leakage from adjacent overlapping windows
    """

    _validate_blocked_split_args(train_fraction, val_fraction, test_fraction, gap_windows)

    feature_names: list[str] | None = None
    target_names: list[str] | None = None

    split_buffers = _empty_sequence_buffers()

    for recording in recordings:
        if feature_names is None:
            feature_names = recording.feature_names
        elif feature_names != recording.feature_names:
            raise ValueError("feature names differ between recordings")

        if target_names is None:
            target_names = recording.target_names
        elif target_names != recording.target_names:
            raise ValueError("target names differ between recordings")

        _append_blocked_ranges(
            recording,
            segment_start=0,
            segment_end=recording.x_windows.shape[0],
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            gap_windows=gap_windows,
            seq_len=seq_len,
            seq_stride=seq_stride,
            split_buffers=split_buffers,
        )

    return _finalize_sequence_splits(
        split_buffers,
        feature_names=feature_names or [],
        target_names=target_names or [],
        split_kind="blocked",
    )


def _contiguous_label_ranges(action_labels: np.ndarray, repetition_labels: np.ndarray) -> list[tuple[int, int]]:
    if action_labels.shape != repetition_labels.shape:
        raise ValueError("action and repetition labels must have the same shape")
    if action_labels.ndim != 1:
        raise ValueError("action and repetition labels must be 1-D")
    if action_labels.size == 0:
        return []

    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, action_labels.size):
        if action_labels[index] != action_labels[index - 1] or repetition_labels[index] != repetition_labels[index - 1]:
            ranges.append((start, index))
            start = index
    ranges.append((start, action_labels.size))
    return ranges


def build_action_stratified_sequence_splits(
    recordings: list[RecordingFeatures],
    *,
    seq_len: int,
    seq_stride: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    gap_windows: int,
) -> dict[str, SequenceSplit]:
    """
    Split each action across held-out repetition segments.

    This is the MVP evaluation split for semantic DoA decoding: each action
    contributes to all three final target-subject partitions. When at least
    three repetitions are available, whole repetitions are assigned to splits.
    If a dataset only provides one or two usable repetitions for an action, the
    code falls back to blocked ranges within each repetition with guard gaps.
    Sequences never cross a repetition boundary.
    """

    _validate_blocked_split_args(train_fraction, val_fraction, test_fraction, gap_windows)

    feature_names: list[str] | None = None
    target_names: list[str] | None = None
    split_buffers = _empty_sequence_buffers()

    for recording in recordings:
        if recording.action_labels is None or recording.repetition_labels is None:
            raise ValueError("action_stratified split requires action and repetition labels")

        if feature_names is None:
            feature_names = recording.feature_names
        elif feature_names != recording.feature_names:
            raise ValueError("feature names differ between recordings")

        if target_names is None:
            target_names = recording.target_names
        elif target_names != recording.target_names:
            raise ValueError("target names differ between recordings")

        ranges_by_action: dict[int, list[tuple[int, int]]] = {}
        for segment_start, segment_end in _contiguous_label_ranges(recording.action_labels, recording.repetition_labels):
            if segment_end - segment_start < seq_len:
                continue
            action = int(recording.action_labels[segment_start])
            ranges_by_action.setdefault(action, []).append((segment_start, segment_end))

        for action, ranges in ranges_by_action.items():
            if len(ranges) < 3:
                for segment_start, segment_end in ranges:
                    _append_blocked_ranges(
                        recording,
                        segment_start=segment_start,
                        segment_end=segment_end,
                        train_fraction=train_fraction,
                        val_fraction=val_fraction,
                        test_fraction=test_fraction,
                        gap_windows=gap_windows,
                        seq_len=seq_len,
                        seq_stride=seq_stride,
                        split_buffers=split_buffers,
                    )
                continue

            n_segments = len(ranges)
            train_count = max(1, int(np.floor(n_segments * train_fraction)))
            val_count = max(1, int(np.floor(n_segments * val_fraction)))
            if train_count + val_count >= n_segments:
                train_count = max(1, n_segments - 2)
                val_count = 1

            split_names = (
                ["train"] * train_count
                + ["val"] * val_count
                + ["test"] * (n_segments - train_count - val_count)
            )

            for split_name, (segment_start, segment_end) in zip(split_names, ranges):
                _append_sequences_to_split(
                    recording,
                    split_name=split_name,
                    segment_start=segment_start,
                    segment_end=segment_end,
                    seq_len=seq_len,
                    seq_stride=seq_stride,
                    split_buffers=split_buffers,
                )

    return _finalize_sequence_splits(
        split_buffers,
        feature_names=feature_names or [],
        target_names=target_names or [],
        split_kind="action-stratified",
    )


def fit_target_normalizer(
    target_matrix: np.ndarray,
    *,
    method: str = "zscore",
    mu: float = DEFAULT_MU_LAW_MU,
) -> dict:
    """Fit target normalization statistics using the training split only."""
    return _fit_array_normalizer(
        target_matrix,
        method=method,
        mu=mu,
        value_name="target",
    )


def apply_target_normalizer(target_matrix: np.ndarray, stats: dict) -> np.ndarray:
    """Apply precomputed target normalization."""
    return _apply_array_normalizer(target_matrix, stats, value_name="target")


def inverse_target_normalizer(target_matrix: np.ndarray, stats: dict) -> np.ndarray:
    """Map normalized targets back into the original angle scale."""
    return _inverse_array_normalizer(target_matrix, stats, value_name="target")


def normalize_sequence_inputs(
    split: SequenceSplit,
    *,
    x_stats: dict,
    y_stats: dict,
) -> SequenceSplit:
    """
    Normalize a sequence split with training-set statistics.

    Feature normalization is applied per feature dimension across all timesteps.
    Target normalization is applied per target dimension.
    """

    n_sequences, seq_len, n_features = split.x.shape
    x_flat = split.x.reshape(n_sequences * seq_len, n_features)
    x_norm = apply_feature_normalizer(x_flat, x_stats).reshape(n_sequences, seq_len, n_features)
    y_norm = apply_target_normalizer(split.y, y_stats)

    return SequenceSplit(
        x=x_norm,
        y=y_norm,
        time_s=split.time_s,
        alignment_indices=split.alignment_indices,
        recording_ids=split.recording_ids,
        feature_names=split.feature_names,
        target_names=split.target_names,
        action_labels=split.action_labels,
        repetition_labels=split.repetition_labels,
    )


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute per-target and averaged MAE, normalized MAE, RMSE, and R2."""
    truth = np.asarray(y_true, dtype=np.float32)
    pred = np.asarray(y_pred, dtype=np.float32)
    error = pred - truth

    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    target_min = np.min(truth, axis=0)
    target_max = np.max(truth, axis=0)
    target_range = target_max - target_min
    safe_range = np.maximum(target_range, 1e-6)
    normalized_mae = mae / safe_range

    ss_res = np.sum(np.square(error), axis=0)
    target_mean = np.mean(truth, axis=0, keepdims=True)
    ss_tot = np.sum(np.square(truth - target_mean), axis=0)
    r2 = np.ones_like(ss_res, dtype=np.float32)
    non_constant_targets = ss_tot > 1e-12
    r2[non_constant_targets] = 1.0 - (ss_res[non_constant_targets] / ss_tot[non_constant_targets])
    r2[~non_constant_targets] = 0.0

    return {
        "mae_by_target": mae.astype(np.float32),
        "normalized_mae_by_target": normalized_mae.astype(np.float32),
        "target_range_by_target": target_range.astype(np.float32),
        "rmse_by_target": rmse.astype(np.float32),
        "r2_by_target": r2.astype(np.float32),
        "mae_mean": float(np.mean(mae)),
        "normalized_mae_mean": float(np.mean(normalized_mae)),
        "rmse_mean": float(np.mean(rmse)),
        "r2_mean": float(np.mean(r2)),
    }


def compute_grouped_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray | None,
) -> dict[str, dict]:
    """Compute regression metrics separately for each integer group label."""
    if group_labels is None:
        return {}

    labels = np.asarray(group_labels)
    if labels.ndim != 1:
        raise ValueError("group_labels must be 1-D")
    if labels.shape[0] != np.asarray(y_true).shape[0]:
        raise ValueError("group_labels must align with y_true rows")

    grouped_metrics: dict[str, dict] = {}
    for label in np.unique(labels):
        mask = labels == label
        grouped_metrics[str(int(label))] = compute_regression_metrics(
            np.asarray(y_true)[mask],
            np.asarray(y_pred)[mask],
        )
    return grouped_metrics


def predict_sequences(
    model: nn.Module,
    x_sequences: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run batched inference over a whole split."""
    model.eval()
    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, x_sequences.shape[0], batch_size):
            batch = torch.from_numpy(x_sequences[start : start + batch_size]).to(device)
            pred = model(batch).cpu().numpy().astype(np.float32)
            outputs.append(pred)

    return np.vstack(outputs).astype(np.float32)


def evaluate_split(
    model: nn.Module,
    split: SequenceSplit,
    *,
    target_stats: dict,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Predict one split, invert target normalization, and compute metrics."""
    pred_norm = predict_sequences(model, split.x, batch_size=batch_size, device=device)
    y_pred = inverse_target_normalizer(pred_norm, target_stats)
    y_true = inverse_target_normalizer(split.y, target_stats)
    metrics = compute_regression_metrics(y_true, y_pred)
    per_action_metrics = compute_grouped_regression_metrics(
        y_true,
        y_pred,
        split.action_labels,
    )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
        "per_action_metrics": per_action_metrics,
        "time_s": split.time_s,
        "alignment_indices": split.alignment_indices,
        "recording_ids": split.recording_ids,
        "action_labels": split.action_labels,
        "repetition_labels": split.repetition_labels,
        "target_names": split.target_names,
    }


def train_one_epoch(
    model: nn.Module,
    loader: data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    *,
    device: torch.device,
    gradient_clip_norm: float | None,
) -> float:
    """Run one full training epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()

        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()

        batch_size = x_batch.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def make_train_loader(split: SequenceSplit, config: CfCTrainingConfig) -> data.DataLoader:
    """Build the training DataLoader with the repo's Windows-safe worker setting."""
    dataset = SequenceRegressionDataset(split.x, split.y)
    return data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )


def plot_angle_predictions(
    evaluation: dict,
    *,
    split_name: str,
    target_index: int,
    max_points: int,
) -> plt.Figure:
    """
    Plot actual angle and predicted angle on the same time axis.

    If a split contains multiple recordings, the first recording is shown. This
    keeps the plot temporally coherent instead of concatenating unrelated traces.
    """

    target_names = evaluation["target_names"]
    if not target_names:
        raise ValueError("evaluation does not contain target names")
    if not 0 <= target_index < len(target_names):
        raise ValueError(f"target_index must be in [0, {len(target_names) - 1}]")

    recording_ids = np.asarray(evaluation["recording_ids"])
    first_recording = recording_ids[0]
    same_recording = recording_ids == first_recording

    time_s = np.asarray(evaluation["time_s"], dtype=np.float32)[same_recording][:max_points]
    y_true = np.asarray(evaluation["y_true"], dtype=np.float32)[same_recording, target_index][:max_points]
    y_pred = np.asarray(evaluation["y_pred"], dtype=np.float32)[same_recording, target_index][:max_points]
    residual = y_pred - y_true

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    target_name = target_names[target_index]

    axes[0].plot(time_s, y_true, label="Actual angle", linewidth=1.5, color="#1f77b4")
    axes[0].plot(time_s, y_pred, label="Predicted angle", linewidth=1.3, color="#ff7f0e")
    axes[0].set_title(
        f"CfC Continuous Regression - {split_name} split\n"
        f"{first_recording} | target={target_name}",
        fontsize=12,
    )
    axes[0].set_ylabel(target_name, fontsize=10)
    axes[0].grid(True, linewidth=0.3, alpha=0.5)
    axes[0].legend(fontsize=9)

    axes[1].plot(time_s, residual, linewidth=1.0, color="#2f2f2f")
    axes[1].axhline(0.0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Time (s)", fontsize=10)
    axes[1].set_ylabel("Prediction Error", fontsize=10)
    axes[1].grid(True, linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    return figure


def summarize_split(split_name: str, split: SequenceSplit) -> None:
    """Print a compact overview of one prepared sequence split."""
    print(
        f"{split_name:>5s} | sequences={split.x.shape[0]:6d} | "
        f"seq_len={split.x.shape[1]:2d} | "
        f"feature_dim={split.x.shape[2]:3d} | "
        f"target_dim={split.y.shape[1]:2d}"
    )


def train_cfc_regressor(config: CfCTrainingConfig) -> dict:
    """
    Full experiment entry point.

    Steps:
    1. Load each recording as aligned traditional features.
    2. Build feature sequences for many-to-one regression.
    3. Fit normalization only on the training split.
    4. Train CfC with early stopping on validation MAE.
    5. Evaluate train/val/test and return a prediction plot.
    """

    set_random_seed(config.random_seed)
    device = resolve_device(config.device)
    print(f"Using device: {device}")

    if config.split_strategy == "recording":
        split_files = resolve_data_splits(config)
        print("Resolved recording split:")
        for split_name, file_paths in split_files.items():
            print(f"  {split_name:>5s}: {[path.name for path in file_paths]}")

        recording_splits = {
            split_name: [load_recording_features(path, config) for path in file_paths]
            for split_name, file_paths in split_files.items()
        }
        sequence_splits = {
            split_name: build_sequence_split(recordings, seq_len=config.seq_len, seq_stride=config.seq_stride)
            for split_name, recordings in recording_splits.items()
        }
    elif config.split_strategy == "blocked_time":
        source_files = resolve_blocked_source_files(config)
        print("Resolved blocked-time source recordings:")
        print(f"  source: {[path.name for path in source_files]}")

        all_recordings = [load_recording_features(path, config) for path in source_files]
        sequence_splits = build_blocked_sequence_splits(
            all_recordings,
            seq_len=config.seq_len,
            seq_stride=config.seq_stride,
            train_fraction=config.blocked_train_fraction,
            val_fraction=config.blocked_val_fraction,
            test_fraction=config.blocked_test_fraction,
            gap_windows=config.blocked_gap_windows,
        )
    else:
        raise ValueError(f"unsupported split_strategy: {config.split_strategy}")

    for split_name, split in sequence_splits.items():
        summarize_split(split_name, split)

    x_stats = fit_feature_normalizer(
        sequence_splits["train"].x.reshape(-1, sequence_splits["train"].x.shape[-1]),
        method=config.feature_normalization,
        mu=config.mu_law_mu,
    )
    y_stats = fit_target_normalizer(
        sequence_splits["train"].y,
        method=config.target_normalization,
        mu=config.mu_law_mu,
    )

    normalized_splits = {
        split_name: normalize_sequence_inputs(split, x_stats=x_stats, y_stats=y_stats)
        for split_name, split in sequence_splits.items()
    }

    train_loader = make_train_loader(normalized_splits["train"], config)

    model = build_cfc_regressor(
        input_dim=normalized_splits["train"].x.shape[-1],
        output_dim=normalized_splits["train"].y.shape[-1],
        hidden_units=config.hidden_units,
        model_family=config.model_family,
        cfc_dropout=config.cfc_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()

    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_eval = evaluate_split(
            model,
            normalized_splits["val"],
            target_stats=y_stats,
            batch_size=config.batch_size,
            device=device,
        )
        val_mae = float(val_eval["metrics"]["mae_mean"])
        val_rmse = float(val_eval["metrics"]["rmse_mean"])

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_mae={val_mae:.5f} | "
            f"val_rmse={val_rmse:.5f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)

    evaluations = {
        split_name: evaluate_split(
            model,
            normalized_splits[split_name],
            target_stats=y_stats,
            batch_size=config.batch_size,
            device=device,
        )
        for split_name in ("train", "val", "test")
    }

    print("\nFinal metrics")
    for split_name, evaluation in evaluations.items():
        metrics = evaluation["metrics"]
        print(
            f"  {split_name:>5s} | "
            f"MAE={metrics['mae_mean']:.5f} | "
            f"RMSE={metrics['rmse_mean']:.5f} | "
            f"R2={metrics['r2_mean']:.5f}"
        )

    prediction_figure = plot_angle_predictions(
        evaluations["test"],
        split_name="test",
        target_index=config.plot_target_index,
        max_points=config.plot_max_points,
    )

    return {
        "config": config,
        "model": model,
        "history": history,
        "evaluations": evaluations,
        "prediction_figure": prediction_figure,
        "x_stats": x_stats,
        "y_stats": y_stats,
        "sequence_splits": sequence_splits,
    }


def subject_sort_key(subject_id: str) -> tuple[int, str]:
    """Sort subject ids numerically instead of lexicographically."""
    digits = "".join(character for character in subject_id if character.isdigit())
    if not digits:
        raise ValueError(f"subject id '{subject_id}' does not contain a numeric suffix")
    return int(digits), subject_id


def discover_target_files(db2_dir: Path, target_source: str) -> dict[str, list[str]]:
    """Group usable DB2 recordings by subject id."""
    grouped: dict[str, list[str]] = {}
    for file_path in list_db2_files(db2_dir):
        if not file_contains_target(file_path, target_source):
            continue
        subject_id = file_path.stem.split("_")[0]
        grouped.setdefault(subject_id, []).append(file_path.name)

    sorted_subjects = sorted(grouped, key=subject_sort_key)
    return {
        subject_id: sorted(grouped[subject_id])
        for subject_id in sorted_subjects
    }


def make_jsonable(value: Any) -> Any:
    """Convert common numeric and path types into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    return value


def summarize_best_epoch(history: list[dict[str, float]]) -> dict[str, float]:
    """Extract the validation-best checkpoint record from the pretraining history."""
    if not history:
        raise ValueError("history is empty")
    return min(history, key=lambda entry: entry["val_mae"])


def save_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Persist the experiment summary as JSON."""
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(summary), handle, indent=2)
    return summary_path
