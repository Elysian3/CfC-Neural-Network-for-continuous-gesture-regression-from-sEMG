"""
feature_extraction.py - window-level EMG features for continuous targets
========================================================================

This module turns aligned EMG windows into compact numeric descriptors.

Pipeline:
    raw .mat
      -> datapreprocess.py   (filtering)
      -> SwRectify.py        (windowing + target alignment)
      -> feature_extraction.py (window-level features for regression)

The output of this module is intentionally model-agnostic. It produces clean
feature matrices and aligned continuous targets first. Model-specific shaping,
such as CfC sequence construction, should happen later once the data contract is
already correct.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from datapreprocess import FS, load_data, preprocess_emg
    from SwRectify import STRIDE_MS, WIN_MS, sliding_window
except ImportError:  # pragma: no cover - package-style fallback
    from .datapreprocess import FS, load_data, preprocess_emg
    from .SwRectify import STRIDE_MS, WIN_MS, sliding_window


FEATURE_ORDER = ("mav", "mavs", "wl", "zc", "ssc")


def _validate_windows(windows: dict) -> tuple[np.ndarray, np.ndarray]:
    """Validate the sliding-window output before extracting features."""
    required = {
        "unrectified",
        "rectified",
        "window_start_indices",
        "window_end_indices",
        "window_center_indices",
        "window_size",
        "stride",
        "fs",
    }
    missing = required.difference(windows)
    if missing:
        raise KeyError(f"windows is missing required keys: {sorted(missing)}")

    unrectified = np.asarray(windows["unrectified"], dtype=np.float32)
    rectified = np.asarray(windows["rectified"], dtype=np.float32)

    if unrectified.ndim != 3 or rectified.ndim != 3:
        raise ValueError("window tensors must have shape (n_windows, window_size, n_channels)")
    if unrectified.shape != rectified.shape:
        raise ValueError("unrectified and rectified windows must have the same shape")

    return unrectified, rectified


def _resolve_channel_thresholds(
    unrectified_windows: np.ndarray,
    threshold: float | np.ndarray | None,
    default_scale: float,
) -> np.ndarray:
    """
    Resolve one threshold per channel for ZC and SSC counting.

    When no explicit threshold is provided, we scale each channel by its mean
    absolute magnitude. This is still a heuristic, but it is now explicit and
    easy to replace with a more principled calibration later.
    """
    n_channels = unrectified_windows.shape[2]

    if threshold is None:
        channel_scale = np.mean(np.abs(unrectified_windows), axis=(0, 1), dtype=np.float64)
        resolved = default_scale * np.maximum(channel_scale, 1e-8)
    else:
        resolved = np.asarray(threshold, dtype=np.float32)
        if resolved.ndim == 0:
            resolved = np.full(n_channels, float(resolved), dtype=np.float32)
        elif resolved.shape != (n_channels,):
            raise ValueError(f"threshold must be scalar or shape ({n_channels},)")

    return resolved.astype(np.float32, copy=False)


def mean_absolute_value(rectified_windows: np.ndarray) -> np.ndarray:
    """Compute mean absolute value for every window and channel."""
    return np.mean(rectified_windows, axis=1, dtype=np.float64).astype(np.float32)


def mean_absolute_value_slope(mav: np.ndarray) -> np.ndarray:
    """
    Compute the window-to-window change in MAV.

    This measures how quickly activation level changes across successive windows.
    The first window has no predecessor, so its slope is defined as zero.
    """
    return np.diff(mav, axis=0, prepend=mav[[0], :]).astype(np.float32)


def waveform_length(unrectified_windows: np.ndarray) -> np.ndarray:
    """Compute waveform length for each window and channel."""
    return np.sum(np.abs(np.diff(unrectified_windows, axis=1)), axis=1, dtype=np.float64).astype(np.float32)


def zero_crossings(
    unrectified_windows: np.ndarray,
    threshold: float | np.ndarray | None = None,
    default_scale: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count zero crossings per window and channel.

    A crossing is counted only when the signal changes sign and the amplitude
    difference is larger than the channel threshold. This suppresses tiny noisy
    sign flips that carry little physiological meaning.
    """
    channel_thresholds = _resolve_channel_thresholds(
        unrectified_windows,
        threshold=threshold,
        default_scale=default_scale,
    )

    left = unrectified_windows[:, :-1, :]
    right = unrectified_windows[:, 1:, :]
    crossings = (left * right < 0.0) & (np.abs(left - right) >= channel_thresholds[None, None, :])
    return crossings.sum(axis=1, dtype=np.int32), channel_thresholds


def slope_sign_changes(
    unrectified_windows: np.ndarray,
    threshold: float | np.ndarray | None = None,
    default_scale: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count slope sign changes per window and channel.

    SSC roughly counts local turning points. It is a cheap proxy for waveform
    shape complexity and often complements amplitude-based features.
    """
    channel_thresholds = _resolve_channel_thresholds(
        unrectified_windows,
        threshold=threshold,
        default_scale=default_scale,
    )

    prev_samples = unrectified_windows[:, :-2, :]
    curr_samples = unrectified_windows[:, 1:-1, :]
    next_samples = unrectified_windows[:, 2:, :]

    left_slope = curr_samples - prev_samples
    right_slope = curr_samples - next_samples

    turning_points = (left_slope * right_slope > 0.0) & (
        (np.abs(left_slope) >= channel_thresholds[None, None, :])
        | (np.abs(right_slope) >= channel_thresholds[None, None, :])
    )
    return turning_points.sum(axis=1, dtype=np.int32), channel_thresholds


def extract_emg_features(
    windows: dict,
    *,
    zc_threshold: float | np.ndarray | None = None,
    ssc_threshold: float | np.ndarray | None = None,
    threshold_scale: float = 0.01,
) -> dict:
    """
    Extract window-level EMG features from one aligned window batch.

    The return value keeps both the per-feature tensors and one flattened matrix
    because different later stages may prefer different shapes.
    """
    unrectified, rectified = _validate_windows(windows)
    n_windows, _, n_channels = unrectified.shape

    mav = mean_absolute_value(rectified)
    mavs = mean_absolute_value_slope(mav)
    wl = waveform_length(unrectified)
    zc, resolved_zc = zero_crossings(unrectified, threshold=zc_threshold, default_scale=threshold_scale)
    ssc, resolved_ssc = slope_sign_changes(unrectified, threshold=ssc_threshold, default_scale=threshold_scale)

    # Tensor shape is (windows, channels, features). This is the most natural
    # representation for inspection because each channel keeps its feature block.
    feature_tensor = np.stack((mav, mavs, wl, zc.astype(np.float32), ssc.astype(np.float32)), axis=-1)

    # The flattened matrix is convenient for baseline regression models and for
    # later sequence construction. Channel-major ordering keeps each channel's
    # feature block contiguous.
    feature_matrix = feature_tensor.reshape(n_windows, n_channels * len(FEATURE_ORDER)).astype(np.float32)
    channel_feature_names = [
        f"ch{channel + 1}_{feature_name}"
        for channel in range(n_channels)
        for feature_name in FEATURE_ORDER
    ]

    target_values = windows.get("target_values")
    if target_values is not None:
        target_values = np.asarray(target_values, dtype=np.float32)

    return {
        "mav": mav,
        "mavs": mavs,
        "wl": wl,
        "zc": zc.astype(np.float32),
        "ssc": ssc.astype(np.float32),
        "feature_order": list(FEATURE_ORDER),
        "feature_tensor": feature_tensor,
        "feature_matrix": feature_matrix,
        "channel_feature_names": channel_feature_names,
        "window_start_indices": np.asarray(windows["window_start_indices"], dtype=np.int32),
        "window_end_indices": np.asarray(windows["window_end_indices"], dtype=np.int32),
        "window_center_indices": np.asarray(windows["window_center_indices"], dtype=np.int32),
        "window_size": int(windows["window_size"]),
        "stride": int(windows["stride"]),
        "window_ms": int(windows["window_ms"]),
        "stride_ms": int(windows["stride_ms"]),
        "fs": float(windows["fs"]),
        "n_windows": int(n_windows),
        "n_channels": int(n_channels),
        "target_values": target_values,
        "target_alignment_indices": (
            None
            if windows.get("target_alignment_indices") is None
            else np.asarray(windows["target_alignment_indices"], dtype=np.int32)
        ),
        "target_mode": windows.get("target_mode"),
        "target_offset_samples": int(windows.get("target_offset_samples", 0)),
        "target_names": windows.get("target_names"),
        "zc_thresholds": resolved_zc,
        "ssc_thresholds": resolved_ssc,
    }


def _resolve_target_columns(target_columns: int | Iterable[int] | None, n_targets: int) -> list[int]:
    """Normalize target column selection into a validated list of indices."""
    if target_columns is None:
        columns = list(range(n_targets))
    elif isinstance(target_columns, int):
        columns = [target_columns]
    else:
        columns = [int(column) for column in target_columns]

    if not columns:
        raise ValueError("target_columns cannot be empty")

    for column in columns:
        if not 0 <= column < n_targets:
            raise ValueError(f"target column {column} is outside [0, {n_targets - 1}]")

    return columns


def _select_targets(
    data: dict,
    target_source: str,
    target_columns: int | Iterable[int] | None,
) -> tuple[np.ndarray, list[int], list[str]]:
    """
    Select one continuous target family from the loaded .mat recording.

    We intentionally choose one family at a time. Mixing glove and inclin into
    one output vector is possible later, but it makes early experiments harder
    to reason about.
    """
    if target_source not in data:
        raise KeyError(f"target source '{target_source}' not present in loaded data")

    target_array = np.asarray(data[target_source], dtype=np.float32)
    if target_array.ndim == 1:
        target_array = target_array[:, np.newaxis]
    elif target_array.ndim != 2:
        raise ValueError(f"{target_source} must be 1-D or 2-D, got {target_array.ndim}-D")

    selected_columns = _resolve_target_columns(target_columns, target_array.shape[1])
    selected_targets = target_array[:, selected_columns].astype(np.float32, copy=False)
    target_names = [f"{target_source}_{column + 1}" for column in selected_columns]
    return selected_targets, selected_columns, target_names


def run_feature_pipeline(
    file_path: str,
    *,
    target_source: str = "glove",
    target_columns: int | Iterable[int] | None = None,
    fs: float = FS,
    window_ms: int = WIN_MS,
    stride_ms: int = STRIDE_MS,
    target_mode: str = "last",
    target_offset_samples: int = 0,
    zc_threshold: float | np.ndarray | None = None,
    ssc_threshold: float | np.ndarray | None = None,
    threshold_scale: float = 0.01,
) -> dict:
    """
    Execute the full feature pipeline from one .mat file to aligned features.

    The output keeps the raw data, filtered EMG, aligned windows, and features
    together so inspection and training code can share the same artifact.
    """
    data = load_data(file_path)
    selected_targets, selected_columns, target_names = _select_targets(
        data,
        target_source=target_source,
        target_columns=target_columns,
    )

    filtered_emg = preprocess_emg(data["emg"], fs=fs)
    windows = sliding_window(
        filtered_emg,
        selected_targets,
        fs=fs,
        window_ms=window_ms,
        stride_ms=stride_ms,
        target_mode=target_mode,
        target_offset_samples=target_offset_samples,
        target_names=target_names,
        target_prefix=target_source,
    )
    feature_set = extract_emg_features(
        windows,
        zc_threshold=zc_threshold,
        ssc_threshold=ssc_threshold,
        threshold_scale=threshold_scale,
    )

    return {
        "data": data,
        "filtered_emg": filtered_emg,
        "selected_targets": selected_targets,
        "windows": windows,
        "feature_set": feature_set,
        "target_source": target_source,
        "target_columns": selected_columns,
        "target_names": target_names,
        "file_path": file_path,
    }


def fit_feature_normalizer(feature_matrix: np.ndarray) -> dict:
    """
    Fit simple z-score statistics for a feature matrix.

    Training code should fit these statistics on the training split only.
    Keeping this as an explicit function makes that requirement obvious.
    """
    feature_array = np.asarray(feature_matrix, dtype=np.float32)
    mean = feature_array.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = feature_array.std(axis=0, dtype=np.float64).astype(np.float32)
    return {
        "mean": mean,
        "std": np.maximum(std, 1e-6).astype(np.float32),
    }


def apply_feature_normalizer(feature_matrix: np.ndarray, stats: dict) -> np.ndarray:
    """Apply precomputed z-score statistics to a feature matrix."""
    feature_array = np.asarray(feature_matrix, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((feature_array - mean) / std).astype(np.float32)


def prepare_regression_data(
    feature_set: dict,
    *,
    normalize: bool = False,
    normalization_stats: dict | None = None,
) -> dict:
    """
    Convert one feature set into `(x, y)` regression arrays.

    This function does not build CfC sequences yet. It prepares one feature
    vector per window and one continuous target vector per window. Sequence
    construction should happen in the training stage, after data splitting.
    """
    feature_matrix = np.asarray(feature_set["feature_matrix"], dtype=np.float32)
    target_values = feature_set.get("target_values")
    if target_values is None:
        raise ValueError("feature_set does not contain aligned target values")

    target_matrix = np.asarray(target_values, dtype=np.float32)

    if normalize:
        stats = normalization_stats or fit_feature_normalizer(feature_matrix)
        x = apply_feature_normalizer(feature_matrix, stats)
    else:
        stats = {
            "mean": np.zeros(feature_matrix.shape[1], dtype=np.float32),
            "std": np.ones(feature_matrix.shape[1], dtype=np.float32),
        }
        x = feature_matrix

    return {
        "x": x,
        "y": target_matrix,
        "feature_names": feature_set["channel_feature_names"],
        "target_names": feature_set.get("target_names"),
        "normalization_mean": np.asarray(stats["mean"], dtype=np.float32),
        "normalization_std": np.asarray(stats["std"], dtype=np.float32),
        "window_start_indices": np.asarray(feature_set["window_start_indices"], dtype=np.int32),
        "window_end_indices": np.asarray(feature_set["window_end_indices"], dtype=np.int32),
        "window_center_indices": np.asarray(feature_set["window_center_indices"], dtype=np.int32),
    }

