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
    from doa_mapping import DOA5_NAMES, apply_linear_doa_mapping
    from datapreprocess import FS, load_data, preprocess_emg
    from SwRectify import STRIDE_MS, WIN_MS, sliding_window
except ImportError:  # pragma: no cover - package-style fallback
    from .doa_mapping import DOA5_NAMES, apply_linear_doa_mapping
    from .datapreprocess import FS, load_data, preprocess_emg
    from .SwRectify import STRIDE_MS, WIN_MS, sliding_window


FEATURE_ORDER = ("mav", "mavs", "wl", "zc", "ssc")
SUPPORTED_FEATURES = ("mav", "mavs", "wl", "zc", "ssc", "rms")
DEFAULT_MU_LAW_MU = 1_048_576.0  # 2^20
DEFAULT_REST_PERCENTILE = 10.0
DEFAULT_REST_THRESHOLD_SCALE = 1.0


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


def _compute_rest_thresholds(
    filtered_emg: np.ndarray,
    stimulus: np.ndarray | None = None,
    rest_percentile: float = DEFAULT_REST_PERCENTILE,
    fs: float = FS,
    window_ms: float = WIN_MS,
    stride_ms: float = STRIDE_MS,
) -> np.ndarray:
    """Compute per-channel ZC/SSC thresholds from rest-state EMG amplitude.

    Primary path: when ``stimulus`` labels are available, use ``stimulus == 0``
    frames to directly measure the per-channel rest RMS (noise floor).

    Fallback path: when no labels exist or no rest frames are found, segment
    the full filtered signal into pseudo-windows internally and take the
    ``rest_percentile``-th percentile of per-segment RMS values.

    Parameters
    ----------
    filtered_emg:
        Bandpass-filtered EMG with shape ``(n_samples, n_channels)``.
    stimulus:
        Optional stimulus/restimulus label array.  ``0`` marks rest periods.
    rest_percentile:
        Percentile used in the fallback path (default 10th).
    fs, window_ms, stride_ms:
        Signal parameters used only in the fallback segmentation.

    Returns
    -------
    np.ndarray
        Per-channel threshold array with shape ``(n_channels,)``, dtype float32.
    """
    emg = np.asarray(filtered_emg, dtype=np.float32)
    if emg.ndim != 2:
        raise ValueError("filtered_emg must have shape (n_samples, n_channels)")
    n_channels = emg.shape[1]

    # ── Primary path: stimulus-guided rest measurement ──────────────────────
    if stimulus is not None:
        stim = np.asarray(stimulus).ravel()
        if stim.shape[0] != emg.shape[0]:
            raise ValueError(
                f"stimulus length {stim.shape[0]} does not match "
                f"EMG sample count {emg.shape[0]}"
            )
        rest_mask = stim == 0
        if np.any(rest_mask):
            rest_emg = emg[rest_mask]
            return np.sqrt(
                np.mean(np.square(rest_emg), axis=0, dtype=np.float64)
            ).astype(np.float32)

    # ── Fallback: percentile of internally-segmented RMS values ─────────────
    window_size = int(round(fs * window_ms / 1000.0))
    stride = int(round(fs * stride_ms / 1000.0))
    if window_size <= 0 or stride <= 0:
        raise ValueError(
            "fallback window/stride resolved to zero samples; "
            "check fs, window_ms, stride_ms"
        )

    n_samples = emg.shape[0]
    if n_samples < window_size:
        global_rms = np.sqrt(np.mean(np.square(emg), axis=0, dtype=np.float64))
        return np.maximum(0.1 * global_rms, 1e-8).astype(np.float32)

    n_segments = 1 + (n_samples - window_size) // stride
    segment_rms = np.empty((n_segments, n_channels), dtype=np.float64)
    for i in range(n_segments):
        start = i * stride
        seg = emg[start:start + window_size]
        segment_rms[i] = np.sqrt(np.mean(np.square(seg), axis=0, dtype=np.float64))

    rest_idx = max(1, int(n_segments * rest_percentile / 100.0))
    sorted_rms = np.sort(segment_rms, axis=0)
    rest_rms = np.mean(sorted_rms[:rest_idx], axis=0, dtype=np.float64)
    return np.maximum(rest_rms, 1e-8).astype(np.float32)


def _resolve_channel_thresholds(
    unrectified_windows: np.ndarray,
    threshold: float | np.ndarray,
) -> np.ndarray:
    """Resolve one threshold per channel for ZC and SSC counting.

    The caller must provide an explicit threshold (scalar or per-channel
    array).  Use :func:`_compute_rest_thresholds` upstream to obtain a
    data-driven value, or pass through ``run_feature_pipeline`` which
    handles this automatically.
    """
    if threshold is None:
        raise ValueError(
            "threshold must not be None; use _compute_rest_thresholds() to obtain "
            "a data-driven value, or go through run_feature_pipeline() for automatic "
            "calibration"
        )
    n_channels = unrectified_windows.shape[2]
    resolved = np.asarray(threshold, dtype=np.float32)
    if resolved.ndim == 0:
        resolved = np.full(n_channels, float(resolved), dtype=np.float32)
    elif resolved.shape != (n_channels,):
        raise ValueError(f"threshold must be scalar or shape ({n_channels},), got {resolved.shape}")
    return resolved


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


def root_mean_square(unrectified_windows: np.ndarray) -> np.ndarray:
    """Compute RMS amplitude for every window and channel."""
    return np.sqrt(np.mean(np.square(unrectified_windows), axis=1, dtype=np.float64)).astype(np.float32)


def zero_crossings(
    unrectified_windows: np.ndarray,
    threshold: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count zero crossings per window and channel.

    A crossing is counted only when the signal changes sign and the amplitude
    difference is larger than the channel threshold. This suppresses tiny noisy
    sign flips that carry little physiological meaning.

    Use :func:`_compute_rest_thresholds` to obtain a data-driven threshold, or
    go through :func:`run_feature_pipeline` which handles this automatically.
    """
    channel_thresholds = _resolve_channel_thresholds(unrectified_windows, threshold)

    left = unrectified_windows[:, :-1, :]
    right = unrectified_windows[:, 1:, :]
    crossings = (left * right < 0.0) & (np.abs(left - right) >= channel_thresholds[None, None, :])
    return crossings.sum(axis=1, dtype=np.int32), channel_thresholds


def slope_sign_changes(
    unrectified_windows: np.ndarray,
    threshold: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count slope sign changes per window and channel.

    SSC roughly counts local turning points. It is a cheap proxy for waveform
    shape complexity and often complements amplitude-based features.

    Use :func:`_compute_rest_thresholds` to obtain a data-driven threshold, or
    go through :func:`run_feature_pipeline` which handles this automatically.
    """
    channel_thresholds = _resolve_channel_thresholds(unrectified_windows, threshold)

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
    feature_order: Iterable[str] = FEATURE_ORDER,
    zc_threshold: float | np.ndarray | None = None,
    ssc_threshold: float | np.ndarray | None = None,
) -> dict:
    """
    Extract window-level EMG features from one aligned window batch.

    ZC and SSC features are only computed when they appear in ``feature_order``.
    When they are needed, explicit ``zc_threshold`` and ``ssc_threshold`` must
    be provided.  Use :func:`run_feature_pipeline` for automatic calibration,
    or call :func:`_compute_rest_thresholds` directly when using this function
    outside the full pipeline.

    The return value keeps both the per-feature tensors and one flattened matrix
    because different later stages may prefer different shapes.
    """
    unrectified, rectified = _validate_windows(windows)
    n_windows, _, n_channels = unrectified.shape
    selected_feature_order = tuple(str(feature_name) for feature_name in feature_order)
    invalid_features = sorted(set(selected_feature_order) - set(SUPPORTED_FEATURES))
    if invalid_features:
        raise ValueError(f"unsupported EMG features: {invalid_features}")
    if not selected_feature_order:
        raise ValueError("feature_order cannot be empty")

    need_zc = "zc" in selected_feature_order
    need_ssc = "ssc" in selected_feature_order

    if need_zc and zc_threshold is None:
        raise ValueError(
            "zc_threshold is required when 'zc' is in feature_order; "
            "use run_feature_pipeline() for automatic calibration, or call "
            "_compute_rest_thresholds() to obtain a data-driven value"
        )
    if need_ssc and ssc_threshold is None:
        raise ValueError(
            "ssc_threshold is required when 'ssc' is in feature_order; "
            "use run_feature_pipeline() for automatic calibration, or call "
            "_compute_rest_thresholds() to obtain a data-driven value"
        )

    mav = mean_absolute_value(rectified)
    mavs = mean_absolute_value_slope(mav)
    wl = waveform_length(unrectified)
    rms = root_mean_square(unrectified)

    resolved_zc = None
    resolved_ssc = None
    zc = None
    ssc = None

    if need_zc:
        zc, resolved_zc = zero_crossings(unrectified, zc_threshold)
    if need_ssc:
        ssc, resolved_ssc = slope_sign_changes(unrectified, ssc_threshold)

    feature_by_name = {
        "mav": mav,
        "mavs": mavs,
        "wl": wl,
        "rms": rms,
    }
    if need_zc:
        feature_by_name["zc"] = zc.astype(np.float32)
    if need_ssc:
        feature_by_name["ssc"] = ssc.astype(np.float32)

    # Tensor shape is (windows, channels, features). This is the most natural
    # representation for inspection because each channel keeps its feature block.
    feature_tensor = np.stack(
        tuple(feature_by_name[feature_name] for feature_name in selected_feature_order),
        axis=-1,
    )

    # The flattened matrix is convenient for baseline regression models and for
    # later sequence construction. Channel-major ordering keeps each channel's
    # feature block contiguous.
    feature_matrix = feature_tensor.reshape(n_windows, n_channels * len(selected_feature_order)).astype(np.float32)
    channel_feature_names = [
        f"ch{channel + 1}_{feature_name}"
        for channel in range(n_channels)
        for feature_name in selected_feature_order
    ]

    target_values = windows.get("target_values")
    if target_values is not None:
        target_values = np.asarray(target_values, dtype=np.float32)

    result = {
        "mav": mav,
        "mavs": mavs,
        "wl": wl,
        "rms": rms,
        "feature_order": list(selected_feature_order),
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
        "target_offset_samples": int(windows.get("target_offset_samples", 0)),
        "target_names": windows.get("target_names"),
    }
    if need_zc:
        result["zc"] = zc.astype(np.float32)
        result["zc_thresholds"] = resolved_zc
    if need_ssc:
        result["ssc"] = ssc.astype(np.float32)
        result["ssc_thresholds"] = resolved_ssc

    return result


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


def _select_mapped_targets(
    data: dict,
    *,
    target_mapping: str,
    target_mapping_source: str,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Build semantic target channels from a mapped raw target family."""
    if target_mapping_source not in data:
        raise KeyError(f"target mapping source '{target_mapping_source}' not present in loaded data")

    source_targets = np.asarray(data[target_mapping_source], dtype=np.float32)
    if source_targets.ndim != 2:
        raise ValueError(f"{target_mapping_source} must be 2-D for target mapping")
    mapped_targets = apply_linear_doa_mapping(source_targets, mapping=target_mapping)
    if target_mapping == "glove_columns":
        from doa_mapping import GLOVE_COLUMN_INDICES, GLOVE_COLUMN_NAMES
        return mapped_targets, list(GLOVE_COLUMN_INDICES), list(GLOVE_COLUMN_NAMES)
    return mapped_targets, list(range(source_targets.shape[1])), list(DOA5_NAMES)


def run_feature_pipeline(
    file_path: str,
    *,
    target_source: str = "glove",
    target_columns: int | Iterable[int] | None = None,
    target_mapping: str | None = None,
    target_mapping_source: str = "glove",
    fs: float = FS,
    window_ms: int = WIN_MS,
    stride_ms: int = STRIDE_MS,
    target_offset_samples: int = 0,
    feature_order: Iterable[str] = FEATURE_ORDER,
    zc_threshold: float | np.ndarray | None = None,
    ssc_threshold: float | np.ndarray | None = None,
    rest_percentile: float = DEFAULT_REST_PERCENTILE,
    rest_threshold_scale: float = DEFAULT_REST_THRESHOLD_SCALE,
) -> dict:
    """
    Execute the full feature pipeline from one .mat file to aligned features.

    ZC and SSC thresholds are automatically calibrated from rest-state EMG
    amplitude when not explicitly provided.  Rest periods are identified via
    stimulus/restimulus labels (``stimulus == 0``).  When labels are
    unavailable, the pipeline falls back to a percentile-based estimate.

    The output keeps the raw data, filtered EMG, aligned windows, and features
    together so inspection and training code can share the same artifact.
    """
    data = load_data(file_path)
    if target_mapping is None:
        selected_targets, selected_columns, target_names = _select_targets(
            data,
            target_source=target_source,
            target_columns=target_columns,
        )
        resolved_target_source = target_source
    else:
        selected_targets, selected_columns, target_names = _select_mapped_targets(
            data,
            target_mapping=target_mapping,
            target_mapping_source=target_mapping_source,
        )
        resolved_target_source = target_mapping

    filtered_emg = preprocess_emg(data["emg"], fs=fs)

    # Auto-calibrate ZC/SSC thresholds from rest-state amplitude when not
    # explicitly provided by the caller.
    if zc_threshold is None or ssc_threshold is None:
        # Prefer restimulus (re-labeled), fall back to stimulus
        stim = data.get("restimulus")
        if stim is None:
            stim = data.get("stimulus")
        auto_threshold = _compute_rest_thresholds(
            filtered_emg,
            stimulus=stim,
            rest_percentile=rest_percentile,
            fs=fs,
            window_ms=window_ms,
            stride_ms=stride_ms,
        )
        auto_threshold = auto_threshold * rest_threshold_scale
        if zc_threshold is None:
            zc_threshold = auto_threshold
        if ssc_threshold is None:
            ssc_threshold = auto_threshold

    windows = sliding_window(
        filtered_emg,
        selected_targets,
        fs=fs,
        window_ms=window_ms,
        stride_ms=stride_ms,
        target_offset_samples=target_offset_samples,
        target_names=target_names,
        target_prefix=resolved_target_source,
    )
    feature_set = extract_emg_features(
        windows,
        feature_order=feature_order,
        zc_threshold=zc_threshold,
        ssc_threshold=ssc_threshold,
    )

    return {
        "data": data,
        "filtered_emg": filtered_emg,
        "selected_targets": selected_targets,
        "windows": windows,
        "feature_set": feature_set,
        "target_source": resolved_target_source,
        "target_columns": selected_columns,
        "target_mapping": target_mapping,
        "target_mapping_source": target_mapping_source if target_mapping is not None else None,
        "target_names": target_names,
        "file_path": file_path,
    }


def fit_feature_normalizer(
    feature_matrix: np.ndarray,
    *,
    method: str = "zscore",
    mu: float = DEFAULT_MU_LAW_MU,
) -> dict:
    """
    Fit normalization statistics for a feature matrix.

    Training code should fit these statistics on the training split only.
    Keeping this as an explicit function makes that requirement obvious.
    """
    feature_array = np.asarray(feature_matrix, dtype=np.float32)
    if method == "zscore":
        mean = feature_array.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = feature_array.std(axis=0, dtype=np.float64).astype(np.float32)
        return {
            "method": "zscore",
            "mean": mean,
            "std": np.maximum(std, 1e-6).astype(np.float32),
        }
    if method == "mu_law":
        center = feature_array.mean(axis=0, dtype=np.float64).astype(np.float32)
        centered = feature_array - center
        scale = np.max(np.abs(centered), axis=0).astype(np.float32)
        if mu <= 0.0:
            raise ValueError("mu must be positive for mu-law normalization")
        return {
            "method": "mu_law",
            "center": center,
            "scale": np.maximum(scale, 1e-6).astype(np.float32),
            "mu": float(mu),
        }
    raise ValueError(f"unsupported feature normalization method: {method}")


def apply_feature_normalizer(feature_matrix: np.ndarray, stats: dict) -> np.ndarray:
    """Apply precomputed feature normalization statistics."""
    feature_array = np.asarray(feature_matrix, dtype=np.float32)
    method = stats.get("method", "zscore")
    if method == "zscore":
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        return ((feature_array - mean) / std).astype(np.float32)
    if method == "mu_law":
        center = np.asarray(stats["center"], dtype=np.float32)
        scale = np.asarray(stats["scale"], dtype=np.float32)
        mu = float(stats["mu"])
        scaled = (feature_array - center) / scale
        compressed = np.sign(scaled) * (np.log1p(mu * np.abs(scaled)) / np.log1p(mu))
        return compressed.astype(np.float32)
    raise ValueError(f"unsupported feature normalization method: {method}")


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
            "method": "zscore",
            "mean": np.zeros(feature_matrix.shape[1], dtype=np.float32),
            "std": np.ones(feature_matrix.shape[1], dtype=np.float32),
        }
        x = feature_matrix

    if stats.get("method", "zscore") == "zscore":
        normalization_mean = np.asarray(stats["mean"], dtype=np.float32)
        normalization_std = np.asarray(stats["std"], dtype=np.float32)
    else:
        normalization_mean = np.asarray(stats["center"], dtype=np.float32)
        normalization_std = np.asarray(stats["scale"], dtype=np.float32)

    return {
        "x": x,
        "y": target_matrix,
        "feature_names": feature_set["channel_feature_names"],
        "target_names": feature_set.get("target_names"),
        "normalization_method": stats.get("method", "zscore"),
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
        "window_start_indices": np.asarray(feature_set["window_start_indices"], dtype=np.int32),
        "window_end_indices": np.asarray(feature_set["window_end_indices"], dtype=np.int32),
        "window_center_indices": np.asarray(feature_set["window_center_indices"], dtype=np.int32),
    }
