from __future__ import annotations
from typing import Iterable
import numpy as np

try:
    from datapreprocess import FS
except ImportError:  # pragma: no cover - package-style fallback
    from .datapreprocess import FS


WIN_MS = 200
STRIDE_MS = 50


def _resolve_window_geometry(
    n_samples: int,
    fs: float,
    window_ms: float,
    stride_ms: float,
) -> tuple[int, int, int]:
    """Convert window settings from milliseconds to sample counts."""
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if stride_ms <= 0:
        raise ValueError("stride_ms must be positive")

    window_size = int(round(fs * window_ms / 1000.0))
    stride = int(round(fs * stride_ms / 1000.0))

    if window_size <= 0:
        raise ValueError("window size resolved to zero samples")
    if stride <= 0:
        raise ValueError("stride resolved to zero samples")
    if n_samples < window_size:
        raise ValueError(
            f"EMG recording has {n_samples} samples, smaller than one window of {window_size} samples"
        )

    n_windows = 1 + (n_samples - window_size) // stride
    return window_size, stride, n_windows


def _1d_to_2d_of_target_array(targets: np.ndarray, n_samples: int) -> np.ndarray:
    target_array = np.asarray(targets, dtype=np.float32)

    if target_array.ndim == 1:
        target_array = target_array[:, np.newaxis]
    elif target_array.ndim != 2:
        raise ValueError("targets must have shape (n_samples,) or (n_samples, n_targets)")

    if target_array.shape[0] != n_samples:
        raise ValueError(
            f"targets must have the same number of samples as EMG: {n_samples}, got {target_array.shape[0]}"
        )

    return target_array


def _normalize_target_names(
    target_names: Iterable[str] | None,
    n_targets: int,
    prefix: str,
) -> list[str]:
    """Return one readable name per target dimension."""
    if target_names is None:
        return [f"{prefix}_{idx + 1}" for idx in range(n_targets)]

    resolved = [str(name) for name in target_names]
    if len(resolved) != n_targets:
        raise ValueError(f"expected {n_targets} target names, got {len(resolved)}")
    return resolved


def sliding_window(
    emg_filtered: np.ndarray,
    targets: np.ndarray | None = None,
    *,
    fs: float = FS,
    window_ms: float = WIN_MS,
    stride_ms: float = STRIDE_MS,
    target_offset_samples: int = 0,
    target_names: Iterable[str] | None = None,
    target_prefix: str = "target",
) -> dict:
    """Slice a continuous EMG recording into overlapping windows.

    Target alignment always uses the last sample inside each window, which
    matches online decoding: the model sees the latest EMG and predicts the
    target at the end of that same window.

    Parameters
    ----------
    emg_filtered:
        Bandpass-filtered EMG with shape (n_samples, n_channels).
    targets:
        Optional continuous target array with shape (n_samples,) or
        (n_samples, n_targets). Typical examples are ``glove`` or ``inclin``.
    target_offset_samples:
        Optional time shift. Shifted windows whose anchor would fall
        outside the recording are discarded.
    """
    emg_array = np.asarray(emg_filtered, dtype=np.float32)
    if emg_array.ndim != 2:
        raise ValueError("emg_filtered must have shape (n_samples, n_channels)")

    n_samples, n_channels = emg_array.shape
    window_size, stride, n_windows = _resolve_window_geometry(n_samples, fs, window_ms, stride_ms)

    window_starts = (np.arange(n_windows, dtype=np.int32) * stride).astype(np.int32)
    window_ends = (window_starts + window_size).astype(np.int32)
    window_centers = (window_starts + ((window_size - 1) // 2)).astype(np.int32)
 
    target_array = None
    resolved_target_names = None
    if targets is not None:
        target_array = _1d_to_2d_of_target_array(targets, n_samples)
        resolved_target_names = _normalize_target_names(target_names, target_array.shape[1], target_prefix)

        # Drop windows whose shifted anchor falls outside the recording.
        shifted_anchors = window_ends - 1 + int(target_offset_samples)
        valid_windows = (shifted_anchors >= 0) & (shifted_anchors < target_array.shape[0])
        if not np.any(valid_windows):
            raise ValueError("target_offset_samples leaves no windows with in-range target anchors")
        window_starts = window_starts[valid_windows]
        window_ends = window_ends[valid_windows]
        window_centers = window_centers[valid_windows]
        n_windows = int(window_starts.shape[0])

    unrectified = np.empty((n_windows, window_size, n_channels), dtype=np.float32)
    rectified = np.empty_like(unrectified)

    for window_idx, (start, end) in enumerate(zip(window_starts, window_ends)):
        segment = emg_array[start:end]
        unrectified[window_idx] = segment
        rectified[window_idx] = np.abs(segment)

    target_values = None
    target_alignment_indices = None

    if target_array is not None:
        anchor_indices = (window_ends - 1 + int(target_offset_samples)).astype(np.int32)
        target_values = target_array[anchor_indices].astype(np.float32, copy=False)
        target_alignment_indices = anchor_indices

    return {
        "unrectified": unrectified,
        "rectified": rectified,
        "window_start_indices": window_starts,
        "window_end_indices": window_ends,
        "window_center_indices": window_centers,
        "window_size": int(window_size),
        "stride": int(stride),
        "window_ms": int(window_ms),
        "stride_ms": int(stride_ms),
        "fs": float(fs),
        "n_windows": int(n_windows),
        "n_channels": int(n_channels),
        "target_values": target_values,
        "target_alignment_indices": target_alignment_indices,
        "target_offset_samples": int(target_offset_samples),
        "target_names": resolved_target_names,
    }
