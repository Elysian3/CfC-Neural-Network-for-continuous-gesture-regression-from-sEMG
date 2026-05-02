"""
SwRectify.py - regression-oriented EMG windowing and target alignment
====================================================================

This module sits between signal preprocessing and feature extraction.

The core idea is simple:
1. Break the continuous EMG stream into overlapping windows.
2. For each window, keep both the bipolar signal and its rectified version.
3. Optionally align a continuous target value, such as glove or inclin data,
   to each window so later stages can learn "window -> angle".

For this project, that alignment step matters more than any single feature.
If the target attached to each window is wrong in time, even a strong model
will learn the wrong relationship.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

try:
    from datapreprocess import FS
except ImportError:  # pragma: no cover - package-style fallback
    from .datapreprocess import FS


WIN_MS = 200
STRIDE_MS = 50
TargetMode = Literal["last", "center", "mean"]


def _resolve_window_geometry(
    n_samples: int,
    fs: float,
    window_ms: int,
    stride_ms: int,
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


def _normalize_target_array(targets: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Ensure targets use shape (n_samples, n_targets).

    A single continuous angle trajectory is accepted as shape (n_samples,).
    Internally we still store it as a 2-D array so the rest of the pipeline
    can handle one target or many targets with the same code path.
    """
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


def _resolve_anchor_indices(
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    mode: TargetMode,
) -> np.ndarray:
    """
    Choose one representative sample index for each window.

    The anchor is the exact timestamp used for sample-based target alignment.
    For example, `mode="last"` means "predict the target at the last sample
    inside the EMG window".
    """
    if mode == "last":
        return window_ends - 1
    if mode == "center":
        return window_starts + ((window_ends - window_starts - 1) // 2)
    if mode == "mean":
        # Mean uses the window center only as a plotting anchor.
        return window_starts + ((window_ends - window_starts - 1) // 2)
    raise ValueError(f"unsupported target mode: {mode}")


def align_targets_to_windows(
    targets: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    *,
    mode: TargetMode = "last",
    offset_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align one target vector to each EMG window.

    Modes:
    - `last`: use the last target sample inside the window.
    - `center`: use the temporal center sample of the window.
    - `mean`: average the target over the whole window.

    `offset_samples` shifts the sample-based modes forward or backward in time.
    This is useful when you later want to compensate for EMG-to-motion delay.
    """
    target_array = np.asarray(targets, dtype=np.float32)
    if target_array.ndim != 2:
        raise ValueError("targets must be a 2-D array after normalization")

    if mode == "mean" and offset_samples != 0:
        raise ValueError("offset_samples is not supported when mode='mean'")

    anchor_indices = _resolve_anchor_indices(window_starts, window_ends, mode) + int(offset_samples)
    if np.any((anchor_indices < 0) | (anchor_indices >= target_array.shape[0])):
        raise ValueError("target alignment anchors must be inside the target sample range")
    anchor_indices = anchor_indices.astype(np.int32)

    if mode == "mean":
        aligned = np.vstack(
            [target_array[start:end].mean(axis=0, dtype=np.float64) for start, end in zip(window_starts, window_ends)]
        ).astype(np.float32)
    else:
        aligned = target_array[anchor_indices].astype(np.float32, copy=False)

    return aligned, anchor_indices


def sliding_window(
    emg_filtered: np.ndarray,
    targets: np.ndarray | None = None,
    *,
    fs: float = FS,
    window_ms: int = WIN_MS,
    stride_ms: int = STRIDE_MS,
    target_mode: TargetMode = "last",
    target_offset_samples: int = 0,
    target_names: Iterable[str] | None = None,
    target_prefix: str = "target",
) -> dict:
    """
    Slice a continuous EMG recording into overlapping windows.

    Compared with the older classification-oriented version, this function does
    not assign a majority-vote class label. Instead, it exposes the exact window
    boundaries and optionally aligns one continuous target value to each window.

    Parameters
    ----------
    emg_filtered:
        Bandpass-filtered EMG with shape (n_samples, n_channels).
    targets:
        Optional continuous target array with shape (n_samples,) or
        (n_samples, n_targets). Typical examples are `glove` or `inclin`.
    target_mode:
        How each window picks its target value. `last` is the default because
        it matches online decoding best: the model sees the latest EMG in the
        window and predicts the target at the end of that same window.
    target_offset_samples:
        Optional time shift for sample-based alignment. Shifted windows whose
        target anchor would fall outside the recording are discarded.
    """
    emg_array = np.asarray(emg_filtered, dtype=np.float32)
    if emg_array.ndim != 2:
        raise ValueError("emg_filtered must have shape (n_samples, n_channels)")

    n_samples, n_channels = emg_array.shape
    window_size, stride, n_windows = _resolve_window_geometry(n_samples, fs, window_ms, stride_ms)

    # These three index arrays are the timing contract for the whole pipeline.
    # Every downstream stage can trace a feature vector back to the raw samples
    # that created it.
    window_starts = (np.arange(n_windows, dtype=np.int32) * stride).astype(np.int32)
    window_ends = (window_starts + window_size).astype(np.int32)
    window_centers = (window_starts + ((window_size - 1) // 2)).astype(np.int32)

    target_array = None
    resolved_target_names = None
    if targets is not None:
        target_array = _normalize_target_array(targets, n_samples)
        resolved_target_names = _normalize_target_names(target_names, target_array.shape[1], target_prefix)

        if target_mode != "mean":
            shifted_anchors = _resolve_anchor_indices(window_starts, window_ends, target_mode) + int(target_offset_samples)
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
        target_values, target_alignment_indices = align_targets_to_windows(
            target_array,
            window_starts,
            window_ends,
            mode=target_mode,
            offset_samples=target_offset_samples,
        )

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
        "target_mode": target_mode,
        "target_offset_samples": int(target_offset_samples),
        "target_names": resolved_target_names,
    }


def print_window_summary(windows: dict) -> None:
    """Print the key structural facts about one window batch."""
    print("\nWindowing summary")
    print(f"  windows            : {windows['n_windows']}")
    print(f"  channels           : {windows['n_channels']}")
    print(f"  unrectified shape  : {windows['unrectified'].shape}")
    print(f"  rectified shape    : {windows['rectified'].shape}")
    print(f"  window size        : {windows['window_ms']} ms ({windows['window_size']} samples)")
    print(f"  stride             : {windows['stride_ms']} ms ({windows['stride']} samples)")

    if windows.get("target_values") is not None:
        target_values = np.asarray(windows["target_values"], dtype=np.float32)
        print(f"  target mode        : {windows['target_mode']}")
        print(f"  target shape       : {target_values.shape}")
        print(f"  target names       : {windows['target_names']}")
        print(
            f"  first alignments   : {np.asarray(windows['target_alignment_indices'])[:5].tolist()}"
        )
