"""
features.py — EMG Feature Extraction Pipeline
==============================================
This module sits one step after datapreprocess.py in the signal pipeline:

    raw .mat  →  datapreprocess.py (filtering)  →  features.py (rectification + features)

Current stage implemented:
    • Full-wave rectification
    • Sliding window segmentation (with majority-vote label assignment)
    • Temporal dynamic visualisation

Planned future stages (to be added here):
    • Time-domain features: MAV, RMS, WL, ZC, SSC, ...
    • Frequency-domain features: MNF, MDF, ...
    • Feature normalisation / standardisation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Import the preprocessing pipeline from the sibling module so we have a
# single source of truth for filtering parameters (FS, filter settings, etc.)
from datapreprocess import FS, load_data, preprocess_emg

# ── Windowing constants ───────────────────────────────────────────────────────
WIN_MS    = 200   # Default window size in milliseconds  → 400 samples @ 2000 Hz
STRIDE_MS = 50    # Default stride in milliseconds       → 100 samples @ 2000 Hz


# ─────────────────────────────────────────────────────────────────────────────
# Full-wave rectification
# ─────────────────────────────────────────────────────────────────────────────

def full_wave_rectify(emg: np.ndarray) -> np.ndarray:
    """
    Apply full-wave rectification to a filtered EMG signal.

    What is rectification?
    ----------------------
    Raw EMG is an alternating signal — it swings both positive and negative
    around zero.  The negative half of the wave carries exactly the same
    muscle-activation information as the positive half; they are mirror images.

    Full-wave rectification takes the absolute value of every sample:

        rectified[n] = |emg[n]|

    This flips all negative values to positive, so the signal now represents
    the *magnitude* of muscle electrical activity at every time point.
    The result is always ≥ 0.

    Why do we do this?
    ------------------
    1. It is the first step toward computing the signal envelope (the smooth
       curve that "follows" the peaks of muscle activity), which is what
       most feature-extraction methods (MAV, RMS, etc.) are built on.
    2. Averaging a raw bipolar EMG over any window gives ≈ 0 (positive and
       negative cancel out).  Averaging the rectified signal gives a
       meaningful number: the Mean Absolute Value (MAV), a proxy for muscle
       force.

    Half-wave vs full-wave
    ----------------------
    Half-wave rectification keeps only the positive part (negatives → 0).
    Full-wave keeps both halves by mirroring negatives, so no information is
    discarded.  Full-wave is the standard in sEMG research.

    Parameters
    ----------
    emg : ndarray, shape (N, C)
        Filtered (and bandpass-applied) EMG matrix.
        N = number of time samples, C = number of channels.

    Returns
    -------
    ndarray, shape (N, C)
        Rectified EMG — all values are non-negative, same shape as input.
    """
    # np.abs() computes the element-wise absolute value.
    # Because emg is a 2-D array (samples × channels), this operates on every
    # sample of every channel simultaneously — no Python loop needed.
    return np.abs(emg)


# ─────────────────────────────────────────────────────────────────────────────
# Quick visual check
# ─────────────────────────────────────────────────────────────────────────────

def plot_rectification(filtered_emg: np.ndarray,
                       rectified_emg: np.ndarray,
                       fs: float = FS,
                       channels: list = None,
                       num_points: int = 2000) -> None:
    """
    Plot filtered vs. rectified EMG side-by-side for a subset of channels.

    Parameters
    ----------
    filtered_emg  : ndarray (N, C) — output of preprocess_emg()
    rectified_emg : ndarray (N, C) — output of full_wave_rectify()
    fs            : float          — sampling frequency in Hz (for time axis)
    channels      : list of int    — channel indices to plot (default: first 4)
    num_points    : int            — number of samples to display
    """
    if channels is None:
        # Default: show the first 4 channels so the figure isn't overwhelming
        channels = list(range(min(4, filtered_emg.shape[1])))

    # Time axis: convert sample index to seconds
    t = np.arange(num_points) / fs

    n_rows = len(channels)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 2.8 * n_rows), sharex=True)

    # If only one channel is selected, axes is 1-D — wrap it for uniform indexing
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Full-Wave Rectification\n(left: bandpass-filtered | right: rectified |emg|)",
        fontsize=13, y=1.002
    )

    for row, ch in enumerate(channels):
        # ── Left panel: filtered signal (bipolar, swings ±) ──────────────────
        # The signal crosses zero many times per second.  You can see both
        # positive (upward) and negative (downward) bursts during muscle activity.
        axes[row, 0].plot(
            t, filtered_emg[:num_points, ch],
            linewidth=0.5, color='steelblue'
        )
        axes[row, 0].axhline(0, color='black', linewidth=0.4, linestyle='--')  # zero reference
        axes[row, 0].set_ylabel(f"CH {ch + 1}", fontsize=8)
        axes[row, 0].grid(True, linewidth=0.3)
        if row == 0:
            axes[row, 0].set_title("Filtered EMG (bipolar)", fontsize=10)

        # ── Right panel: rectified signal (non-negative) ──────────────────────
        # The signal is now entirely above zero.  The envelope of this curve
        # directly reflects the level of muscle contraction over time.
        axes[row, 1].plot(
            t, rectified_emg[:num_points, ch],
            linewidth=0.5, color='darkorange'
        )
        axes[row, 1].axhline(0, color='black', linewidth=0.4, linestyle='--')  # zero reference
        axes[row, 1].grid(True, linewidth=0.3)
        if row == 0:
            axes[row, 1].set_title("Rectified EMG  |emg|", fontsize=10)

    for col in range(2):
        axes[-1, col].set_xlabel("Time (s)", fontsize=9)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Sliding window segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _majority_vote_labels(labels: np.ndarray,
                          start: int,
                          window_size: int) -> int:
    """
    Assign a single integer label to a window via majority vote.

    Parameters
    ----------
    labels      : ndarray (N,) — per-sample stimulus labels (integers)
    start       : int          — first sample index of this window
    window_size : int          — number of samples in the window

    Returns
    -------
    int
        The label that appears most frequently in the window.
        In case of a tie, np.argmax returns the smallest label index
        (typically 0 = rest); callers may choose to flag/discard ties.
    """
    window_labels = labels[start : start + window_size].astype(int)
    counts = np.bincount(window_labels)          # counts[k] = #samples with label k
    return int(np.argmax(counts))                # label with highest count wins


def sliding_window(emg_filtered: np.ndarray,
                   labels: np.ndarray,
                   fs: float = FS,
                   window_ms: int = WIN_MS,
                   stride_ms: int = STRIDE_MS) -> dict:
    """
    Segment a continuous EMG recording into overlapping windows and assign
    one gesture label per window via majority vote.

    Both the unrectified (bandpass-filtered) and rectified (|filtered|) versions
    of every window are returned so that downstream feature extractors can
    use whichever representation they need:

        • Unrectified → Zero Crossings (ZC), Slope Sign Changes (SSC)
        • Rectified   → Mean Absolute Value (MAV), RMS, Waveform Length (WL)

    Parameters
    ----------
    emg_filtered : ndarray, shape (N, C)
        Output of preprocess_emg() — bandpass-filtered, NOT yet rectified.
        N = total samples, C = channels (12 for NinaPro DB2).
    labels : ndarray, shape (N,) or (N, 1)
        Per-sample stimulus labels from the .mat file
        (use 'restimulus' for cleaner label alignment).
    fs : float
        Sampling frequency in Hz (default: 2000).
    window_ms : int
        Window length in milliseconds (default: 200 → 400 samples).
    stride_ms : int
        Stride between consecutive windows in milliseconds
        (default: 50 → 100 samples, giving 75 % overlap).

    Returns
    -------
    dict with keys:
        'unrectified'  : ndarray (n_windows, window_size, C)
        'rectified'    : ndarray (n_windows, window_size, C)
        'labels'       : ndarray (n_windows,)  — majority-vote label per window
        'window_size'  : int    — window length in samples
        'stride'       : int    — stride in samples
        'window_ms'    : int
        'stride_ms'    : int
        'fs'           : float
        'is_tie'       : ndarray (n_windows,) bool — True where the vote was a tie
    """
    # ── Flatten labels to 1-D ────────────────────────────────────────────────
    labels = np.asarray(labels).ravel()

    # ── Convert ms → samples ─────────────────────────────────────────────────
    window_size = int(fs * window_ms / 1000)   # 200 ms → 400 samples
    stride      = int(fs * stride_ms / 1000)   # 50 ms  → 100 samples

    N, C = emg_filtered.shape
    n_windows = (N - window_size) // stride + 1

    # ── Pre-allocate output arrays ───────────────────────────────────────────
    unrect  = np.empty((n_windows, window_size, C), dtype=np.float64)
    rect    = np.empty_like(unrect)
    win_labels = np.empty(n_windows, dtype=np.int32)
    is_tie     = np.zeros(n_windows, dtype=bool)

    for i in range(n_windows):
        start = i * stride
        end   = start + window_size

        seg = emg_filtered[start:end, :]       # (window_size, C)
        unrect[i] = seg
        rect[i]   = np.abs(seg)

        # ── Majority vote label ───────────────────────────────────────────
        win_lbl_slice = labels[start:end].astype(int)
        counts = np.bincount(win_lbl_slice)
        best   = int(np.argmax(counts))
        win_labels[i] = best

        # Flag ties: if the runner-up matches the winner's count it's a tie
        if counts.size > 1:
            sorted_counts = np.sort(counts)[::-1]
            if sorted_counts[0] == sorted_counts[1]:
                is_tie[i] = True

    return {
        'unrectified' : unrect,
        'rectified'   : rect,
        'labels'      : win_labels,
        'window_size' : window_size,
        'stride'      : stride,
        'window_ms'   : window_ms,
        'stride_ms'   : stride_ms,
        'fs'          : fs,
        'is_tie'      : is_tie,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Temporal dynamic visualisation
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette for up to 18 gesture classes (0 = rest → grey, 1-17 → colours)
_LABEL_COLOURS = [
    '#aaaaaa',  # 0  rest
    '#e6194b',  # 1
    '#3cb44b',  # 2
    '#4363d8',  # 3
    '#f58231',  # 4
    '#911eb4',  # 5
    '#42d4f4',  # 6
    '#f032e6',  # 7
    '#bfef45',  # 8
    '#fabed4',  # 9
    '#469990',  # 10
    '#dcbeff',  # 11
    '#9A6324',  # 12
    '#fffac8',  # 13
    '#800000',  # 14
    '#aaffc3',  # 15
    '#808000',  # 16
    '#ffd8b1',  # 17
]


def plot_temporal_dynamics(emg_filtered: np.ndarray,
                           labels: np.ndarray,
                           windows: dict,
                           channel: int = 0,
                           display_samples: int = 4000,
                           highlight_windows: int = 8) -> None:
    """
    Temporal dynamic visualisation — three stacked panels:

        Panel 1 (top)    — Raw filtered EMG for one channel with colour-coded
                           per-sample gesture labels as a background band.
        Panel 2 (middle) — Rectified EMG for the same channel with sliding
                           window boundaries drawn as shaded blocks.  Each
                           block is coloured by its majority-vote label.
        Panel 3 (bottom) — Discrete majority-vote label per window plotted as
                           a step function so you can see how the vote changes
                           over time.

    Parameters
    ----------
    emg_filtered     : ndarray (N, C)   — filtered (unrectified) EMG
    labels           : ndarray (N,)     — per-sample stimulus labels
    windows          : dict             — output of sliding_window()
    channel          : int              — which EMG channel to display (0-indexed)
    display_samples  : int              — how many raw samples to show (default 4000 = 2 s)
    highlight_windows: int              — how many consecutive windows to shade
                                         (starting from the first non-rest window)
    """
    labels   = np.asarray(labels).ravel()
    stride   = windows['stride']
    win_size = windows['window_size']
    fs       = windows['fs']

    n_show   = min(display_samples, emg_filtered.shape[0])
    t        = np.arange(n_show) / fs                     # time axis (s)

    raw_ch   = emg_filtered[:n_show, channel]             # unrectified
    rect_ch  = np.abs(raw_ch)                             # rectified
    lbl_show = labels[:n_show]

    # Which windows fall entirely within the display range?
    win_starts = np.array([i * stride for i in range(windows['labels'].size)])
    win_ends   = win_starts + win_size
    in_range   = win_ends <= n_show
    vis_idx    = np.where(in_range)[0]                    # indices into windows arrays

    all_labels  = np.unique(windows['labels'])
    n_labels    = int(all_labels.max()) + 1
    colours     = (_LABEL_COLOURS * ((n_labels // len(_LABEL_COLOURS)) + 1))[:n_labels]

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=False)
    fig.suptitle(
        f'Temporal Dynamics — CH {channel + 1}  '
        f'(window={windows["window_ms"]} ms, stride={windows["stride_ms"]} ms)',
        fontsize=13
    )

    # ── Panel 1: Raw filtered EMG + per-sample label background ──────────────
    ax1 = axes[0]
    ax1.set_title('Raw Filtered EMG with Per-Sample Labels', fontsize=10)

    # Draw a colour band at the bottom of the plot for per-sample labels
    y_min, y_max = raw_ch.min(), raw_ch.max()
    band_h = (y_max - y_min) * 0.07         # height of label band
    band_y = y_min - band_h * 1.3

    prev_lbl = lbl_show[0]
    seg_start = 0
    for s in range(1, n_show):
        if lbl_show[s] != prev_lbl or s == n_show - 1:
            t0 = seg_start / fs
            t1 = s / fs
            ax1.axvspan(t0, t1, ymin=0.0, ymax=0.06,
                        color=colours[prev_lbl], alpha=0.85, linewidth=0)
            prev_lbl  = lbl_show[s]
            seg_start = s

    ax1.plot(t, raw_ch, linewidth=0.5, color='steelblue', zorder=3)
    ax1.set_ylabel('Amplitude (V)', fontsize=8)
    ax1.grid(True, linewidth=0.3, alpha=0.5)
    ax1.set_xlim(0, n_show / fs)

    # ── Panel 2: Rectified EMG + window shading (majority-vote colour) ───────
    ax2 = axes[1]
    ax2.set_title('Rectified EMG with Sliding Window Boundaries (colour = majority-vote label)',
                  fontsize=10)

    ax2.plot(t, rect_ch, linewidth=0.5, color='darkorange', zorder=3)

    # Shade every visible window; only shade `highlight_windows` for clarity
    # Find first non-rest window as anchor
    non_rest = vis_idx[windows['labels'][vis_idx] != 0] if any(windows['labels'][vis_idx] != 0) else vis_idx
    anchor   = non_rest[0] if len(non_rest) else (vis_idx[0] if len(vis_idx) else 0)
    hl_range = range(anchor, min(anchor + highlight_windows, len(vis_idx)))

    for k in vis_idx:
        ws = win_starts[k] / fs
        we = win_ends[k]   / fs
        lbl = windows['labels'][k]
        is_highlight = (k in range(anchor, anchor + highlight_windows))
        ax2.axvspan(ws, we, color=colours[lbl],
                    alpha=0.35 if is_highlight else 0.12,
                    linewidth=0)
        if is_highlight:
            ax2.axvline(ws, color=colours[lbl], linewidth=0.8, alpha=0.7)

    ax2.set_ylabel('|Amplitude| (V)', fontsize=8)
    ax2.grid(True, linewidth=0.3, alpha=0.5)
    ax2.set_xlim(0, n_show / fs)

    # ── Panel 3: Majority-vote label per window (step function) ──────────────
    ax3 = axes[2]
    ax3.set_title('Majority-Vote Label per Window (step function)', fontsize=10)

    # Build x coordinates: each window covers [start, end)
    step_x = []
    step_y = []
    for k in vis_idx:
        ws  = win_starts[k] / fs
        we  = win_ends[k]   / fs
        lbl = windows['labels'][k]
        step_x.extend([ws, we])
        step_y.extend([lbl,  lbl])

    ax3.plot(step_x, step_y, color='#333333', linewidth=1.2, drawstyle='steps-post')

    # Colour fill under the step function
    for k in vis_idx:
        ws  = win_starts[k] / fs
        we  = win_ends[k]   / fs
        lbl = windows['labels'][k]
        ax3.axvspan(ws, we, ymin=0, ymax=1,
                    color=colours[lbl], alpha=0.3, linewidth=0)

    # Mark tie windows
    tie_idx = vis_idx[windows['is_tie'][vis_idx]]
    for k in tie_idx:
        ws = win_starts[k] / fs
        ax3.axvline(ws, color='red', linewidth=0.8, linestyle=':', alpha=0.7)

    ax3.set_ylabel('Gesture Label', fontsize=8)
    ax3.set_xlabel('Time (s)', fontsize=9)
    ax3.set_yticks(sorted(all_labels))
    ax3.grid(True, linewidth=0.3, alpha=0.5)
    ax3.set_xlim(0, n_show / fs)

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=colours[lbl],
                       label=f'Label {lbl}' + (' (rest)' if lbl == 0 else ''))
        for lbl in sorted(all_labels)
    ]
    tie_patch = mpatches.Patch(color='red', label='Tie window (⚠)', alpha=0.5)
    fig.legend(handles=legend_patches + [tie_patch],
               loc='lower center', ncol=min(10, len(all_labels) + 1),
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os

    # ── Load and filter ───────────────────────────────────────────────────────
    mat_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'src', 'data', 'DB2', 'S1_E1_A1.mat'
    )

    data = load_data(mat_path)

    # preprocess_emg() runs the full filter chain:
    #   DC removal → 50 Hz notch → 20–450 Hz bandpass
    filtered_emg = preprocess_emg(data['emg'])

    # ── Rectify ───────────────────────────────────────────────────────────────
    rectified_emg = full_wave_rectify(filtered_emg)

    # ── Sanity check ──────────────────────────────────────────────────────────
    print(f"Filtered  EMG — min: {filtered_emg.min():.4f}  max: {filtered_emg.max():.4f}")
    print(f"Rectified EMG — min: {rectified_emg.min():.4f}  max: {rectified_emg.max():.4f}")
    assert rectified_emg.min() >= 0.0, "Rectification failed: negative values found!"
    print("Assertion passed: all rectified values are non-negative.\n")

    # ── Visualise rectification ───────────────────────────────────────────────
    plot_rectification(
        filtered_emg,
        rectified_emg,
        fs=FS,
        channels=[0, 1, 2, 3],
        num_points=2000
    )

    # ── Sliding window segmentation ───────────────────────────────────────────
    # Use 'restimulus' (re-processed labels) for cleaner gesture boundaries.
    # Fall back to 'stimulus' if not present.
    label_key = 'restimulus' if 'restimulus' in data else 'stimulus'
    labels_raw = data[label_key].ravel()

    print(f"\nUsing label field: '{label_key}'")
    print(f"Unique labels in recording: {np.unique(labels_raw)}")

    windows = sliding_window(
        filtered_emg,
        labels_raw,
        fs=FS,
        window_ms=WIN_MS,
        stride_ms=STRIDE_MS,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    n_win   = windows['labels'].size
    n_ties  = windows['is_tie'].sum()
    unique, counts = np.unique(windows['labels'], return_counts=True)

    print(f"\n── Sliding Window Summary ──────────────────────────────────")
    print(f"  Window size : {WIN_MS} ms  ({windows['window_size']} samples)")
    print(f"  Stride      : {STRIDE_MS} ms  ({windows['stride']} samples)")
    print(f"  Total windows extracted : {n_win}")
    print(f"  Tie windows (ambiguous) : {n_ties}  ({100*n_ties/n_win:.1f}%)")
    print(f"\n  Label distribution:")
    for lbl, cnt in zip(unique, counts):
        tag = '(rest)' if lbl == 0 else ''
        print(f"    label {lbl:3d} {tag:6s}: {cnt:5d} windows  ({100*cnt/n_win:.1f}%)")

    print(f"\n  unrectified windows shape : {windows['unrectified'].shape}")
    print(f"  rectified   windows shape : {windows['rectified'].shape}")

    # ── Temporal dynamic visualisation ────────────────────────────────────────
    # Shows 2 s of data (4000 samples) across three stacked panels:
    #   1. Raw filtered EMG + per-sample label colour band
    #   2. Rectified EMG + sliding window boundary shading
    #   3. Majority-vote label per window as a step function
    plot_temporal_dynamics(
        filtered_emg,
        labels_raw,
        windows,
        channel=0,
        display_samples=4000,
        highlight_windows=10,
    )
