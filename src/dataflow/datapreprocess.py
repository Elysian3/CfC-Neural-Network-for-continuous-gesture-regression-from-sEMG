import numpy as np
import scipy.io as sio
import scipy.signal


# ── NinaPro DB2 constants ────────────────────────────────────────────────────
FS = 2000          # Sampling frequency (Hz)
NOTCH_FREQ = 50.0  # Power-line interference frequency (Hz)
NOTCH_Q = 30.0     # Notch quality factor (sharper = higher Q)
BP_LOW = 20.0      # Bandpass lower cutoff (Hz)
BP_HIGH = 450.0    # Bandpass upper cutoff (Hz)
BP_ORDER = 4       # Butterworth filter order


def load_data(file_path: str) -> dict:
    """
    Load data from a .mat file using scipy.io.loadmat.

    Parameters
    ----------
    file_path : str
        Path to the .mat file (e.g., 'src/data/DB2/S1_E1_A1.mat').

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'emg'          : ndarray (N, 12)  – EMG signals
        - 'acc'          : ndarray (N, 36)  – Accelerometer data
        - 'stimulus'     : ndarray (N, 1)   – Stimulus labels
        - 'glove'        : ndarray (N, 22)  – Glove sensor data
        - 'inclin'       : ndarray (N, 2)   – Inclinometer data
        - 'subject'      : ndarray (1, 1)   – Subject ID
        - 'exercise'     : ndarray (1, 1)   – Exercise ID
        - 'repetition'   : ndarray (N, 1)   – Repetition labels
        - 'restimulus'   : ndarray (N, 1)   – Re-stimulus labels
        - 'rerepetition' : ndarray (N, 1)   – Re-repetition labels
    """
    # loadmat parses the binary .mat file and returns a Python dict.
    # struct_as_record=True keeps MATLAB structs as numpy recarrays (safer default).
    raw = sio.loadmat(file_path, struct_as_record=True)

    # Only pull out the signal variables we care about.
    # MATLAB also stores metadata under '__header__', '__version__', '__globals__' — we skip those.
    variable_names = [
        'emg', 'acc', 'stimulus', 'glove', 'inclin',
        'subject', 'exercise', 'repetition', 'restimulus', 'rerepetition',
    ]

    data = {}
    for name in variable_names:
        if name in raw:
            data[name] = raw[name]
        else:
            print(f"Warning: variable '{name}' not found in {file_path}")

    return data


def preprocess_emg(emg_raw: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to raw EMG signals.

    Pipeline (in order):
        1. DC removal  – subtract per-channel mean
        2. Notch filter – remove 50 Hz power-line interference (Q=30)
        3. Bandpass filter – 4th-order Butterworth, 20–450 Hz

    Parameters
    ----------
    emg_raw : ndarray, shape (N, C)
        Raw EMG matrix with N samples and C channels.
    fs : float
        Sampling frequency in Hz (default: 2000).

    Returns
    -------
    ndarray, shape (N, C)
        Preprocessed EMG, same shape as input.
    """
    # ── Step 1: DC removal ───────────────────────────────────────────────────
    # Subtracting the per-channel mean centres each signal around zero, which is
    # required before any frequency-domain filtering.
    emg = emg_raw - np.mean(emg_raw, axis=0)

    # ── Step 2: 50 Hz notch filter ───────────────────────────────────────────
    # Mains power (50 Hz in China/EU) couples into unshielded EMG leads as a
    # strong sinusoidal artefact.  iirnotch() designs a 2nd-order IIR notch
    # (infinite impulse response) at exactly 50 Hz.
    #   w0  – target frequency to suppress (Hz)
    #   Q   – quality factor: bandwidth of the notch = w0/Q.
    #          Q=30 → bandwidth ≈ 1.67 Hz, narrow enough to spare nearby signal.
    # filtfilt() applies the filter forward AND backward (zero-phase), so the
    # output has no time-delay distortion — critical for keeping spike timing.
    b_notch, a_notch = scipy.signal.iirnotch(w0=NOTCH_FREQ, Q=NOTCH_Q, fs=fs)
    emg = scipy.signal.filtfilt(b_notch, a_notch, emg, axis=0)

    # ── Step 3: Bandpass filter (20–450 Hz) ──────────────────────────────────
    # Surface EMG energy lives almost entirely between 20 Hz and 450 Hz:
    #   • Below 20 Hz: motion artefacts and electrode movement noise.
    #   • Above 450 Hz: thermal/electronic noise (and aliasing risk near Nyquist).
    # butter() designs a Butterworth filter — maximally flat passband, no ripple.
    #   N=4    – filter order; higher = steeper roll-off, but more ringing.
    #   output='sos' – second-order sections format, numerically more stable than
    #                  raw b/a coefficients for higher-order filters.
    # sosfiltfilt() is the SOS equivalent of filtfilt: zero-phase, forward+backward.
    sos_bp = scipy.signal.butter(
        N=BP_ORDER, Wn=[BP_LOW, BP_HIGH], btype='bandpass', fs=fs, output='sos'
    )
    emg = scipy.signal.sosfiltfilt(sos_bp, emg, axis=0)

    return emg



if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt

    # Build an absolute path to the .mat file regardless of the working directory.
    # __file__ → this script → go up two levels to reach the project root → into data/.
    mat_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'src', 'data', 'DB2', 'S1_E1_A1.mat'
    )

    # whosmat() lists variable names, shapes, and dtypes without loading the full file —
    # a cheap sanity check before committing to a full load.
    print("Variables in .mat file:")
    print(sio.whosmat(mat_path))
    print()

    data = load_data(mat_path)
    emg_raw = data['emg']
    filtered_emg = preprocess_emg(emg_raw)

    print("Preprocessing complete.")
    for key, value in data.items():
        print(f"{key:15s}  shape={str(value.shape):20s}  dtype={value.dtype}")
    print(f"\nFiltered EMG  shape={str(filtered_emg.shape):20s}  dtype={filtered_emg.dtype}")

    # ── Visualisation: time-domain, all channels ──────────────────────────────
    # Create a DC-removed version of the raw signal as the "before" reference.
    # We compare it against the fully filtered output so you can see the effect
    # of the notch + bandpass on each channel independently.
    emg_dc = emg_raw - np.mean(emg_raw, axis=0)   # DC-only for "raw" reference
    n_channels = emg_raw.shape[1]
    num_points = min(4000, emg_raw.shape[0])       # show ~2 s of data (4000/2000 Hz)
    t = np.arange(num_points) / FS                 # time axis in seconds

    # subplots(rows=n_channels, cols=2): left column = raw, right column = filtered.
    # sharex=True links the x-axes so zooming one panel zooms all.
    fig, axes = plt.subplots(n_channels, 2, figsize=(18, 2.5 * n_channels), sharex=True)
    fig.suptitle("EMG Preprocessing — All Channels\n(left: DC-removed raw | right: notch + bandpass filtered)",
                 fontsize=13, y=1.002)

    for ch in range(n_channels):
        # Left panel: raw (DC-removed only)
        axes[ch, 0].plot(t, emg_dc[:num_points, ch], linewidth=0.4, color='steelblue')
        axes[ch, 0].set_ylabel(f"CH {ch + 1}", fontsize=8)
        axes[ch, 0].grid(True, linewidth=0.3)
        if ch == 0:
            axes[ch, 0].set_title("Raw (DC removed)", fontsize=10)

        # Right panel: fully filtered signal
        axes[ch, 1].plot(t, filtered_emg[:num_points, ch], linewidth=0.4, color='darkorange')
        axes[ch, 1].grid(True, linewidth=0.3)
        if ch == 0:
            axes[ch, 1].set_title("Filtered (notch + bandpass)", fontsize=10)

    for col in range(2):
        axes[-1, col].set_xlabel("Time (s)", fontsize=9)

    plt.tight_layout()
    plt.show()

    # ── Visualisation: Power Spectral Density (channel 1) ────────────────────
    # Welch's method estimates the PSD by averaging the periodograms of overlapping
    # segments (nperseg=1024 samples ≈ 0.5 s per segment at 2000 Hz).
    # The log-scale y-axis lets you see both the dominant 50 Hz spike and the
    # broadband sEMG floor on the same plot.
    # Vertical marker lines confirm that the filters hit their intended frequencies.
    fig_psd, ax_psd = plt.subplots(figsize=(10, 4))
    for signal, label, color in [
        (emg_dc[:, 0],       "Raw (DC removed)",         "steelblue"),
        (filtered_emg[:, 0], "Filtered (notch+bandpass)", "darkorange"),
    ]:
        # welch() returns frequency bins (f) and power estimates (pxx)
        f, pxx = scipy.signal.welch(signal, fs=FS, nperseg=1024)
        ax_psd.semilogy(f, pxx, label=label, color=color, linewidth=1)

    # Reference lines so you can visually verify filter placement
    ax_psd.axvline(50,  color='red',  linestyle='--', linewidth=0.8, label="50 Hz notch")
    ax_psd.axvline(20,  color='gray', linestyle=':',  linewidth=0.8, label="BP low (20 Hz)")
    ax_psd.axvline(450, color='gray', linestyle='-.',  linewidth=0.8, label="BP high (450 Hz)")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (V²/Hz)")
    ax_psd.set_title("Power Spectral Density — Channel 1")
    ax_psd.legend(fontsize=8)
    ax_psd.grid(True, which='both', linewidth=0.3)
    ax_psd.set_xlim(0, FS / 2)   # show up to Nyquist (1000 Hz)
    plt.tight_layout()
    plt.show()
