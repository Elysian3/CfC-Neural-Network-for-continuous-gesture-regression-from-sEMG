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
        - 'subject'      : ndarray (1, 1)   – Subject I D
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
    for name in variable_names: # eliminate MATLAB internal terms
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
    emg = emg_raw - np.mean(emg_raw, axis=0)


    b_notch, a_notch = scipy.signal.iirnotch(w0=NOTCH_FREQ, Q=NOTCH_Q, fs=fs)
    emg = scipy.signal.filtfilt(b_notch, a_notch, emg, axis=0)


    sos_bp = scipy.signal.butter(
        N=BP_ORDER, Wn=[BP_LOW, BP_HIGH], btype='bandpass', fs=fs, output='sos'
    )
    emg = scipy.signal.sosfiltfilt(sos_bp, emg, axis=0) # sos_bp stands for second-order coeffcients

    return emg



