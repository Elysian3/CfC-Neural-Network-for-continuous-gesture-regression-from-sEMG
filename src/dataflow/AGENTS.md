<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-23 | Updated: 2026-05-23 -->

# dataflow

## Purpose

The signal processing pipeline that transforms raw sEMG recordings into normalized feature matrices ready for model training. Each module represents one stage in a linear chain, and the output of each stage is the input contract for the next.

## Pipeline: From Raw Signal to Feature Matrix

```
.mat file
  → datapreprocess.py    (load + filter: DC → notch 50Hz → BP 20-450Hz)
  → SwRectify.py         (sliding windows 200ms/50ms + target alignment)
  → feature_extraction.py (requested subset of MAV, MAVS, WL, ZC, SSC, RMS)
  → doa_mapping.py       (optional: 18/22-channel glove → 5 DoAs)
  → model input
```

Each stage is **model-agnostic**: the output is a clean numeric matrix. Sequence construction (sliding windows over feature vectors to create CfC input sequences) happens later in the training code, after data splitting — this is intentional to prevent temporal leakage across train/val/test.

## Key Files

| File | Description |
|------|-------------|
| `datapreprocess.py` | `.mat` loading (`load_data`), EMG filtering (`preprocess_emg`), and constants (`FS=2000`, `BP_LOW=20`, `BP_HIGH=450`) |
| `SwRectify.py` | Sliding window decomposition (`sliding_window`), target alignment (`align_targets_to_windows`), constants (`WIN_MS=200`, `STRIDE_MS=50`) |
| `feature_extraction.py` | EMG feature extraction (`extract_emg_features`), end-to-end pipeline runner (`run_feature_pipeline`), normalization (`fit/apply_feature_normalizer`), regression data prep (`prepare_regression_data`) |
| `doa_mapping.py` | 18-DoF CyberGlove → 5-DoA linear mapping (`apply_linear_doa_mapping`), official DB8 matrix, DB2 fallback remap |

## First-Principles Logic per Stage

### 1. datapreprocess.py — Why these filters, in this order?

- **DC removal first**: A DC offset is unphysical (muscles don't produce DC). It comes from electrode half-cell potential or amplifier bias. Must remove before frequency-domain filtering because a DC step would ring through any IIR filter.
- **50 Hz notch**: Mains power capacitively couples into high-impedance (~MΩ) electrode leads. Q=30 gives ~1.67 Hz bandwidth — the narrowest notch that reliably captures 50 Hz drift without removing physiological EMG energy at adjacent frequencies.
- **20-450 Hz bandpass**: Surface EMG power spectrum peaks around 50-150 Hz and rolls off sharply. Below 20 Hz: electrode movement artifacts (not neural). Above 450 Hz: thermal/electronic noise floor. 4th-order Butterworth chosen for maximally flat passband (no ripple to bias feature extraction).
- **Zero-phase filtering (`filtfilt`)**: The temporal relationship between EMG and movement is the signal the model must learn. A causal filter introduces phase lag — the EMG appears to happen later than it did. `filtfilt` (forward+backward) gives zero phase distortion, preserving the true EMG-to-motion latency.

### 2. SwRectify.py — Why 200ms windows, 50ms stride, "last" alignment?

- **200 ms window**: Neural conduction + electromechanical delay + muscle contraction dynamics mean EMG leads force by ~50-150 ms. A 200 ms window captures both the pre-activation EMG and the subsequent mechanical response. Shorter windows give noisy feature estimates; longer windows blur transient events.
- **50 ms stride (75% overlap)**: Provides smooth feature trajectories at 20 Hz update rate. Sufficient for prosthetic control (users perceive <100 ms delay).
- **Rectified signal retained alongside bipolar**: Rectification (`|x|`) exposes the activation envelope. The bipolar signal preserves zero-crossing information for frequency-proxy features (ZC, SSC).
- **"last" target alignment**: Matches online decoding — at inference time, the system sees EMG up to time T and must predict the joint angle at time T. Training with the same alignment prevents train/serve skew.
- **Target offset (default 200 samples = 100 ms)**: Compensates for electromechanical delay — the EMG at time T predicts the mechanical response at time T+100ms.

### 3. feature_extraction.py — Why these five features?

- **MAV (Mean Absolute Value)**: The rectified EMG mean. Direct proxy for muscle activation level — correlates with force (though the relationship is nonlinear).
- **MAVS (MAV Slope)**: ΔMAV between consecutive windows. Captures whether activation is rising (contraction onset), falling (relaxation), or steady (hold). Critical for dynamic gestures.
- **WL (Waveform Length)**: Cumulative absolute sample-to-sample difference. Aggregates both amplitude and frequency information in one scalar. Sensitive to both activation level and motor unit firing rate.
- **ZC (Zero Crossings)**: Count of sign changes with hysteresis threshold. Crude frequency proxy — higher ZC rate = more high-frequency motor unit content. Threshold suppresses noise-floor crossings.
- **SSC (Slope Sign Changes)**: Count of local turning points. Another frequency/shape proxy, redundantly encoding spectral information from a different angle.

The supported paper/deployment CLI defaults to RMS-only input (12 channels ×
1 feature = 12-D). The reusable API also supports MAV, MAVS, WL, ZC, and SSC
for controlled research ablations. It computes only requested features and
calibrates rest thresholds only when selected ZC/SSC features need them.

### 4. doa_mapping.py — Why 18 DoFs → 5 DoAs?

18 independent CyberGlove joint angles are redundant for grasp classification — most functional grasps (power, precision, lateral, hook, etc.) can be parameterized by 5 Degrees of Actuation. The mapping matrix comes from the official DB8 supplementary materials (`Data_Sheet_1.PDF`). For DB2's 22-channel glove, four fingertip channels (7, 10, 14, 18) are excluded because the DB8 figure marks them as n/a — they measure fingertip contact pressure, not joint angle.

## Working Principles (Applied to Signal Processing)

These are the same principles from the root `AGENTS.md`, applied specifically to this pipeline:

1. **Test-Driven**: Every new feature type in `feature_extraction.py` must have a synthetic-data test in `tests/test_regression_pipeline.py` that validates the numerical output against a hand-computed value. Before merging any filter parameter change: run the full test suite. No passing tests = the change is a hypothesis, not code.

2. **Ask First, Never Guess**: Every filter cutoff frequency, window duration, and feature parameter must have a stated reason. "I saw it in a paper" is not enough — state *why* the paper chose that value from the underlying physiology. If you don't know the reason, ask the user or consult `reference files/` before touching the constant.

3. **Scientific Rigor — Signal & Math**: The `windows` dict is a mathematical contract. Its keys (`unrectified`, `rectified`, `window_start_indices`, etc.) have exact meanings — changing a key name or the shape of its value breaks every downstream consumer. Before modifying any pipeline stage: trace the full chain to verify every consumer still receives the expected contract.

4. **Dual Role**: If you see filter parameters that contradict EMG physiology (e.g., a low-cut above 30 Hz that would remove motor unit firing rate information), say so directly. The pipeline is physics, not opinion.

5. **Engineering Five Steps — Pipeline Edition**:
   - **Question**: Is this filtering/feature/windowing step actually needed, or is it cargo-cult from another paper?
   - **Delete**: Remove dead code paths and avoid computing unrequested features or thresholds.
   - **Simplify**: One module = one pipeline stage. Don't add a new file for a single function that belongs in `feature_extraction.py`.
   - **Accelerate**: Test pipeline changes on one recording, one channel first. Don't run the full dataset.
   - **Automate**: If you find yourself manually checking window shapes or feature ranges, add an assertion.

6. **Explain Every Action**: Every filter constant or pipeline change must come with a reason visible to the next reader — either in the commit message, the docstring, or a comment. "Changed Q from 30 to 50 because the 50 Hz line noise in DB8 recordings shows wider drift (±2 Hz) than DB2 (±1 Hz), so the notch needs to be narrower."

7. **Strict Intent Attribution**: Never fabricate or misattribute a filter design choice to the user. A decision like "we used a 4th-order Butterworth because..." must be traceable — either the user specified it, the reference paper specified it, or the agent derived it from first principles. State which one. "The agent chose 4th-order Butterworth because..." is correct. "As you requested, 4th-order Butterworth..." — when the user never said that — is a violation.

8. **Hardware-gated feature engineering (Principle 8):** The feature vector dimensionality (`len(feature_order) × num_channels`) is the input dimension to the model, which directly controls the first-layer weight matrix size. Adding a feature type adds ~(channels × hidden_units) parameters. Every feature addition must be justified not only by information gain but also by its memory cost on the ESP32-S3 (~512 KB SRAM). A richer feature set that bloats the model beyond the hardware budget is a net regression.

## For AI Agents

### Working In This Directory
- Each module should remain focused on one pipeline stage. Add new features to `feature_extraction.py` rather than creating new modules.
- The `try/except ImportError` pattern at the top of each file supports both package-style imports (`from dataflow import ...`) and direct execution (`python datapreprocess.py`). Maintain this.
- All constants (`FS`, `WIN_MS`, `STRIDE_MS`, filter params) are defined at module level — do not bury them inside functions.
- When adding a new feature type, add it to `SUPPORTED_FEATURES` and the `FEATURE_ORDER` default.

### Testing Requirements
- Tests are in `tests/test_regression_pipeline.py` — verify new pipeline logic there using synthetic arrays.
- The test suite covers: window geometry, target alignment modes, feature extraction with known inputs, and the full `run_feature_pipeline` path.

### Common Patterns
- NumPy arrays use `dtype=np.float32` throughout — matches PyTorch default, avoids unnecessary casts.
- Functions validate inputs explicitly rather than relying on downstream errors.
- The `windows` dict is the inter-stage contract — its keys (`unrectified`, `rectified`, `window_start_indices`, etc.) must remain stable.

## Dependencies

### Internal
- `doa_mapping.py` is imported by `feature_extraction.py` for mapped-target pipelines.
- `datapreprocess.py` constants (`FS`) are imported by `SwRectify.py` and `feature_extraction.py`.

### External
- `scipy.signal` — `butter`, `iirnotch`, `filtfilt`, `sosfiltfilt`, `welch`
- `scipy.io` — `loadmat`, `whosmat`
- `numpy` — all array operations
