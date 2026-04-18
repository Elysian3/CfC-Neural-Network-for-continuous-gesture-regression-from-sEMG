# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Antikythera** is a research project implementing a **Liquid Time-Constant (LTC) Network + Closed-form Continuous-time (CfC) model** for real-time sEMG (surface electromyography) decoding on edge systems. The goal is gesture/movement recognition from EMG signals, intended for prosthetics or human-machine interfaces.

## Running Scripts

All scripts should be run from the project root (`D:\Project Antikythera\`) to ensure relative paths resolve correctly.

```bash
# Run the main entry point
python main.py

# Run the core data preprocessing / visualization script
python src/models/datapreprocess.py

# Run the LNN training script
python src/models/train.py

# Run the lower-limb LNN test
python test/TestOfLnn.py

# Run the image classification baseline test
python test/test1onimage.py
# or
python test/review.py
```

**Windows note:** `DataLoader` must use `num_workers=0` on Windows (already set in existing code).

## Installing Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `pytorch-lightning`, `ncps` (Neural Circuit Policies — provides `LTC` and `CfC`), `numpy`, `scipy`, `matplotlib`, `seaborn`, `torchvision`.

## Architecture

### Data Pipeline

Two datasets are used:

1. **NinaPro DB2** (`src/data/DB2/*.mat`) — Upper-limb sEMG, 12-channel, 2000 Hz. `.mat` files loaded via `scipy.io.loadmat`. Fields: `emg` (N×12), `stimulus`, `repetition`, `restimulus`, `rerepetition`, `acc`, `glove`, `inclin`.
2. **Lower Limb Test Data** (`src/data/LowerLimbTestData/*.txt`) — Plain-text files (e.g., `1Amar.txt`, `1Apie.txt`, `1Asen.txt`). Last column is the label/target; preceding columns are features. Loaded with `numpy.loadtxt`.

Preprocessing for DB2 EMG (`src/models/datapreprocess.py`):
- Mean-center the raw EMG
- 4th-order Butterworth bandpass filter (20–450 Hz, `scipy.signal.butter` + `sosfiltfilt`)

### Model

`CfC` (Closed-form Continuous-time) from the `ncps` library, wired with `AutoNCP`:
- `AutoNCP(units, out_features)` auto-generates a Neural Circuit Policy wiring
- Wrapped in a `SequenceLearner(pl.LightningModule)` for PyTorch Lightning training
- Loss: MSE (`nn.MSELoss`)
- Optimizer: Adam, lr=0.01, gradient clip = 1.0
- Trainer: `accelerator="auto"`, `devices="auto"` (auto-selects CUDA/CPU), logs to `log/` via `CSVLogger`

### Hardware Deployment

`src/hardwareOperation/load-model.cpp` — C++ stub for loading the trained model on an edge device (currently a placeholder).

### Test Scripts

- `test/TestOfLnn.py` — Main integration test for the LNN pipeline on lower-limb data; contains the full `SequenceLearner` class (currently data loading is implemented, training loop commented out).
- `test/test1onimage.py` / `test/review.py` — Baseline experiments with a standard MLP on FashionMNIST (used for framework validation, not the core sEMG task).

## Key Conventions

- All paths in scripts are constructed relative to each file's `__file__` location using `os.path` — do not hardcode absolute paths.
- Training logs are written to `log/` at the project root (PyTorch Lightning `CSVLogger("log")`).
- The `src/models/train.py` file has a syntax error (`def main()` missing colon — fix before running).
