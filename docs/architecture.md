# Antikythera: DenseCfC + ATL — Complete Architecture & Pipeline Reference

> Generated from systematic code trace of the full training pipeline, June 2026.

---

## 1. FILE MAP

```
src/
├── dataflow/                          ← Raw data → Features (no ML)
│   ├── datapreprocess.py              ← DC removal, notch, bandpass filtering
│   ├── SwRectify.py                   ← Sliding window (200ms/50ms), rectification, last-sample target alignment
│   ├── feature_extraction.py          ← 6 supported EMG features (mav, mavs, wl, zc, ssc, rms)
│   └── doa_mapping.py                 ← 22 glove columns → 5 DoA linear projection
│
├── deep learning/
│   ├── train.py                       ← MODEL DEFINITIONS + low-level training
│   │   ├── CfCTrainingConfig          ← ALL hyperparameter defaults
│   │   ├── DomainDiscriminator        ← MLP(256→128→1+Sigmoid) for GAN ATL
│   │   ├── DenseCfCLinearRegressor    ← DenseCfC + Linear(256→5) head (AutoNCP removed 2026-07)
│   │   ├── SequenceSplit              ← Central data struct (x, y, labels, metadata)
│   │   ├── build_sequence_split()     ← Window sequences → (n_seq, 8, feature_dim) tensors
│   │   ├── train_one_epoch()          ← Standard regression training loop
│   │   ├── evaluate_split()           ← Predict → inverse-norm → compute R²/MAE/RMSE
│   │   ├── compute_regression_metrics() ← Per-DoA + averaged metrics
│   │   └── normalize_sequence_inputs() ← μ-law normalization
│   │
│   ├── run_db2_paper_cfc_finetune.py  ← MAIN PIPELINE + ATL training
│   │   ├── parse_args()               ← CLI interface
│   │   ├── load_subject_filtered_split() ← Full data pipeline per subject
│   │   ├── select_repetition_split()  ← 4 train / 2 test repetitions per action
│   │   ├── _augment_sequence_batch()  ← EMG data augmentation
│   │   ├── _atl_training_epoch()      ← ONE epoch: GAN alternating (L_DD → DD, L_mapping+L_subject → New-t-net)
│   │   ├── fine_tune_head()           ← ATL loop (GAN: Multi-s-net + New-t-net + DD) OR standard head-only FT
│   │   └── main()                     ← Pipeline orchestrator
│   │
│   └── (5 legacy scripts removed 2026-07; utilities migrated to train.py)
│
└── hardwareOperation/
    ├── hardware_preflight.py          ← Quantization, SRAM, and golden-contract gate
    ├── export_weights.py              ← Training-stats-gated header export into firmware/main
    ├── test_cfc_pc.c                  ← Golden test linked with canonical firmware C
    └── test_features_pc.c             ← RMS-to-CfC integration test
firmware/main/                          ← Single source for C inference and ESP-IDF app
```

---

## 2. COMPLETE DATA FLOW (one subject, e.g. S35)

```
S35_E2_A1.mat  (raw .mat file)
│
├─► load_data()
│   └─► emg: (N, 12), glove: (N, 22), restimulus: (N,), rerepetition: (N,)
│
├─► preprocess_emg(emg, fs=2000)
│   ├─► 1. DC removal: emg -= mean(emg, axis=0)
│   ├─► 2. Notch 50Hz: iirnotch(w0=50, Q=30) → filtfilt()
│   └─► 3. Bandpass 20-450Hz: butter(N=4) → sosfiltfilt()
│
├─► For each (action ∈ [18-23]) × (repetition ∈ [1-6]):
│   │
│   ├─► apply_linear_doa_mapping(glove_segment)
│   │   └─► glove(22) @ DOA5_W.T(22×5) → 5 DoA angles
│   │
│   ├─► sliding_window(emg_filtered, targets, fs=2000, window=200ms, stride=50ms)
│   │   ├─► 400-sample windows, 100-sample stride
│   │   ├─► Output: unrectified (N_win, 400, 12) + rectified (N_win, 400, 12)
│   │   └─► Target alignment: last sample of each window, offset=200 samples
│   │
│   └─► extract_emg_features(windows)
│       ├─► mav  = mean(|rectified|, axis=window)        → (N_win, 12)
│       ├─► rms  = sqrt(mean(unrectified², axis=window))  → (N_win, 12)
│       ├─► mavs = diff(mav, axis=window)                 → (N_win, 12)
│       ├─► wl   = sum(|diff(unrectified)|, axis=window)  → (N_win, 12)
│       ├─► zc   = zero_crossings(unrectified)             → (N_win, 12)
│       ├─► ssc  = slope_sign_changes(unrectified)         → (N_win, 12)
│       └─► feature_matrix: (N_win, 12)  ← current RMS-only deployment default
│
├─► build_sequence_split(recordings, seq_len=8, seq_stride=1)
│   └─► Sequences: (N_seq, 8, 12) → target at last window
│       Label: action + repetition tags attached
│
└─► select_repetition_split(split, train_reps_per_action=4, seed=42+subj)
    ├─► train_indices: 4 reps per action (random)
    └─► test_indices:  remaining 2 reps per action
```

**At this point, one subject = two SequenceSplit objects:**

The diagram lists every supported research feature, but extraction is
on-demand. With the current RMS-only CLI default, only RMS is evaluated and the
matrix dimension is 12. MAV/MAVS/WL/ZC/SSC dimensions apply only when an
experiment explicitly selects them; ZC/SSC rest-threshold calibration is also
skipped otherwise.

- `train_split`: x=(~2800, 8, 12), y=(~2800, 5)
- `test_split`:  x=(~1400, 8, 12), y=(~1400, 5)

---

## 3. COMPLETE HYPERPARAMETER TABLE

### 3a. Data Preprocessing

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `fs` | 2000 Hz | EMG sampling rate |
| `NOTCH_FREQ` | 50 Hz | Power-line notch |
| `NOTCH_Q` | 30 | Notch quality factor (BW=1.67Hz) |
| `BP_ORDER` | 4 | Butterworth filter order |
| `BP_LOW` | 20 Hz | High-pass cutoff |
| `BP_HIGH` | 450 Hz | Low-pass cutoff |
| `window_ms` | 200 ms | Sliding window length |
| `stride_ms` | 50 ms | Window stride (75% overlap) |
| `target_alignment` | last-sample | Anchor = window_ends − 1 (hardcoded) |
| `target_offset_samples` | 200 | Forward time shift for alignment |

### 3b. Feature Extraction

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `feature_order` | `("rms",)` | Current paper/deployment CLI default |
| `feature_dim` | 12 (1×12) | Current per-window deployment input |
| `SUPPORTED_FEATURES` | mav, mavs, wl, zc, ssc, rms | All available features |
| ZC/SSC threshold | rest-state RMS (stimulus==0) | Auto-calibrated per-channel via `_compute_rest_thresholds` |
| `rest_percentile` | 10.0 | Fallback percentile when stimulus labels unavailable |
| `rest_threshold_scale` | 1.0 | Multiplier on auto-computed thresholds |

**Why rest-state calibration matters:** The old threshold (`0.01 × global_mean_abs`) was effectively dead — it filtered <6% of sample differences, producing the physiological absurdity of rest ZC > active ZC (bandpass-filtered noise at 450 Hz crosses zero more often than contraction energy at 20-100 Hz). Rest-state thresholds (`stimulus==0` frames → per-channel RMS) give all 5 DoAs positive R² for the first time.

#### RMS vs MAV Comparison (h=256, S1, 400 epochs, patience 30)

| | MAV | RMS | Δ |
|---|-----|-----|---|
| Overall R² | **0.273** | 0.240 | MAV +0.033 |
| RMSE | **21.69** | 22.41 | MAV -0.72 |
| Best epoch | 369 | 389 | — |

| Per-target R² | MAV | RMS | Winner |
|--------------|-----|-----|--------|
| thumb_rotation | -0.271 | **-0.156** | RMS +0.115 |
| thumb_flexion | **0.232** | 0.083 | MAV +0.149 |
| index_flexion | 0.414 | **0.435** | RMS +0.022 |
| middle_flexion | **0.511** | 0.408 | MAV +0.103 |
| ring_little_flexion | **0.481** | 0.430 | MAV +0.051 |

**Verdict**: MAV marginally outperforms RMS (3/5 DoAs, +0.033 overall R²). RMS wins on thumb_rotation and index_flexion but loses on the other three. MAV's rectified-only signal (absolute value) appears slightly more robust for cross-subject sEMG decoding when fully trained. The `--feature-order` CLI arg is preserved for future feature ablation experiments.

### 3c. DoA Mapping

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `target_mapping` | "doa5" | 22 glove → 5 DoA |
| DOA5 targets | thumb_rotation, thumb_flexion, index_flexion, middle_flexion, ring_little_flexion | 5 semantic DoAs |
| Mapping matrix | 5×22 linear weights (DOA5_W) | DB8 official matrix → DB2 remap |
| Paper actions | (18, 19, 20, 21, 22, 23) | 6 gesture classes |

### 3d. Sequence Construction

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `seq_len` | 8 | 550 ms signal coverage: 200 ms + 7×50 ms |
| `seq_stride` | 1 | Sequence stride (dense sampling) |

### 3e. Normalization

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `feature_normalization` | "mu_law" | Feature compression |
| `target_normalization` | "mu_law" | Target compression |
| `mu_law_mu` | 255.0 (G.711 standard) | Compression strength |

Formula (forward): `y = sign(x) × log1p(μ×|x|) / log1p(μ)`  
Formula (inverse): `x = sign(y) × expm1(|y|×log1p(μ)) / μ`

**μ=255 (NOT 2^20):** The previous value 220 was a PDF extraction artifact — superscript "2²⁰" was rendered as "220". μ=2^20 (1,048,576) causes training divergence (val_mae oscillates 35-50, no downward trend) by compressing EMG features too aggressively. μ=255 (ITU-T G.711 μ-law standard) converges normally.
**CRITICAL**: Normalizer stats (center, scale) are fit ONLY on source training data.

### 3f. Repetition Split

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `train_repetitions_per_action` | 4 | Training reps (out of 6) |
| `random_seed` | `42 + subject_number` | Different split per subject |

### 3g. Model Architecture (DenseCfCLinearRegressor)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `model_family` | "dense_cfc_linear" | Dense (no sparsity mask) |
| `hidden_units` | 256 | CfC internal state dimension |
| `input_dim` | 12 | RMS × 12 channels in the current deployment configuration |
| `output_dim` | 5 | 5 DoA targets |
| `cfc_dropout` | 0.3 | Dropout in CfC backbone |

**Architecture:**
```
Input: (batch, 8, 12)
  │
  └─► CfC(12 → 256, backbone_dropout configurable, return_sequences=True)
      │
      ├─► backbone: Linear(268, 128)   [weight: (128, 268)]
      ├─► ff1:      Linear(128, 256)   [weight: (256, 128)]
      ├─► ff2:      Linear(128, 256)   [weight: (256, 128)]
      ├─► time_a:   Linear(128, 256)   [weight: (256, 128)]
      └─► time_b:   Linear(128, 256)   [weight: (256, 128)]
      
      Output per step: y_sequence = (batch, 8, 256)
  │
  ├─► y_sequence[:, -1, :]          → final_state: (batch, 256)
  ├─► Dropout(0.3)(final_state)     → dropped: (batch, 256)
  └─► Linear(256, 5)(dropped)       → prediction: (batch, 5)
```

**Parameter counts by size:**

| h | backbone | ff1 | ff2 | time_a | time_b | head | Total |
|---|----------|-----|-----|--------|--------|------|-------|
| 128 | 18,048 | 16,512 | 16,512 | 16,512 | 16,512 | 645 | 84,741 |
| 256 | 34,432 | 33,024 | 33,024 | 33,024 | 33,024 | 1,285 | 167,813 |
| 512 | 67,200 | 66,048 | 66,048 | 66,048 | 66,048 | 2,565 | 333,957 |

### 3h. Pretraining Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `learning_rate` | 1e-4 | AdamW learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `batch_size` | 128 | Training batch size |
| `max_epochs` | 400 | Maximum epochs |
| `early_stopping_patience` | 30 | Stop if val MAE worsens 30 epochs |
| `gradient_clip_norm` | 1.0 | Max gradient L2 norm |
| `loss_fn` | MSELoss | Regression loss |

### 3i. Data Augmentation (pretraining only)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `augment_prob` | 0.5 | Apply augmentation probability |
| `amplitude_range` | (0.7, 1.3) | Per-channel scaling |
| `noise_std` | 0.05 | Gaussian noise σ (relative to channel std) |
| `time_shift_max` | 2 | Max ±2 step random shift |

### 3j. ATL (GAN-Style, Lin & He 2024 Section 3.3)

**Architecture:**
```
Multi-s-net (frozen, eval) ──→ F_s ──→ DD ──→ src_pred (source=0)
New-t-net  (trainable)      ──→ F_t ──→ DD ──→ tgt_pred (target=1)
                             ──→ pred_t ──→ L_subject = w × MSE(pred_t, target_y)
                             F_t ──→ DD ──→ L_mapping = -log(DD(F_t))
```

**Losses (Eqs 1.8-1.11):**
- `L_DD = -log(DD(F_s)) - log(1 - DD(F_t))` → optimizes DD (distinguish source vs target)
- `L_mapping = -log(DD(F_t))` → optimizes New-t-net (standard GAN generator loss — fool DD)
- `L_subject = w × MSE(pred_t, target_y)` → optimizes New-t-net (regression on target data)

**Training loop** (two optimizer steps per batch, standard GAN pattern):
1. Forward F_s (no_grad) + F_t → L_DD.backward() → dd_optimizer.step()
2. Recompute F_t (fresh graph) → (L_mapping + L_subject).backward() → target_optimizer.step()

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `fine_tune_epochs` | 30 | Max ATL epochs |
| `cfc_atl_lr` | 1e-4 | New-t-net learning rate |
| `dd_lr` | 1e-4 | DD learning rate |
| `atl_subject_weight` (w) | 1.0 | Target regression weight (Eq 1.11) |
| `dd_hidden` | 128 | DD hidden layer size |
| `ft_early_stopping_patience` | 5 | Stop if val MAE worsens 5 epochs |

### 3k. GAN Training Dynamics

Standard GAN generator/discriminator alternating training. No gradient reversal
layer — the adversarial signal flows through L_mapping as a separate loss term.
DD gradients from step 2's backward accumulate but are cleared by step 1's
`zero_grad()` at the start of the next iteration. New-t-net is recomputed in
step 2 because step 1's backward frees the autograd graph through F_t.

---

## 4. PRETRAINING FLOW

```
train_model_with_validation(source_train, source_val, config, ...)
│
├─► normalize both splits with source-fitted μ-law stats
├─► build_cfc_regressor(input=12, output=5, hidden=256, family="dense_cfc_linear", dropout=0.3)
├─► optimizer = AdamW(lr=1e-4, wd=1e-4)
├─► loss_fn = MSELoss()
│
└─► for epoch in 1..400:
    │
    ├─► for (x_batch, y_batch) in DataLoader(source_train, batch=128, shuffle=True):
    │   ├─► if random() < 0.5:
    │   │   └─► _augment_sequence_batch(x): scale(0.7-1.3)× + noise(5%) + shift(±2)
    │   ├─► pred = model(x_batch)
    │   ├─► loss = MSE(pred, y_batch)
    │   ├─► loss.backward()
    │   ├─► clip_grad_norm_(1.0)
    │   └─► optimizer.step()
    │
    ├─► val_metrics = evaluate_split(model, source_val, ...)
    ├─► if val_mae improved: save best_state
    └─► if patience ≥ 30: early stop
```

---

## 5. ATL TRAINING FLOW (GAN-Style, Lin & He 2024 Section 3.3)

### 5a. Initialization

```
fine_tune_head(model, support_split, source_split, enable_atl=True, ...)
│
├─► source_model = deepcopy(model) → freeze, eval()    # Multi-s-net
├─► target_model = deepcopy(model) → trainable, train() # New-t-net (warm-start)
├─► dd = DomainDiscriminator(in=256, hidden=128)
│     └─► Linear(256,128) → BN → ReLU → Linear(128,1) → Sigmoid
├─► target_optimizer = AdamW(target_model.parameters(), lr=1e-4)
├─► dd_optimizer     = AdamW(dd.parameters(),          lr=1e-4)
└─► source_loader / target_loader (batch=128, shuffle)
```

### 5b. One ATL Epoch (GAN Alternating)

```
for each target_batch:
    source_x, _ = next(source_loader)

    # Multi-s-net: frozen deterministic features
    with torch.no_grad():
        _, F_s = source_model.forward_with_features(source_x)

    # ── Step 1: Train DD ──
    pred_t, F_t = target_model.forward_with_features(target_x)
    L_DD = -log(DD(F_s)) - log(1 - DD(F_t))
    L_DD.backward() → dd_optimizer.step()

    # ── Step 2: Train New-t-net (RECOMPUTE F_t — Step 1 freed the graph) ──
    pred_t, F_t = target_model.forward_with_features(target_x)
    L_mapping = -log(DD(F_t))
    L_subject = w × MSE(pred_t, target_y)
    (L_mapping + L_subject).backward() → target_optimizer.step()
```

### 5c. Gradient Flow (No GRL)

```
Step 1: L_DD gradients → DD params (direct, normal BP)
        L_DD gradients → F_t → target_model (accumulated, cleared by Step 2's zero_grad)

Step 2: L_mapping gradients → F_t → target_model (pulls features to fool DD)
        L_subject gradients → pred_t → target_model (regression)
        L_mapping gradients → DD params (accumulated, cleared by next iter's dd_opt.zero_grad)
```

New-t-net receives two aligned gradient forces:
- **L_mapping**: pushes F_t to be classified as source (adversarial)
- **L_subject**: pushes pred_t toward ground-truth angles (regression)

No gradient sign reversal — the adversarial signal is an explicit loss term (standard GAN generator loss), not a GRL hack.

### 5d. Validation & Early Stopping

```
target_model.eval()
val_mae = evaluate_split(target_model, val_split, ...)["mae_mean"]
target_model.train()
if val_mae < best_val_mae: save best_state
if patience ≥ 5: early stop
```

---

## 6. EVALUATION METRICS

```
evaluate_split(model, split, target_stats, ...)
│
├─► predict_sequences(model, x, batch_size, device)
│
├─► y_pred = inverse_target_normalizer(y_norm_pred, target_stats)
├─► y_true = inverse_target_normalizer(y_norm, target_stats)
│   └─► Both back in original angle space (degrees)
│
└─► compute_regression_metrics(y_true, y_pred)
    │
    ├─► mae = mean(|pred - true|, axis=0)
    ├─► rmse = sqrt(mean((pred-true)², axis=0))
    ├─► r2 = 1 - SS_res / SS_tot             [per DoA, can be negative]
    └─► r2_mean = mean(r2)                   [overall metric]
```

**R² interpretation**: 1.0 = perfect prediction, 0 = predicts mean, <0 = worse than guessing the mean.

---

## 7. ESP32-S3 INT8 DEPLOYMENT

| | h=128 | h=256 | h=512 |
|---|-------|-------|-------|
| Params | 90,885 | 173,957 | 340,101 |
| FP32 | 355 KB | 680 KB | 1,328 KB |
| INT8 | 91 KB | 173 KB | 338 KB |
| SRAM margin (400KB) | 309 KB | 227 KB | 62 KB |
| MACs/inference | 717K | 1.37M | 2.69M |
| Latency (SIMD) | 1.0 ms | 1.9 ms | 3.7 ms |
| Power @20Hz | 2.0 mW | 3.8 mW | 7.4 mW |

**Method**: Per-tensor symmetric INT8: `w_q = clamp(round(w × 127/max_abs), -127, 127)`  
**Error**: h=256 measured at 3.6% relative error (within 5% deployability threshold)  
**Verdict**: h=256 is the optimal deployability point. h=512 is feasible but tight (62KB SRAM headroom).

---

## 8. CROSS-SUBJECT RESULTS (h=256 ATL, λ_cap=0.25, dd_lr=cfc_lr=1e-4)

### Current Best: S1 (July 2026, μ=255, rest-state ZC/SSC)

| DoA | R² |
|-----|-----|
| thumb_rotation | +0.570 |
| thumb_flexion | +0.672 |
| index_flexion | +0.593 |
| middle_flexion | +0.679 |
| ring_little_flexion | +0.777 |
| **Mean** | **0.658** |

All 5 DoAs positive R² for the first time. Previous S1 R²=0.698 (May 2026) was inflated by broken ZC/SSC thresholds (effectively zero, counting noise as signal) and μ=220 artifact.

### 31-Subject Pretraining (June 2026, held-out S1/S3/S5)

| Subject | ATL R² |
|---------|--------|
| S1 | 0.650 |
| S3 | 0.512 |
| S5 | 0.496 |
| **Mean** | **0.553** |

### Historical: 7-Subject (May 2026, λ_cap=0.3, old ZC/SSC, μ=220)

| Subject | Zero-shot R² | ATL R² | ΔR² |
|---------|-------------|--------|------|
| S1 | 0.273 | **0.698** | +0.425 |
| S35 | -0.051 | **0.557** | +0.609 |
| S40 | -0.029 | **0.554** | +0.583 |
| S37 | -0.228 | **0.459** | +0.687 |
| S38 | 0.016 | **0.408** | +0.392 |
| S39 | -0.292 | **0.399** | +0.691 |
| S36 | -0.546 | **0.369** | +0.915 |

**7-subject mean: 0.492 ± 0.110** (Note: these values predate ZC/SSC rest-state calibration and μ=255 fix; not directly comparable to 0.658 baseline.)

Per-action analysis reveals index_flexion and middle_flexion have systematically weaker transfer, while ring_little_flexion transfers best.

---

## 9. KNOWN ISSUES

1. **λ schedule hardcoded**: `epoch/10.0` denominator means schedule saturates by epoch 5 regardless of total ATL epochs. For runs >10 epochs, λ is flat for the majority of training. Should be parameterized as `epoch/max_epochs`.

2. **Early stop ignores DD balance**: Validation uses MSE only. The loop stops when regression plateaus, even if domain adaptation hasn't converged. A composite score (e.g., MSE + α × |DD_acc - 0.5|) would be more principled.

3. **High per-action variance**: thumb_rotation on actions 19/22/23 consistently underperforms — confirmed as structural EMG limitation (mixed-sign glove columns c0:+0.639, c3:-0.639 cause gradient cancellation in shared DoA loss). The glove_columns approach (predict 13 individual columns, map post-hoc) was tested and underperformed (R²=0.505 vs doa5 0.658).

4. **Single-subject hyperparameter tuning risk**: All hyperparameters (λ_cap=0.25, dd_lr=1e-4) were optimized on S1. Optimal values may differ for other subjects. Multi-subject sweeps would be more robust but are computationally expensive.

5. **Run-to-run variance ~0.05 R²**: Even with identical settings, single runs are unreliable. Multiple repeats or sweeps needed for confident conclusions. No infrastructure for this currently.

### Resolved (was in previous Known Issues)

| Issue | Resolution |
|-------|-----------|
| DD wins adversarial game (95%+ accuracy) | Experimentally shown to be normal at low λ. Hypothesis that "DD winning = broken" was falsified (June 2026). Lower λ_cap gives strictly better R². |
| Pre-mortem monitoring was dead code | Removed from fine_tune_head() (June 2026). The intervention (capping λ when DD overfits) was actively harmful — decreased λ when DD was winning, which is the wrong direction. |
| dd_lr=1e-3 vs cfc_lr=1e-4 (10:1 ratio) | Equalized at 1e-4/1e-4 (June 2026). The 10:1 ratio had no basis in DANN literature. |
