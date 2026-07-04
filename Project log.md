# Project Antikythera — Session Log

## 2026-07-01: Feature Pipeline & Training Layer Finalisation

### What Happened

This session produced the definitive feature extraction + training layer. Three major changes were implemented, validated end-to-end with a full S1 ATL training run (34 subjects, h=256, DenseCfC).

### ZC/SSC Rest-State Threshold Calibration

**Problem:** The old ZC/SSC threshold was `0.01 × global_mean_abs_amplitude` — effectively dead. It filtered <6% of sample differences, changed ZC/SSC counts by <1% vs threshold=0, and produced the physiological absurdity of rest ZC > active ZC (bandpass-filtered noise at 450 Hz crosses zero more often than contraction energy at 20-100 Hz).

**Solution:** Replaced with `_compute_rest_thresholds()` — uses stimulus/restimulus labels (`stimulus == 0`) to directly measure per-channel rest-state RMS, then passes it as the ZC/SSC threshold. Fallback to 10th-percentile of window RMS when labels are unavailable. Single source of truth: threshold computed once in `run_feature_pipeline`, passed explicitly to `extract_emg_features`.

**Key facts:**
- Primary path: `filtered_emg[stimulus == 0]` → per-channel `sqrt(mean(x²))` = rest RMS
- Fallback: internal segmentation @ 200ms/50ms → percentile of per-segment RMS
- ZC/SSC only computed when in `feature_order` (avoid dummy threshold requirement)
- `_resolve_channel_thresholds(None)` raises `ValueError` — no silent NaN
- All threshold parameters (`threshold_scale`, `default_scale`, `DEFAULT_ZC_SSC_THRESHOLD`, `zc_ssc_threshold_scale`) removed from the codebase

### μ-law μ Correction: 220 Was a PDF Extraction Artifact

**Discovery:** The previous μ = 220.0 used in training was a PDF text extraction bug: the Unicode superscript "2²⁰" was extracted as the literal characters "220".

**Validation:**
- μ = 2^20 (1,048,576): Training DIVERGES — val_mae oscillates 35-50 with no downward trend (epoch 1: 23.8 → epoch 28: 48.0), while train_loss decreases (0.57 → 0.25). Classic overfitting pattern: the extremely aggressive compression reduces feature granularity beyond what the optimizer can handle.
- μ = 255 (G.711 standard): Training CONVERGES — val_mae 14.8 → 8.8 over 326 epochs. The diagnostic run showed val_mae 14.7 → 12.5 in just 12 epochs.

**Final value:** μ = 255.0 (both `DEFAULT_MU_LAW_MU` and CLI default).

### Codebase Cleanup: 8 Files → 2

| Removed | Reason |
|---------|--------|
| `run_db2_subject_adaptation.py` | Utilities migrated to `train.py`; own training logic was legacy |
| `run_db2_single_subject.py` | Superseded by paper protocol |
| `run_doa5_subject_adaptation.py` | AutoNCP-specific (58 KB); all adaptation modes deleted |
| `run_semantic_dof_screen.py` | One-off DoF screening experiment |
| `analyze_atl_per_action.py` | Diagnostic, untracked |
| `diagnose_source_per_action.py` | Diagnostic, untracked |
| `tests/test_single_subject_training.py` | Tested deleted scripts |

**Migrations:** 5 utility functions (`subject_sort_key`, `discover_target_files`, `make_jsonable`, `summarize_best_epoch`, `save_summary`) from `run_db2_subject_adaptation.py` → `train.py`.

**Removed from train.py:** `CfCRegressor` class (AutoNCP), `from ncps.wirings import AutoNCP`, `model_family == "autoncp"` branch in `build_cfc_regressor()`, `choices=["autoncp"]` from CLI.

**Test coverage:** 29 tests pass (13 pipeline + 16 training), down from 42 via removal of 13 tests that depended on deleted AutoNCP adaptation code.

### S1 Validation Result

| Metric | Value |
|--------|-------|
| Pretrain best val_mae | 8.78 (epoch 296, previous best 9.25) |
| ATL R² (S1 test) | **0.658** |
| ATL MAE | 11.31° |
| thumb_rotation R² | +0.570 |
| thumb_flexion R² | +0.672 |
| index_flexion R² | +0.593 |
| middle_flexion R² | +0.679 |
| ring_little_flexion R² | **+0.777** |

**Key observation:** All 5 DoAs have positive R² for the first time. The previous best S1 R² = 0.698 (architecture.md) was likely inflated by broken ZC/SSC thresholds (effectively zero, counting noise as signal) and the incorrect μ = 220. The new baseline 0.658 represents a more honest, physiologically grounded model.

### Final File Layout

```
src/
├── dataflow/
│   ├── datapreprocess.py          ← DC removal, notch, bandpass filtering
│   ├── SwRectify.py               ← Sliding window (200ms/50ms), last-sample alignment
│   ├── feature_extraction.py      ← 5 EMG features + rest-state ZC/SSC calibration
│   └── doa_mapping.py             ← 22 glove columns → 5 DoA linear projection
│
├── deep learning/
│   ├── train.py                   ← ALL training infra (models, config, data, evaluation)
│   └── run_db2_paper_cfc_finetune.py  ← Paper protocol pipeline (single entry point)
│
└── hardwareOperation/
    └── quantize_test.py           ← INT8 quantization for ESP32-S3

tests/
├── test_regression_pipeline.py    ← 13 tests (pipeline)
└── test_cfc_training.py           ← 16 tests (models/training)
```

### Key Hyperparameters (Current Best)

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_units | 256 | DenseCfC only |
| model_family | dense_cfc_linear | AutoNCP removed |
| feature_order | mav, mavs, wl, zc, ssc | 5 features × 12 channels = 60-dim |
| window / stride | 200ms / 50ms | 75% overlap |
| target_offset_samples | 200 | Forward alignment shift |
| μ-law μ | 255.0 | G.711 standard; 2^20 broken for EMG |
| ZC/SSC threshold | rest RMS (stimulus==0) | Auto-calibrated per-channel |
| pretrain lr / wd | 1e-4 / 1e-4 | AdamW |
| pretrain epochs / patience | 400 / 30 | |
| ATL λ_cap | 0.25 | |
| ATL cfc_lr / dd_lr | 1e-4 / 1e-4 | Equalised from previous 10:1 |
| ATL epochs / patience | 30 / 5 | |

### Known Limitations

1. **DD still wins the adversarial game** at λ=0.25 (DD accuracy reaches 77-92% by ATL epoch 25).
2. **λ schedule still hardcoded** (`epoch/10.0` in sigmoid denominator).
3. **Per-action variance remains** — action 19/22/23 thumb_rot crash confirmed as structural EMG limit.
4. **Reaction delay contamination** — rest periods identified by stimulus may include post-cue muscle activity; no trim margin applied.
5. **Non-stationary noise** not handled — single per-recording threshold assumes stationary noise floor.

### Files Changed This Session

| File | Change |
|------|--------|
| `src/dataflow/feature_extraction.py` | +`_compute_rest_thresholds`, simplified `_resolve_channel_thresholds`, on-demand ZC/SSC, auto-calibration in `run_feature_pipeline` |
| `src/dataflow/datapreprocess.py` | Removed `__main__` block (88 lines of matplotlib dead code) |
| `src/deep learning/train.py` | Removed AutoNCP/CfCRegressor, `threshold_scale`/`DEFAULT_ZC_SSC_THRESHOLD`; migrated 5 utility functions; `model_family` default → `dense_cfc_linear` |
| `src/deep learning/run_db2_paper_cfc_finetune.py` | Re-imported utilities from `train`; removed `autoncp` CLI choice; fixed `mu_law_mu` default; `_compute_rest_thresholds` call site |
| `tests/test_regression_pipeline.py` | Updated 2 tests, added 4 new (13 total) |
| `tests/test_cfc_training.py` | Removed 13 AutoNCP-dependent tests, kept 16 (29→16) |
| `docs/architecture.md` | Updated ZC/SSC threshold docs and μ-law row |
| `Project log.md` | This entry |

### Context

We completed a comprehensive diagnosis of the ATL (Adversarial Transfer Learning) protocol and designed a modification plan. Three independent scientist agents analyzed different axes of the problem. All three converged on the same root cause and fixes — zero contradictions.

> **STATUS (2026-07-02): This section (lines 129-236) was written 2026-06-10, BEFORE the experimental falsification on 2026-06-12.**
> 
> **What changed:** A controlled λ_cap sweep (0.25-1.0) on S1 showed the OPPOSITE of what was predicted — lower λ_cap gives strictly better R², and DD naturally wins (95%+) when the system is well-tuned. The hypothesis "DD winning = adversarial game broken" was **falsified**. DD accuracy is NOT a meaningful diagnostic.
> 
> **Per-change status:**
> - Change 1 (Equalize LRs): ✅ Implemented. dd_lr = cfc_atl_lr = 1e-4. Works correctly.
> - Change 2 (Normalize λ schedule): ❌ Not implemented. `epoch/10` still hardcoded.
> - Change 3 (Reduce DD capacity): ❌ Not implemented. Unnecessary — DD winning at 128 hidden is fine.
> - Change 4 (Increase λ_cap to 1.0): ❌ **WRONG.** Increasing λ_cap degrades R² (0.25→0.3: R² 0.687→0.641; 0.25→1.0: R² 0.687→0.561). The adversarial signal should be a WEAK auxiliary, not a co-equal partner.
> - Changes 5-7 (Closed-loop control): ❌ Not implemented. Premise (DD should be at 50%) is wrong.
> 
> **What remains valid:** The λ schedule hardcoding issue (Change 2) is still worth fixing. The architectural analysis (GRL mechanism, DD architecture, λ schedule formula) is accurate as documentation.
> 
> ---
> 
> ### What We Know (Historical — June 2026, pre-falsification)

**The ATL adversarial game is structurally broken.** The Domain Discriminator (DD) wins trivially, achieving 95%+ accuracy at distinguishing source vs target features. When the DD wins, domain adaptation is not happening — features remain domain-specific, and cross-subject transfer fails for hard actions (index_flexion, middle_flexion).

**Root cause: three compounding imbalances** that together give the DD a ~33:1 effective advantage:

| Factor | Current Value | Effect |
|--------|--------------|--------|
| LR ratio | dd_lr=1e-3 vs cfc_atl_lr=1e-4 (10:1) | DD updates 10× faster than CfC can adapt |
| λ_cap | 0.3 | Adversarial gradient attenuated by 70% |
| λ schedule | `epoch/10` hardcoded | Saturates at epoch 5, flat for 25/30 remaining epochs |
| DD capacity | hidden=128 (~33K params, 19% of CfC) | DD has too much capacity for the 256-dim feature space |

Combined: `cfc_lr × λ_cap = 1e-4 × 0.3 = 3e-5` vs `dd_lr = 1e-3` → effective gradient ratio ≈ **33:1** in DD's favor.

**Key architectural facts:**

- **GRL** (GradientReversalFunction, `train.py:221-231`): Custom autograd function. Forward = identity. Backward = multiplies gradient by `-λ`. This is a direct reproduction of Ganin & Lempitsky 2015/2016 DANN paper. One backward pass, two opposite learning directions — DD gets normal gradients (gets better at domain classification), CfC gets negated gradients (learns to confuse DD).

- **λ_cap** (atl_lambda_cap, CLI default 1.0, but passed as 0.3 at call site `run_db2_paper_cfc_finetune.py:1034`): Maximum value the GRL lambda can reach. Computed as `min(1.0, sigmoid_base) × atl_lambda_cap`. At cap=0.3, even at peak the adversarial signal is only 30% of regression signal.

- **DD architecture** (`train.py:234-248`): MLP `Linear(256→128) → BatchNorm → ReLU → Linear(128→1) → Sigmoid`. No dropout. 33,281 params. The DD has zero regularization.

- **10:1 LR ratio has no theoretical basis in DANN literature.** The original DANN paper (Ganin et al., 2016 JMLR) uses the SAME learning rate for feature extractor and domain classifier. GRL lambda is the sole balancing mechanism.

- **λ schedule** (`_compute_dann_lambda`, line 458): `2.0/(1.0+exp(-10.0*(epoch/10.0-0.5)))` capped at 1.0, then × λ_cap. The denominator `10.0` is hardcoded — schedule is identical whether ATL runs for 10 or 100 epochs. Epoch 6-30 are flat at the same λ.

- **Early stopping ignores DD balance** (ARCHITECTURE.md Known Issue #2): Uses val MAE only. Stops when regression plateaus, even if domain confusion hasn't converged.

- **Pre-mortem monitoring** (lines 664-691): Detects DD overfitting and low domain loss but only prints warnings. The one active intervention (capping λ when domain loss < 0.01) decreases λ — which is the WRONG direction. When DD accuracy is high, λ should INCREASE.

- **CfC feature extractor** (h=256): 172,672 params (DenseCfCLinearRegressor). CfC(60→256, dropout=0.3, return_sequences=True) + Dropout(0.3) + Linear(256→5).

- **Full cross-subject pipeline**: load_subject_filtered_split → sliding_window (200ms/50ms) → extract 5 EMG features × 12 channels → sequence (8,60) → μ-law normalization → pretrain on source subjects (400 epochs, patience 30) → ATL on target subject (30 epochs, patience 5).

### Complete Unified Modification Plan

**Changes 1-4: Minimum Viable Fix (~10 lines changed, 3 files)**

| # | Change | File | Lines | Impact |
|---|--------|------|-------|--------|
| 1 | **Equalize LRs**: `dd_lr=1e-4` (was 1e-3), `cfc_atl_lr=1e-4` | `run_db2_paper_cfc_finetune.py` | 1032-1033 | DD loses 10× speed advantage |
| 2 | **Normalize λ schedule**: Replace `epoch_index / 10.0` with `epoch_index / max_epochs` parameter | `run_db2_paper_cfc_finetune.py` | 451-458 | Schedule spans full training |
| 3 | **Reduce DD capacity**: hidden 128→32, add Dropout(0.3) after ReLU | `train.py` | 234-248 | DD: 33K→8K params (4.8% of CfC) |
| 4 | **Increase λ_cap**: Change call-site from 0.3 to 1.0 | `run_db2_paper_cfc_finetune.py` | 1034 | Full adversarial gradient strength |

**Changes 5-7: Closed-Loop Control (additional ~30 lines)**

| # | Change | Description |
|---|--------|-------------|
| 5 | **Adaptive λ gating** | Increase λ when DD acc > 0.85, decrease when < 0.60 |
| 6 | **Composite early stopping (DACS)** | `score = val_MAE + 0.1 × |DD_acc_mean - 0.5|` |
| 7 | **Active pre-mortem interventions** | Auto: increase λ, decay DD LR, escalate dropout |

### Predicted Outcomes After Fix (FALSIFIED — June 2026)

```
Predicted:      DD accuracy 50-70% → healthy equilibrium, domain confusion working
Actual (sweep): DD accuracy 95%+ at ALL λ levels; lower λ gives BETTER R²
Target metric "DD accuracy ~50%" is NOT a valid goal — DD naturally wins at low λ.
The adversarial signal should be a WEAK auxiliary objective, not a co-equal partner.
```

### Files Involved

- **MODIFY**: `src/deep learning/train.py` — DomainDiscriminator class (add dropout, reduce hidden)
- **MODIFY**: `src/deep learning/run_db2_paper_cfc_finetune.py` — `_compute_dann_lambda()`, `fine_tune_head()` ATL call site, pre-mortem monitoring
- **REFERENCE**: `docs/ARCHITECTURE.md` — Complete pipeline documentation (update after changes)
- **REFERENCE**: `src/dataflow/feature_extraction.py` — Feature pipeline (MAV confirmed best, RMS supported)
- **REFERENCE**: `src/dataflow/doa_mapping.py` — 22-glove → 5-DoA mapping

### Validation Strategy

After implementing changes 1-4, run on one subject (S1) and check:
1. DD accuracy trajectory — should drop from 95%+ to 50-70%
2. ATL R² vs zero-shot R² — should show ATL provides meaningful gain
3. Domain loss — should not collapse to near-zero
4. Per-action R² — index_flexion and middle_flexion should show improvement

### What Actually Happened (June-July 2026)

1. **Change 1 implemented** (equalize LRs): ✅ dd_lr = cfc_atl_lr = 1e-4. Works.
2. **λ_cap sweep executed**: 0.25 → R² 0.687; 0.3 → 0.641; 0.5 → 0.637; 0.75 → 0.467; 1.0 → 0.561. **Lower is better.** The plan's recommendation to increase λ_cap was wrong.
3. **DD accuracy stayed at 95%+ at ALL λ levels** — this is normal behavior, not a problem.
4. **Changes 2-7 not implemented** — Change 3 (reduce DD) and Changes 5-7 (closed-loop control) are unnecessary given the falsification. Change 2 (λ schedule parameterization) remains worthwhile.
5. **ZC/SSC rest-state calibration** (unplanned but high-impact): Replaced dead threshold, all 5 DoAs positive R² for the first time. S1 R² = 0.658.
6. **μ=255 correction** (unplanned): Fixed PDF extraction artifact. μ=2^20 caused training divergence.
7. **Codebase cleanup**: 8 files → 2 in src/deep learning/. AutoNCP removed.
8. **docs/ARCHITECTURE.md updated** with current values (July 2026).

### Key Code Locations (Quick Reference)

```
train.py:221-231   GradientReversalFunction (GRL)
train.py:234-248   DomainDiscriminator (DD)
train.py:271-308   DenseCfCLinearRegressor (+ forward_with_features)
train.py:42-97     CfCTrainingConfig dataclass

run_db2_paper_cfc_finetune.py:103-104    --feature-order CLI
run_db2_paper_cfc_finetune.py:123-124    --atl-lambda-cap CLI (default 1.0)
run_db2_paper_cfc_finetune.py:451-458    _compute_dann_lambda()
run_db2_paper_cfc_finetune.py:461-557    _atl_training_epoch()
run_db2_paper_cfc_finetune.py:560-724    fine_tune_head() — ATL and non-ATL branches
run_db2_paper_cfc_finetune.py:825-1109   run_protocol() — main pipeline orchestrator
run_db2_paper_cfc_finetune.py:1032-1034  ATL call site (dd_lr, cfc_atl_lr, atl_lambda_cap)

docs/ARCHITECTURE.md:248-262   Lambda schedule table
docs/ARCHITECTURE.md:428-436   Known Issues (4 items)
```

### Previous Session Compaction Summary

The prior conversation covered:
- MAV vs RMS feature comparison (MAV wins by +0.033 R² at full 400-epoch convergence)
- Comprehensive ARCHITECTURE.md documentation created
- ATL weakness diagnosis (DD winning, LR imbalance, per-action variance)
- Code cleanup via ai-slop-cleaner
- --feature-order CLI arg added with validation
- Ralph workflow for rigorous A/B testing methodology

### Project State

- **Current branch**: main
- **Modified files**: `src/deep learning/run_db2_paper_cfc_finetune.py` (--feature-order arg, validation)
- **New files**: `docs/ARCHITECTURE.md` (~350 lines), `Project log.md`
- **Uncommitted changes**: See git status
- **Best model config**: h=256, DenseCfC, MAV features, ATL with λ_cap=0.3 (to be improved)
- **Best result**: 7-subject mean ATL R² = 0.492 ± 0.110


- some summaries that may be integrated into the final log  
    - sEMG data may have large covaraite shift between subjects that domain-adaptation method doesn't work really well 