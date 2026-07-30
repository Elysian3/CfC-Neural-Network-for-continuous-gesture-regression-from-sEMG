<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-23 | Updated: 2026-05-23 -->

# deep learning

## Purpose

Model training and experiment orchestration. Contains the DenseCfC regressor
implementation, training loop, and paper-reproduction pipeline (pretrain +
optional GAN-style ATL fine-tune without GRL).

## Key Files

| File | Description |
|------|-------------|
| `train.py` | Import-only core module: `CfCTrainingConfig`, `DenseCfCLinearRegressor`, sequence splitting, training, evaluation, shared normalization wrappers, and `DomainDiscriminator` |
| `run_db2_paper_cfc_finetune.py` | Single experiment entry point: RMS-only by default, DenseCfC pretraining, head-only fine-tuning or alternating GAN-style ATL |

## First-Principles Architecture

### Why CfC (Closed-form Continuous-time) for EMG?

Standard RNNs (LSTM, GRU) have fixed time constants per neuron, governed by learned weights that are static after training. But EMG dynamics span multiple timescales:
- **Fast transients** (~10-50 ms): motor unit action potential bursts during movement onset
- **Slow dynamics** (~200-500 ms): force ramps, sustained holds, fatigue effects

A CfC neuron's time constant τ is a function of its input: τ = f(x). When the EMG changes rapidly, τ shrinks (fast integration). When the signal is stable, τ grows (slow integration, noise rejection). The "closed-form" means the ODE has an analytical solution — no numerical integration step that could diverge with stiff dynamics.

### Model Variant

| Variant | Architecture | Use Case |
|---------|-------------|----------|
| `DenseCfCLinearRegressor` (dense_cfc_linear) | Dense CfC encoder → Dropout → linear head | Current deployment uses 12-D RMS input; research APIs permit other validated feature orders. |

The dense variant separates the recurrent encoder from the linear readout — this enables `forward_with_features()` which returns the 256-dim final state for the Domain Discriminator during ATL. CfCRegressor (AutoNCP sparse wiring) was removed in July 2026.

### Data Splitting Strategies — Why Three?

| Strategy | Method | Leakage Risk | Best For |
|----------|--------|-------------|----------|
| `recording` | Whole files → train/val/test | None | Final evaluation (conservative) |
| `blocked_time` | Each file → contiguous time blocks 70/15/15 + 16-window gap | Low | Hyperparameter tuning (uses all data) |
| `action_stratified` | Per-action repetition segments assigned to splits | Low-Medium | Semantic DoA decoding (balanced action coverage) |

The 16-window gap in blocked splitting is critical: with 75% overlap (150ms out of 200ms), adjacent windows share 75% of their samples. Without a gap, a window in the training set could share 75% of its EMG samples with a window in the test set — the model would memorize rather than generalize.

### Why μ-law over Z-score for Feature Normalization?

EMG feature distributions are heavy-tailed: most windows have low activation (rest), but a few have extreme values (maximal voluntary contraction). Z-score normalization assumes approximate normality — μ-law's logarithmic compression handles the long tail without letting outliers dominate the normalized range. The default μ=255 matches telephony standards where μ-law was first characterized.

### Why MSELoss?

Standard MSE is used for regression. The loss is computed on μ-law-normalized targets, so large angle errors are already compressed by the log transform — MSE in normalized space is effectively robust to extreme values without needing Huber/SmoothL1.

## Training Protocol

```
run_db2_paper_cfc_finetune.py    ← Single entry point: DenseCfC pretrain + ATL fine-tune
    ├── Pretraining: 31 source subjects → 400 epochs → checkpoint
    ├── ATL fine-tune: resume checkpoint → alternating DD/target-network steps
    └── Evaluation: per-DoA R² + MAE + RMSE on held-out test repetitions
```

5 legacy scripts (run_db2_single_subject, run_db2_subject_adaptation, run_doa5_subject_adaptation, run_semantic_dof_screen, analyze_atl_per_action) removed July 2026. Utilities migrated to train.py.

## Working Principles (Applied to Model Training)

These are the same principles from the root `AGENTS.md`, applied specifically to training experiments:

1. **Test-Driven — Model Edition**: Every new model variant (`build_cfc_regressor`) must have a construction test verifying input_dim → output_dim → hidden_units shape contract. Every new loss function must have a test with known input/output computing the expected loss by hand. Every new split strategy must have a no-leakage test (no shared windows, no shared recordings across splits).

**Hardware-gated design (Principle 8):** Every architectural choice — `hidden_units`, feature set — cascades directly to the ESP32-S3 memory budget (~512 KB SRAM). Before training, the agent must estimate `total_params × dtype_bytes` and verify the FP32 footprint is under ~400 KB. INT8 quantization is a deployment requirement, not an optimization. A model that scores R²=0.99 but cannot fit on-chip is not a deliverable.

2. **Ask First, Never Guess — Hyperparameter Edition**: "Let me try lr=0.001 and see" is guessing. The correct flow: (a) check `Project log.md` for prior experiments with this model family, (b) check `CfCTrainingConfig` defaults which encode the current best-known values, (c) if deviating, state the reason: "Increasing learning rate from 1e-3 to 5e-3 because the loss curve shows no movement in the first 5 epochs, indicating the optimizer is stuck."

3. **Scientific Rigor — Evaluation Edition**: Never report just the mean R². Always report: per-target R² (all 5 DoAs or all 10 glove columns), per-action R² (which gestures does the model fail on?), train/val/test split sizes so the reader can judge statistical power. A single scalar is a press release, not a measurement.

4. **Dual Role**: If the user proposes comparing two models by looking at prediction plots, push back: plots show qualitative fit; metrics quantify it. Both are needed — the plot shows *where* the model fails, the metric shows *how much*. If a hyperparameter search is proposed without a hold-out validation set, refuse: optimizing on the test set is p-hacking, not science.

5. **Engineering Five Steps — Experiment Edition**:
   - **Question**: What hypothesis does this experiment test? If it can't be stated before running, it's not an experiment — it's a fishing expedition.
   - **Delete**: Turn off unnecessary features/layers before adding new ones. Ablation studies are more informative than kitchen-sink models.
   - **Simplify**: Test on one subject, one exercise, 5 epochs first. If it doesn't work there, it won't work on the full dataset.
   - **Accelerate**: `max_windows_per_file` exists for a reason — use it to cut iteration time.
   - **Automate**: Every experiment script must save a JSON summary with all config + metrics. Manual inspection of console output is not reproducible.

6. **Explain Every Action**: Every hyperparameter choice in a config override must have a comment explaining why it differs from the default. Every new experiment script must have a header comment stating the scientific question it investigates.

7. **Strict Intent Attribution**: Never misattribute an experimental design choice to the user. If the user asks "try a lower learning rate" and the agent runs a full grid search, the grid search was the agent's initiative — not "the user requested a grid search." Every protocol decision must be traceable to its actual source: user directive, reference paper, prior experiment log, or agent hypothesis. Attributing an agent's speculation to the user corrupts the experiment's provenance.

## For AI Agents

### Working In This Directory
- All scripts import from `train.py` — this is the single source of truth for model architecture, training loop, and evaluation.
- New experiment scripts should import from `train.py` (not copy-paste its functions).
- `sys.path` manipulation at the top of each script ensures `dataflow/` modules are importable regardless of working directory.
- All scripts use `matplotlib.use("Agg")` — plots are saved to disk, never displayed interactively.
- Results are written to `log/` at the repo root.

### Testing Requirements
- `tests/test_cfc_training.py` — tests for model construction, splitting, metrics, normalization
- Tests mock training with tiny synthetic data; they do not run full experiments.

### Common Patterns
- Configuration via frozen dataclass (`CfCTrainingConfig`) with `build_best_cfc_config(**overrides)` — prevents scattered magic numbers.
- Train/val/test split code is separate from model code — `build_blocked_sequence_splits`, `build_action_stratified_sequence_splits`.
- Normalization statistics are **always** fit on training split only (`fit_feature_normalizer`, `fit_target_normalizer`), then applied to val/test via `apply_*_normalizer`.
- `recording_id` is tracked through every `SequenceSplit` so evaluations can group metrics by source file.

## Dependencies

### Internal
- `src/dataflow/` — `feature_extraction` (pipeline, normalization), `doa_mapping` (DoA targets), `datapreprocess` (loading), `SwRectify` (windowing)
- `src/data/DB2/` — training data

### External
- `ncps.torch` — `CfC` layer (dense connections only; AutoNCP removed)
- `torch` + `torch.nn` — model training
- `numpy`, `matplotlib` — data handling and plotting

## Operational Guardrails (Training-Specific)

See root `AGENTS.md` for general guardrails. These are specific to experiment launches.

### Before Launching Any Experiment

1. **Read the notepad.** The Priority Context in `.omc/notepad.md` has the current best config. Never reconstruct hyperparameters from memory.
2. **Check `Project log.md`** for the most recent experiment context — what was tested last, what the current hypothesis is, what the next step should be.
3. **Smoke-test new code paths first.** A 2-epoch run on 2-3 subjects catches shape mismatches, import errors, and data pipeline bugs before wasting a full 400-epoch run.
4. **Use `--skip-fine-tune` for pretrain-only runs.** Don't accidentally trigger ATL when you only want pretraining.
5. **Check `--target-mapping` is consistent.** If resuming a `doa5` checkpoint, the current config must also use `doa5` (and vice versa for `glove_columns`). The code now validates this.
6. **Verify output dimensions.** In `glove_columns` mode, the model outputs 13 columns, and post-hoc DoA metrics are in `summary.json` under `doa_metrics`.

### Common Pitfalls

| # | Pitfall | Detection | Fix |
|---|---------|-----------|-----|
| 1 | Pretrain checkpoint h ≠ ATL h | `mat1/mat2 shape mismatch` in DomainDiscriminator | Match `--hidden-units` between runs; let checkpoint config override |
| 2 | Python 3.14 loads 3.13 checkpoint | `ModuleNotFoundError: No module named 'pathlib._local'` | Use Python 3.13 for all experiment runs |
| 3 | `glove_columns` mode prints wrong column count | Output shows 10 columns instead of 13 | Fixed: print now shows `target_mapping` + actual output dim |
| 4 | Forgotten `.to(device)` after `load_state_dict` | Model runs on CPU despite `--device cuda` | Always chain `.to(device)` after loading state dict |
| 5 | Normalizers fit on target data leak | Over-optimistic zero-shot R² | `fit_target_normalizer` must only see source train data |
