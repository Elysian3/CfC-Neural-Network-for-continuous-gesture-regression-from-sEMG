idf.py build &&
Set-Location -Path "D:\Project Antikythera\firmware"

# Project Antikythera

Project Antikythera decodes 12-channel forearm surface EMG into the ten MCP/PIP
CyberGlove targets used by the DB2 RoFormer paper baseline. The deployment path uses
a dense Closed-form Continuous-time (CfC) network and targets ESP32-S3 INT8
inference.

The J10 contract selects one-based glove channels `2, 3, 5, 6, 8, 9, 12, 13,
16, 17` (zero-based indices `1, 2, 4, 5, 7, 8, 11, 12, 15, 16`) in paper order.

## Target deployment configuration

```text
12-channel sEMG at 2000 Hz
  -> DC removal
  -> offline training: zero-phase 50 Hz notch and 20-450 Hz bandpass
  -> 200 ms windows, 50 ms stride
  -> RMS only: 12 values per window
  -> 8-window sequence: 550 ms signal coverage
  -> mu-law normalization (mu=255; training statistics only)
  -> DenseCfCLinearRegressor(input=12, hidden=256, output=10)
  -> inverse target mu-law normalization
  -> ten paper-selected CyberGlove targets in the original sensor scale
```

The 550 ms coverage is `200 ms + 7 * 50 ms`; it is not 400 ms. The default
10-output RMS-only model has 169,098 parameters. Weight matrices remain INT8
in flash; biases, quantization scales, recurrent buffers, and normalization
statistics retain their declared floating-point representations.

The headers currently checked into `firmware/main` are a legacy five-output
compile/parity fixture and declare `TARGET_NORMALIZATION_AVAILABLE=0`. They do
not constitute a J10 deployment artifact. Exporting a newly trained
10-output checkpoint replaces both headers, enables target inverse
normalization, and makes the firmware output original-scale glove targets.
Production compilation fails closed while the legacy normalization fixture is
present; only the PC golden tests explicitly opt into that fixture.

The ESP32 path uses stateful causal biquads because a real-time controller
cannot use future samples. The current PC golden tests cover RMS, μ-law, and
CfC inference, but not filter parity; end-to-end filter train/serve skew remains
an explicit deployment validation gap.

The acquisition code in `firmware/main/main.c` is still a hardware scaffold,
not a verified 12-channel frontend: one 74HC4051 exposes only eight selectable
inputs and the current ADC task is explicitly one-shot. This work verifies the
checkpoint/export/CfC inference contract; a physical deployment still requires
a real 12-channel acquisition design (or retraining for the verified frontend)
and causal preprocessing parity.

Adversarial Transfer Learning (ATL) is optional. When enabled, the source model
is frozen, a domain discriminator and target model are trained in alternating
GAN-style optimizer steps, and the target model minimizes an explicit mapping
loss plus its regression loss. There is no Gradient Reversal Layer (GRL).

## Research API versus deployment defaults

The paper-protocol CLI defaults to RMS-only input. The reusable dataflow and
training APIs still support `mav`, `mavs`, `wl`, `zc`, `ssc`, and `rms` for
controlled ablation studies. Feature extraction computes only requested
features; rest-state threshold calibration runs only when ZC or SSC requires
it. Alternative feature sets are research configurations until their INT8
memory and PC golden tests pass.

### Default migration

The supported paper CLI now defaults to RMS-only input and μ-law `mu=255`.
Earlier generic defaults used five features and `mu=2^20`; invoking the CLI
without explicit flags therefore changes the experiment protocol. This
migration is intentional, test-locked, and should be recorded with any result
compared against an older checkpoint.

## Code map

| Path | Responsibility |
|---|---|
| `src/dataflow/datapreprocess.py` | Load DB2 recordings and apply zero-phase filters |
| `src/dataflow/SwRectify.py` | Window signals and align continuous targets |
| `src/dataflow/feature_extraction.py` | On-demand EMG features and shared normalization math |
| `src/dataflow/doa_mapping.py` | Paper J10 selection plus legacy five-DoA mapping |
| `src/deep learning/train.py` | Import-only model, split, train, normalize, and evaluation library |
| `src/deep learning/run_db2_paper_cfc_finetune.py` | Supported pretrain/fine-tune experiment CLI |
| `src/hardwareOperation/hardware_preflight.py` | Checkpoint metadata, INT8 error, SRAM estimate, and golden tensors |
| `src/hardwareOperation/export_weights.py` | Export weights and normalization headers to `firmware/main` |
| `src/hardwareOperation/test_cfc_pc.c` | PC golden test using canonical firmware inference sources |
| `src/hardwareOperation/test_features_pc.c` | PC RMS-to-CfC integration test using canonical firmware sources |
| `firmware/main/` | Canonical portable C inference and ESP-IDF application |

## Setup

Use Python 3.13 for checkpoint-based experiments. The project depends on
PyTorch, `ncps`, NumPy, SciPy, and Matplotlib:

```powershell
python -m pip install -r requirements.txt
```

Place NinaPro DB2 files at
`src/data/DB2/DB2_s{id}/S{id}_E{exercise}_A1.mat`.

## Run

From the repository root:

```powershell
python "src/deep learning/run_db2_paper_cfc_finetune.py" `
  --target-subject S1 `
  --target-mapping joint_angles10 `
  --target-offset-samples 0 `
  --hidden-units 256 `
  --batch-size 1024 `
  --gpu-resident `
  --stateful `
  --enable-atl
```

`--stateful` applies causal TBPTT to pretraining. Pretraining and ATL run for
their requested fixed epoch counts and return the final epoch; neither uses an
internal validation split or checkpoint selection. ATL uses every S1 support
label, while S1 query labels are reserved for the final zero-shot and adapted
evaluations (query EMG may still advance the full-stream state as unlabeled
context). Before ATL, all unscored target labels are physically replaced with
NaN sentinels; their EMG frames and provenance remain as causal context. ATL is
always causal/stateful when enabled. The default target
keeps the ten paper-selected DB2 MCP/PIP channels in their declared order.
`--target-offset-samples` is required: use `0` for synchronized angle estimation
or `200` for 100 ms-ahead prediction at DB2's 2 kHz sampling rate. In
stateful mode, `--batch-size 1024` is a maximum
stream-lane count, so the actual batch cannot exceed the number of independent
recording streams. `--train-repetitions-per-action` remains accepted for CLI
compatibility, but it now means target-subject support repetitions per action;
all selected source repetitions are supervised. The CLI writes resolved configuration, split sizes,
per-target metrics, and checkpoints under `log/`.

For AutoNCP training, add `--model-family autoncp_cfc_linear` to the command
above and use a separate `--output-dir`. Dense CfC remains the default.
`--hidden-units` is the total inter/command/motor state size and must exceed
the target count by at least three. AutoNCP uses fixed `sparsity_level=0.5`
and wiring seed `22222`; its masks are saved in the checkpoint. Its motor
output has one feature per target, followed by dropout and a linear
regression head (`Linear(10, 10)` for J10). `--cfc-dropout` applies only to
this readout in AutoNCP mode.

Stateful training carries the full hidden state. ATL aligns motor features
(ten for J10); head-only fine-tuning updates only the linear readout.
Chain norm diagnostics identify whether they describe the full dense state
or the AutoNCP motor output. Resume with the same `--model-family`; dense
checkpoints cannot initialize AutoNCP. This is a research training option:
the existing ESP32 export supports only the dense model.

## Verify

The core Python regression suite uses synthetic data and requires no DB2 files:

```powershell
python -m pytest tests/
```

The PC tests compile against `firmware/main`, not copied C sources:

```powershell
gcc -O2 -std=c11 src/hardwareOperation/test_cfc_pc.c firmware/main/cfc_inference.c -lm -o test_cfc_pc
gcc -O2 -std=c11 src/hardwareOperation/test_features_pc.c firmware/main/features.c firmware/main/cfc_inference.c -lm -o test_features_pc
```

Run both generated executables and require a zero exit code. Keep generated
`.o` and `.exe` files out of version control.

Header export fails closed unless it receives feature-normalization statistics
and the checkpoint or stats artifact supplies target-normalization parameters.
The latter are required so firmware returns original-scale joint angles:

```powershell
python src/hardwareOperation/export_weights.py `
  --checkpoint log/.../model.pt `
  --norm-stats log/.../feature_normalization.npz
```

For the detailed signal, model, adaptation, and deployment design, see
`docs/architecture.md`.
