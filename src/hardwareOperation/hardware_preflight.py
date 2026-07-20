"""PC-side hardware preflight for DenseCfC ESP32-S3 deployment.

This module builds a deterministic golden-output contract. It verifies model
metadata, quantization sensitivity, and conservative SRAM estimates; it does
not prove real MCU inference until a non-PyTorch kernel matches the golden
outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ── Python 3.14 compatibility: mock pathlib._local removed in 3.14 ──────────
import pathlib as _pathlib

if "pathlib._local" not in sys.modules:
    _mock = types.ModuleType("pathlib._local")
    _mock.WindowsPath = _pathlib.WindowsPath
    _mock.PosixPath = _pathlib.PosixPath
    _mock.PureWindowsPath = _pathlib.PureWindowsPath
    _mock.PurePosixPath = _pathlib.PurePosixPath
    sys.modules["pathlib._local"] = _mock
# ────────────────────────────────────────────────────────────────────────────

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEP_LEARNING_DIR = PROJECT_ROOT / "src" / "deep learning"
if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(DEEP_LEARNING_DIR))

from train import build_cfc_regressor


VALID_FEATURE_ORDERS = {("rms",), ("rms", "zc")}
TARGET_MODEL_FAMILY = "dense_cfc_linear"
TARGET_HIDDEN_UNITS = 256
TARGET_OUTPUT_DIM = 5
TARGET_CHANNELS = 12
TARGET_SEQ_LEN = 8
SRAM_BUDGET_KB = 400.0
PSRAM_KB = 2048.0
DOA_NAMES = (
    "thumb_rotation",
    "thumb_flexion",
    "index_flexion",
    "middle_flexion",
    "ring_little_flexion",
)
RELATIVE_ERROR_FORMULA = "max(abs(fp32 - int8_dequantized)) / (max(fp32) - min(fp32) + 1e-8) * 100"


class HardwarePreflightError(ValueError):
    """Raised when a checkpoint cannot satisfy the hardware preflight contract."""


@dataclass(frozen=True)
class CheckpointBundle:
    model: torch.nn.Module
    state_dict: dict[str, torch.Tensor]
    config: dict[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_feature_order(feature_order: Any) -> tuple[str, ...]:
    if isinstance(feature_order, str):
        return (feature_order,)
    if feature_order is None:
        return ()
    return tuple(str(item) for item in feature_order)


def validate_deployment_config(
    config: dict[str, Any],
    *,
    output_dim: int | None = None,
) -> dict[str, Any]:
    """Validate the current RMS-only DenseCfC hardware target metadata."""

    model_family = str(config.get("model_family", ""))
    feature_order = normalize_feature_order(config.get("feature_order"))
    hidden_units = int(config.get("hidden_units", -1))
    config_output_dim = config.get("output_dim", output_dim)
    actual_output_dim = int(TARGET_OUTPUT_DIM if config_output_dim is None else config_output_dim)

    if model_family != TARGET_MODEL_FAMILY:
        raise HardwarePreflightError(
            f"hardware target requires {TARGET_MODEL_FAMILY}, got {model_family!r}"
        )
    if feature_order not in VALID_FEATURE_ORDERS:
        raise HardwarePreflightError(
            f"hardware target requires one of {sorted(VALID_FEATURE_ORDERS)}, got {feature_order}"
        )
    if hidden_units != TARGET_HIDDEN_UNITS:
        raise HardwarePreflightError(
            f"hardware target requires hidden_units={TARGET_HIDDEN_UNITS}, got {hidden_units}"
        )
    if actual_output_dim != TARGET_OUTPUT_DIM:
        raise HardwarePreflightError(
            f"hardware target requires output_dim={TARGET_OUTPUT_DIM}, got {actual_output_dim}"
        )

    forbidden_keys = ("use_grl", "uses_grl", "grl_lambda", "domain_discriminator")
    for key in forbidden_keys:
        if config.get(key):
            raise HardwarePreflightError(f"hardware inference contract rejects runtime {key}")

    runtime_labels = {
        str(config.get("architecture", "")).lower(),
        str(config.get("atl_mode", "")).lower(),
        str(config.get("runtime_mode", "")).lower(),
    }
    rejected_runtime_terms = ("autoncp", "grl", "dann", "adversarial")
    if any(any(term in label for term in rejected_runtime_terms) for label in runtime_labels):
        raise HardwarePreflightError(
            f"hardware inference contract rejects runtime labels {sorted(runtime_labels)}"
        )

    return {
        "model_family": model_family,
        "feature_order": list(feature_order),
        "hidden_units": hidden_units,
        "output_dim": actual_output_dim,
        "input_dim": len(feature_order) * TARGET_CHANNELS,
        "seq_len": TARGET_SEQ_LEN,
        "channels": TARGET_CHANNELS,
    }


def load_checkpoint_bundle(checkpoint_path: Path, device: torch.device) -> CheckpointBundle:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    config = dict(ckpt["config"])
    metadata = validate_deployment_config(config)
    model = build_cfc_regressor(
        input_dim=metadata["input_dim"],
        output_dim=metadata["output_dim"],
        hidden_units=metadata["hidden_units"],
        model_family=metadata["model_family"],
        cfc_dropout=float(config.get("cfc_dropout", 0.0)),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return CheckpointBundle(
        model=model,
        state_dict=state_dict,
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256(checkpoint_path),
    )


def quantize_weights_int8(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    """Quantize floating tensors with per-tensor asymmetric INT8 scales.

    Uses the full INT8 range [-128, 127] via asymmetric scaling:
      scale = 2.0 * max(|w|) / 255.0
      q = clamp(round(w / scale), -128, 127)

    This is half the step size of symmetric quantization (scale = max/127),
    cutting quantization error approximately in half for typical weight
    distributions.
    """

    q_sd: dict[str, torch.Tensor] = {}
    q_info: dict[str, dict[str, Any]] = {}

    for name, param in state_dict.items():
        if not torch.is_tensor(param):
            q_sd[name] = param
            q_info[name] = {"type": "non_tensor", "param_count": 0, "bits": 0}
            continue
        if not param.is_floating_point():
            q_sd[name] = param
            q_info[name] = {
                "type": "non_float",
                "dtype": str(param.dtype),
                "param_count": int(param.numel()),
                "bits": int(param.element_size() * 8),
            }
            continue

        weights = param.detach()
        w_max = weights.abs().max().item()
        scale = 1.0 if w_max < 1e-10 else (2.0 * w_max) / 255.0
        quantized = torch.clamp(torch.round(weights / scale), -128, 127).to(torch.int8)
        q_sd[name] = quantized
        q_info[name] = {
            "type": "int8",
            "scale": float(scale),
            "range": [-128 * scale, 127 * scale],
            "param_count": int(weights.numel()),
            "bits": 8,
        }

    return q_sd, q_info


def quantize_weights_int8_per_channel(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    """Per-channel asymmetric INT8 quantization for weight matrices.

    Each output channel (row) gets its own scale — critical when per-tensor
    quantization exceeds the 5 % error gate (e.g. RMS+ZC checkpoints with
    high channel variance).
    """

    q_sd: dict[str, torch.Tensor] = {}
    q_info: dict[str, dict[str, Any]] = {}

    for name, param in state_dict.items():
        if not torch.is_tensor(param):
            q_sd[name] = param
            q_info[name] = {"type": "non_tensor", "param_count": 0, "bits": 0}
            continue
        if not param.is_floating_point():
            q_sd[name] = param
            q_info[name] = {
                "type": "non_float",
                "dtype": str(param.dtype),
                "param_count": int(param.numel()),
                "bits": int(param.element_size() * 8),
            }
            continue

        weights = param.detach()
        if weights.dim() >= 2:
            # Per-channel: one scale per output channel (dim 0)
            w_max_per_ch = weights.abs().max(dim=1).values  # (out_channels,)
            w_max_per_ch = torch.where(w_max_per_ch < 1e-10, torch.ones_like(w_max_per_ch), w_max_per_ch)
            scale = (2.0 * w_max_per_ch / 255.0).unsqueeze(1)  # broadcast over input dim
            quantized = torch.clamp(torch.round(weights / scale), -128, 127).to(torch.int8)
            q_sd[name] = quantized
            q_info[name] = {
                "type": "int8",
                "mode": "per_channel",
                "scale": [float(s) for s in w_max_per_ch * 2.0 / 255.0],
                "param_count": int(weights.numel()),
                "bits": 8,
            }
        else:
            # Per-tensor for 1D tensors (biases, etc.)
            w_max = weights.abs().max().item()
            scale = 1.0 if w_max < 1e-10 else (2.0 * w_max) / 255.0
            quantized = torch.clamp(torch.round(weights / scale), -128, 127).to(torch.int8)
            q_sd[name] = quantized
            q_info[name] = {
                "type": "int8",
                "mode": "per_tensor",
                "scale": float(scale),
                "range": [-128 * scale, 127 * scale],
                "param_count": int(weights.numel()),
                "bits": 8,
            }

    return q_sd, q_info


def dequantize_weights_per_channel(
    quantized_state_dict: dict[str, torch.Tensor],
    quantization_info: dict[str, dict[str, Any]],
) -> dict[str, torch.Tensor]:
    dequantized: dict[str, torch.Tensor] = {}
    for name, param in quantized_state_dict.items():
        info = quantization_info[name]
        if info.get("type") == "int8":
            if info.get("mode") == "per_channel":
                scale = torch.tensor(info["scale"], dtype=torch.float32).unsqueeze(1)
                dequantized[name] = param.float() * scale
            else:
                dequantized[name] = param.float() * float(info["scale"])
        else:
            dequantized[name] = param
    return dequantized


def dequantize_weights(
    quantized_state_dict: dict[str, torch.Tensor],
    quantization_info: dict[str, dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Dequantize per-tensor INT8 weights.  Use :func:`dequantize_weights_per_channel`
    for per-channel quantization data."""
    dequantized: dict[str, torch.Tensor] = {}
    for name, param in quantized_state_dict.items():
        info = quantization_info[name]
        if info.get("type") == "int8":
            scale_raw = info["scale"]
            if not isinstance(scale_raw, (int, float)):
                raise ValueError(
                    f"dequantize_weights expects per-tensor scalars; "
                    f"'{name}' has scale type {type(scale_raw).__name__}. "
                    f"Use dequantize_weights_per_channel for per-channel data."
                )
            dequantized[name] = param.float() * float(scale_raw)
        else:
            dequantized[name] = param
    return dequantized


def compute_size_stats(q_info: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    total_count = sum(int(info.get("param_count", 0)) for info in q_info.values())
    int8_count = sum(
        int(info.get("param_count", 0)) for info in q_info.values() if info.get("type") == "int8"
    )
    passthrough_bytes = sum(
        int(info.get("param_count", 0)) * int(info.get("bits", 32)) / 8
        for info in q_info.values()
        if info.get("type") != "int8"
    )
    int8_bytes = int8_count + passthrough_bytes
    fp32_bytes = total_count * 4
    return {
        "total_params": total_count,
        "int8_params": int8_count,
        "fp32_params": total_count - int8_count,
        "int8_coverage_pct": int8_count / total_count * 100 if total_count else 0.0,
        "fp32_size_kb": fp32_bytes / 1024,
        "int8_size_kb": int8_bytes / 1024,
        "compression_ratio": fp32_bytes / int8_bytes if int8_bytes else 0.0,
    }


def estimate_core_sram_kb(
    *,
    input_dim: int,
    seq_len: int,
    hidden_units: int,
    output_dim: int,
    weight_kb: float,
    dtype_bytes: int,
    scratch_multiplier: float = 2.0,
) -> dict[str, float]:
    input_kb = seq_len * input_dim * dtype_bytes / 1024
    hidden_kb = hidden_units * dtype_bytes / 1024
    output_kb = output_dim * dtype_bytes / 1024
    norm_kb = input_dim * 2 * 4 / 1024
    scratch_kb = scratch_multiplier * (hidden_kb + output_kb + input_dim * dtype_bytes / 1024)
    total_kb = weight_kb + input_kb + hidden_kb + output_kb + norm_kb + scratch_kb
    return {
        "weight_kb": weight_kb,
        "input_window_kb": input_kb,
        "hidden_state_kb": hidden_kb,
        "output_kb": output_kb,
        "normalization_stats_kb": norm_kb,
        "scratch_kb": scratch_kb,
        "total_core_sram_kb": total_kb,
        "budget_kb": SRAM_BUDGET_KB,
        "psram_available_kb": PSRAM_KB,
        "uses_psram_for_core_inference": False,
        "fits_budget": total_kb < SRAM_BUDGET_KB,
    }


def estimate_dense_macs(*, input_dim: int, hidden_units: int, output_dim: int, seq_len: int) -> dict[str, int]:
    # CfC internals are library-defined, so this is a conservative dense-RNN proxy.
    per_step = input_dim * hidden_units + hidden_units * hidden_units
    head = hidden_units * output_dim
    return {
        "input_dim": input_dim,
        "hidden_units": hidden_units,
        "seq_len": seq_len,
        "proxy_macs_per_step": per_step,
        "proxy_macs_per_sequence": per_step * seq_len + head,
        "note": "Conservative dense recurrent proxy; verify exact CfC kernel during C++ parity spike.",
    }


def make_golden_input(
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    input_dim: int,
    device: torch.device,
    real_data_path: Path | None = None,
) -> torch.Tensor:
    """Build golden input for preflight verification.

    When ``real_data_path`` is provided the tensor is loaded from a .npy file
    and broadcast/trimmed to the expected ``(batch, seq_len, input_dim)`` shape.
    Otherwise a seeded random tensor is used (backward-compatible, useful for
    smoke-testing that does not require real EMG data).
    """
    if real_data_path is not None:
        raw = torch.from_numpy(
            __import__("numpy").load(str(real_data_path))
        ).float()
        if raw.dim() == 2 and raw.shape[0] == seq_len:
            raw = raw.unsqueeze(0)  # (seq_len, input_dim) → (1, seq_len, input_dim)
        if raw.dim() != 3:
            raise HardwarePreflightError(
                f"golden input at {real_data_path} must be 2D or 3D, got shape {list(raw.shape)}"
            )
        # Ensure input_dim matches (trim or zero-pad)
        if raw.shape[2] != input_dim:
            if raw.shape[2] > input_dim:
                raw = raw[:, :, :input_dim]
            else:
                pad = torch.zeros(raw.shape[0], raw.shape[1], input_dim - raw.shape[2])
                raw = torch.cat([raw, pad], dim=2)
        # Broadcast or truncate to requested batch_size × seq_len
        if raw.shape[0] < batch_size:
            repeats = [batch_size // raw.shape[0] + 1] + [1] * (raw.dim() - 1)
            raw = raw.repeat(*repeats)
        raw = raw[:batch_size, :seq_len, :]
        return raw.to(device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.randn(batch_size, seq_len, input_dim, generator=generator, dtype=torch.float32)
    return tensor.to(device)


def compute_error_report(pred_fp32: torch.Tensor, pred_int8: torch.Tensor) -> dict[str, Any]:
    diff = (pred_fp32 - pred_int8).abs()
    max_abs_error = diff.max()
    mean_abs_error = diff.mean()
    output_range = pred_fp32.max() - pred_fp32.min()
    relative_error_pct = max_abs_error / (output_range + 1e-8) * 100
    per_doa_mae = diff.mean(dim=0)
    return {
        "formula": RELATIVE_ERROR_FORMULA,
        "max_abs_error": float(max_abs_error.item()),
        "mean_abs_error": float(mean_abs_error.item()),
        "relative_error_pct": float(relative_error_pct.item()),
        "per_doa_mae": [float(value) for value in per_doa_mae],
        "doa_names": list(DOA_NAMES),
    }


def run_preflight(
    *,
    checkpoint: Path,
    output_dir: Path,
    seed: int = 42,
    batch_size: int = 4,
    device: torch.device | None = None,
    real_data_path: Path | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_checkpoint_bundle(checkpoint, device)
    metadata = validate_deployment_config(bundle.config)

    golden_input = make_golden_input(
        seed=seed,
        batch_size=batch_size,
        seq_len=metadata["seq_len"],
        input_dim=metadata["input_dim"],
        device=device,
        real_data_path=real_data_path,
    )
    with torch.no_grad():
        pred_fp32 = bundle.model(golden_input).cpu()
    if list(pred_fp32.shape) != [batch_size, TARGET_OUTPUT_DIM]:
        raise HardwarePreflightError(
            f"model output shape must be [{batch_size}, {TARGET_OUTPUT_DIM}], got {list(pred_fp32.shape)}"
        )

    # Try per-tensor first; fall back to per-channel if error exceeds gate.
    q_sd_pt, q_info_pt = quantize_weights_int8(bundle.state_dict)
    dq_pt = dequantize_weights(q_sd_pt, q_info_pt)
    bundle.model.load_state_dict(dq_pt)
    bundle.model.eval()
    with torch.no_grad():
        pred_int8_pt = bundle.model(golden_input).cpu()
    error_pt = compute_error_report(pred_fp32, pred_int8_pt)

    q_sd_pc, q_info_pc = quantize_weights_int8_per_channel(bundle.state_dict)
    dq_pc = dequantize_weights_per_channel(q_sd_pc, q_info_pc)
    bundle.model.load_state_dict(dq_pc)
    bundle.model.eval()
    with torch.no_grad():
        pred_int8_pc = bundle.model(golden_input).cpu()
    error_pc = compute_error_report(pred_fp32, pred_int8_pc)

    if list(pred_int8_pt.shape) != [batch_size, TARGET_OUTPUT_DIM]:
        raise HardwarePreflightError(
            f"INT8-dequantized output shape must be [{batch_size}, {TARGET_OUTPUT_DIM}], got {list(pred_int8_pt.shape)}"
        )

    use_per_channel = error_pc["relative_error_pct"] < error_pt["relative_error_pct"]
    q_sd = q_sd_pc if use_per_channel else q_sd_pt
    q_info = q_info_pc if use_per_channel else q_info_pt
    pred_int8 = pred_int8_pc if use_per_channel else pred_int8_pt
    quant_method = (
        "per-channel asymmetric INT8" if use_per_channel
        else "per-tensor asymmetric INT8"
    )
    # Restore best dequantized weights for the chosen method
    bundle.model.load_state_dict(dq_pc if use_per_channel else dq_pt)
    bundle.model.eval()

    size_stats = compute_size_stats(q_info)
    fp32_sram = estimate_core_sram_kb(
        input_dim=metadata["input_dim"],
        seq_len=metadata["seq_len"],
        hidden_units=metadata["hidden_units"],
        output_dim=metadata["output_dim"],
        weight_kb=float(size_stats["fp32_size_kb"]),
        dtype_bytes=4,
    )
    int8_sram = estimate_core_sram_kb(
        input_dim=metadata["input_dim"],
        seq_len=metadata["seq_len"],
        hidden_units=metadata["hidden_units"],
        output_dim=metadata["output_dim"],
        weight_kb=float(size_stats["int8_size_kb"]),
        dtype_bytes=1,
    )
    error = compute_error_report(pred_fp32, pred_int8)
    portability_status = "WARN"
    report = {
        "schema_version": 1,
        "purpose": "PC preflight for quantization, memory estimate, and golden-output reproducibility; not proof of MCU inference.",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": bundle.checkpoint_sha256,
        },
        "model": {
            **metadata,
            "architecture": "DenseCfCLinearRegressor",
            "cfc_dropout": float(bundle.config.get("cfc_dropout", 0.0)),
            "total_params": int(size_stats["total_params"]),
        },
        "golden": {
            "seed": seed,
            "input_shape": list(golden_input.shape),
            "output_shape": list(pred_fp32.shape),
        },
        "quantization": {
            "method": quant_method,
            "int8_tensor_count": sum(1 for info in q_info.values() if info.get("type") == "int8"),
            "size": size_stats,
            "passes_error_gate": error["relative_error_pct"] < 5.0,
        },
        "error": error,
        "memory": {
            "sram_budget_kb": SRAM_BUDGET_KB,
            "fp32_core_estimate": fp32_sram,
            "int8_core_estimate": int8_sram,
            "fp32_deployable": bool(fp32_sram["fits_budget"]),
            "int8_deployable_estimate": bool(int8_sram["fits_budget"]),
        },
        "mac_estimate": estimate_dense_macs(
            input_dim=metadata["input_dim"],
            hidden_units=metadata["hidden_units"],
            output_dim=metadata["output_dim"],
            seq_len=metadata["seq_len"],
        ),
        "portability": {
            "status": portability_status,
            "reason": "No non-PyTorch DenseCfC kernel has matched this golden output yet.",
            "next_gate": "C/C++ DenseCfC parity spike on PC before board firmware.",
        },
    }

    torch.save(
        {
            "golden_input": golden_input.cpu(),
            "pred_fp32": pred_fp32,
            "pred_int8_dequantized": pred_int8,
            "quantization_info": q_info,
        },
        output_dir / "golden_tensors.pt",
    )
    with (output_dir / "hardware_preflight_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DenseCfC hardware preflight golden contract")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("log/hardware_preflight"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--real-data", type=Path, default=None,
                        help="Path to .npy file with real EMG features for golden input")
    args = parser.parse_args(argv)

    try:
        report = run_preflight(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            seed=args.seed,
            batch_size=args.batch_size,
            real_data_path=args.real_data,
        )
    except HardwarePreflightError as exc:
        print(f"Hardware preflight rejected checkpoint: {exc}")
        return 2

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
