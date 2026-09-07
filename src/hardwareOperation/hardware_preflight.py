"""Deterministic PC preflight for the deployable DenseCfC firmware contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEP_LEARNING_DIR = PROJECT_ROOT / "src" / "deep learning"
if str(DEEP_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(DEEP_LEARNING_DIR))

from train import build_cfc_regressor

TARGET_MODEL_FAMILY = "dense_cfc_linear"
TARGET_FEATURE_ORDER = ("rms",)
TARGET_HIDDEN_UNITS = 256
TARGET_BACKBONE_UNITS = 128
SUPPORTED_OUTPUT_DIMS = frozenset({5, 10, 13})
LEGACY_OUTPUT_DIM = 5
TARGET_CHANNELS = 12
TARGET_SEQ_LEN = 8
SRAM_BUDGET_KB = 400.0
PSRAM_KB = 2048.0
MAX_RELATIVE_ERROR_PCT = 5.0
RELATIVE_ERROR_FORMULA = (
    "max(abs(fp32 - int8_dequantized)) / "
    "(max(fp32) - min(fp32) + 1e-8) * 100"
)


class HardwarePreflightError(ValueError):
    """Raised when a checkpoint violates the firmware deployment contract."""


@dataclass(frozen=True)
class CheckpointBundle:
    model: torch.nn.Module
    state_dict: dict[str, torch.Tensor]
    config: dict[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    target_normalization_stats: dict[str, Any] | None = None
    target_contract: dict[str, Any] | None = None


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_feature_order(feature_order: Any) -> tuple[str, ...]:
    if isinstance(feature_order, str):
        return (feature_order.lower(),)
    if feature_order is None:
        return ()
    return tuple(str(item).lower() for item in feature_order)


def _state_dict_output_dim(state_dict: dict[str, torch.Tensor]) -> int | None:
    head_weight = state_dict.get("head.weight")
    if torch.is_tensor(head_weight) and head_weight.ndim == 2:
        return int(head_weight.shape[0])
    return None


def _validate_optional_checkpoint_metadata(config: dict[str, Any]) -> None:
    """Reject supplied input/sequence metadata that disagrees with firmware."""
    for key, expected in (("input_dim", TARGET_CHANNELS), ("seq_len", TARGET_SEQ_LEN)):
        if key in config:
            try:
                actual = int(config[key])
            except (TypeError, ValueError) as error:
                raise HardwarePreflightError(
                    f"checkpoint {key} must be an integer compatible with firmware"
                ) from error
            if actual != expected:
                raise HardwarePreflightError(
                    f"checkpoint {key}={config[key]!r} is incompatible with "
                    f"firmware {key}={expected}"
                )
    if "emg_channels" in config:
        channels = config["emg_channels"]
        try:
            channel_count = len(channels)
        except TypeError as error:
            raise HardwarePreflightError(
                "checkpoint emg_channels must be a sequence of firmware inputs"
            ) from error
        if isinstance(channels, str) or channel_count != TARGET_CHANNELS:
            raise HardwarePreflightError(
                "checkpoint emg_channels must describe exactly "
                f"{TARGET_CHANNELS} firmware inputs"
            )


def validate_target_contract(
    target_contract: dict[str, Any] | None,
    *,
    config: dict[str, Any],
    output_dim: int,
) -> None:
    """Reject checkpoint target metadata that cannot describe the model head."""
    if target_contract is None:
        return
    if not isinstance(target_contract, dict):
        raise HardwarePreflightError("checkpoint target_contract must be a dictionary")
    if int(target_contract.get("output_dim", -1)) != output_dim:
        raise HardwarePreflightError(
            "checkpoint target_contract output_dim disagrees with head.weight"
        )
    target_names = target_contract.get("target_names")
    if target_names is not None and len(target_names) != output_dim:
        raise HardwarePreflightError(
            "checkpoint target_contract target_names width is inconsistent"
        )
    for config_key, contract_key in (
        ("target_mapping", "mapping_name"),
        ("target_mapping_version", "mapping_version"),
    ):
        if (
            config_key in config
            and config[config_key] != target_contract.get(contract_key)
        ):
            raise HardwarePreflightError(
                f"checkpoint {config_key} disagrees with target_contract {contract_key}"
            )


def validate_firmware_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    output_dim: int,
) -> None:
    """Validate the fixed tensor shapes emitted by the DenseCfC firmware exporter."""
    layer_shapes = (
        (
            "cfc.rnn_cell.backbone.0",
            (TARGET_BACKBONE_UNITS, TARGET_CHANNELS + TARGET_HIDDEN_UNITS),
        ),
        ("cfc.rnn_cell.ff1", (TARGET_HIDDEN_UNITS, TARGET_BACKBONE_UNITS)),
        ("cfc.rnn_cell.ff2", (TARGET_HIDDEN_UNITS, TARGET_BACKBONE_UNITS)),
        ("cfc.rnn_cell.time_a", (TARGET_HIDDEN_UNITS, TARGET_BACKBONE_UNITS)),
        ("cfc.rnn_cell.time_b", (TARGET_HIDDEN_UNITS, TARGET_BACKBONE_UNITS)),
        ("head", (output_dim, TARGET_HIDDEN_UNITS)),
    )
    for layer, expected_weight_shape in layer_shapes:
        weight = state_dict.get(f"{layer}.weight")
        bias = state_dict.get(f"{layer}.bias")
        if not torch.is_tensor(weight) or tuple(weight.shape) != expected_weight_shape:
            actual = list(weight.shape) if torch.is_tensor(weight) else None
            raise HardwarePreflightError(
                f"{layer}.weight must have shape {expected_weight_shape}, got {actual}"
            )
        expected_bias_shape = (expected_weight_shape[0],)
        if not torch.is_tensor(bias) or tuple(bias.shape) != expected_bias_shape:
            actual = list(bias.shape) if torch.is_tensor(bias) else None
            raise HardwarePreflightError(
                f"{layer}.bias must have shape {expected_bias_shape}, got {actual}"
            )


def validate_deployment_config(
    config: dict[str, Any],
    *,
    output_dim: int | None = None,
) -> dict[str, Any]:
    """Validate metadata shared by the 5-, 10-, and 13-output firmware paths."""
    _validate_optional_checkpoint_metadata(config)
    model_family = str(config.get("model_family", ""))
    feature_order = normalize_feature_order(config.get("feature_order"))
    hidden_units = int(config.get("hidden_units", -1))
    configured_output = config.get("output_dim")
    if (
        configured_output is not None
        and output_dim is not None
        and int(configured_output) != int(output_dim)
    ):
        raise HardwarePreflightError(
            "checkpoint config output_dim disagrees with head.weight"
        )
    actual_output_dim = int(
        output_dim
        if output_dim is not None
        else configured_output
        if configured_output is not None
        else LEGACY_OUTPUT_DIM
    )

    if model_family != TARGET_MODEL_FAMILY:
        raise HardwarePreflightError(
            f"hardware target requires {TARGET_MODEL_FAMILY}, got {model_family!r}"
        )
    if feature_order != TARGET_FEATURE_ORDER:
        raise HardwarePreflightError(
            f"hardware target requires feature_order={TARGET_FEATURE_ORDER}, "
            f"got {feature_order}"
        )
    if hidden_units != TARGET_HIDDEN_UNITS:
        raise HardwarePreflightError(
            f"hardware target requires hidden_units={TARGET_HIDDEN_UNITS}, "
            f"got {hidden_units}"
        )
    if actual_output_dim not in SUPPORTED_OUTPUT_DIMS:
        raise HardwarePreflightError(
            "hardware target supports output_dim in "
            f"{sorted(SUPPORTED_OUTPUT_DIMS)}, got {actual_output_dim}"
        )

    for key in ("use_grl", "uses_grl", "grl_lambda", "domain_discriminator"):
        if config.get(key):
            raise HardwarePreflightError(f"hardware inference rejects runtime {key}")
    runtime_labels = {
        str(config.get("architecture", "")).lower(),
        str(config.get("atl_mode", "")).lower(),
        str(config.get("runtime_mode", "")).lower(),
    }
    rejected_terms = ("autoncp", "grl", "dann", "adversarial")
    if any(any(term in label for term in rejected_terms) for label in runtime_labels):
        raise HardwarePreflightError(
            f"hardware inference rejects runtime labels {sorted(runtime_labels)}"
        )

    return {
        "model_family": model_family,
        "feature_order": list(feature_order),
        "hidden_units": hidden_units,
        "output_dim": actual_output_dim,
        "input_dim": TARGET_CHANNELS,
        "seq_len": TARGET_SEQ_LEN,
        "channels": TARGET_CHANNELS,
    }


def load_checkpoint_bundle(
    checkpoint_path: Path,
    device: torch.device,
) -> CheckpointBundle:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = dict(checkpoint["config"])
    output_dim = _state_dict_output_dim(state_dict)
    if output_dim is None:
        raise HardwarePreflightError("checkpoint is missing a two-dimensional head.weight")
    metadata = validate_deployment_config(config, output_dim=output_dim)
    target_contract = checkpoint.get("target_contract")
    validate_target_contract(target_contract, config=config, output_dim=output_dim)
    validate_firmware_state_dict(state_dict, output_dim=output_dim)
    model = build_cfc_regressor(
        input_dim=metadata["input_dim"],
        output_dim=metadata["output_dim"],
        hidden_units=metadata["hidden_units"],
        model_family=metadata["model_family"],
        cfc_dropout=float(config.get("cfc_dropout", 0.0)),
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return CheckpointBundle(
        model=model,
        state_dict=state_dict,
        config=config,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256(checkpoint_path),
        target_normalization_stats=checkpoint.get("target_normalization_stats"),
        target_contract=target_contract,
    )


def quantize_firmware_weights(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    """Mirror export_weights.py: row-wise INT8 matrices and FP32 biases."""
    quantized: dict[str, torch.Tensor] = {}
    info: dict[str, dict[str, Any]] = {}
    for name, parameter in state_dict.items():
        if not torch.is_tensor(parameter):
            quantized[name] = parameter
            info[name] = {"type": "non_tensor", "param_count": 0, "bits": 0}
        elif parameter.is_floating_point() and parameter.ndim >= 2:
            values = parameter.detach().float()
            maximum = values.abs().flatten(start_dim=1).max(dim=1).values
            maximum = torch.where(maximum < 1e-10, torch.ones_like(maximum), maximum)
            scale = maximum / 127.0
            reshape = (scale.shape[0],) + (1,) * (values.ndim - 1)
            q_value = torch.clamp(
                torch.round(values / scale.reshape(reshape)),
                -127,
                127,
            ).to(torch.int8)
            quantized[name] = q_value
            info[name] = {
                "type": "int8",
                "mode": "per_output_channel",
                "scale": scale.cpu().tolist(),
                "param_count": int(values.numel()),
                "bits": 8,
            }
        else:
            quantized[name] = parameter.detach().clone()
            info[name] = {
                "type": "float_passthrough" if parameter.is_floating_point() else "non_float",
                "param_count": int(parameter.numel()),
                "bits": int(parameter.element_size() * 8),
            }
    return quantized, info


def dequantize_firmware_weights(
    quantized: dict[str, torch.Tensor],
    info: dict[str, dict[str, Any]],
) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for name, parameter in quantized.items():
        layer_info = info[name]
        if layer_info.get("type") != "int8":
            state_dict[name] = parameter
            continue
        scale = torch.tensor(layer_info["scale"], dtype=torch.float32)
        reshape = (scale.shape[0],) + (1,) * (parameter.ndim - 1)
        state_dict[name] = parameter.float() * scale.reshape(reshape)
    return state_dict


def compute_size_stats(q_info: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    total_count = sum(int(item.get("param_count", 0)) for item in q_info.values())
    int8_count = sum(
        int(item.get("param_count", 0))
        for item in q_info.values()
        if item.get("type") == "int8"
    )
    passthrough_bytes = sum(
        int(item.get("param_count", 0)) * int(item.get("bits", 32)) / 8
        for item in q_info.values()
        if item.get("type") != "int8"
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
) -> dict[str, float | bool]:
    input_kb = seq_len * input_dim * dtype_bytes / 1024
    hidden_kb = hidden_units * dtype_bytes / 1024
    output_kb = output_dim * dtype_bytes / 1024
    norm_kb = (input_dim * 2 + output_dim * 2) * 4 / 1024
    scratch_kb = scratch_multiplier * (
        hidden_kb + output_kb + input_dim * dtype_bytes / 1024
    )
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


def compute_error_report(
    pred_fp32: torch.Tensor,
    pred_int8: torch.Tensor,
) -> dict[str, Any]:
    difference = (pred_fp32 - pred_int8).abs()
    output_range = pred_fp32.max() - pred_fp32.min()
    max_abs_error = difference.max()
    per_target_mae = difference.mean(dim=0)
    return {
        "formula": RELATIVE_ERROR_FORMULA,
        "max_abs_error": float(max_abs_error.item()),
        "mean_abs_error": float(difference.mean().item()),
        "relative_error_pct": float((max_abs_error / (output_range + 1e-8) * 100).item()),
        "per_target_mae": [float(value) for value in per_target_mae],
    }


def _inverse_target_normalization(
    prediction: torch.Tensor,
    stats: dict[str, Any] | None,
) -> tuple[torch.Tensor, str]:
    if stats is None:
        raise HardwarePreflightError(
            "checkpoint is missing target inverse-normalization statistics"
        )
    if not isinstance(stats, dict):
        raise HardwarePreflightError("target normalization statistics must be a dictionary")
    if stats.get("method") != "mu_law":
        raise HardwarePreflightError("firmware requires mu-law target normalization")
    try:
        center = torch.as_tensor(stats["center"], dtype=prediction.dtype)
        scale = torch.as_tensor(stats["scale"], dtype=prediction.dtype)
        mu = float(stats["mu"])
    except (KeyError, TypeError, ValueError) as error:
        raise HardwarePreflightError(
            "target inverse-normalization requires center, scale, and mu"
        ) from error
    if center.shape != (prediction.shape[-1],) or scale.shape != center.shape:
        raise HardwarePreflightError("target normalization width must match model output")
    if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or not np.isfinite(mu):
        raise HardwarePreflightError("target normalization statistics must be finite")
    if torch.any(scale <= 0) or mu <= 0:
        raise HardwarePreflightError("target normalization scale and mu must be positive")
    expanded = torch.sign(prediction) * torch.expm1(
        prediction.abs() * np.log1p(mu)
    ) / mu
    return expanded * scale + center, "joint_angle_original_scale"


def run_preflight(
    *,
    checkpoint: Path,
    output_dir: Path,
    seed: int = 42,
    batch_size: int = 4,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    bundle = load_checkpoint_bundle(checkpoint, device)
    inferred_output_dim = _state_dict_output_dim(bundle.state_dict)
    if inferred_output_dim is None:
        inferred_output_dim = int(bundle.config.get("output_dim", LEGACY_OUTPUT_DIM))
    metadata = validate_deployment_config(
        bundle.config,
        output_dim=inferred_output_dim,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    golden_input = torch.randn(
        batch_size,
        metadata["seq_len"],
        metadata["input_dim"],
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    with torch.no_grad():
        fp32_normalized = bundle.model(golden_input).cpu()

    quantized, quantization_info = quantize_firmware_weights(bundle.state_dict)
    dequantized = dequantize_firmware_weights(quantized, quantization_info)
    quantized_model = copy.deepcopy(bundle.model)
    quantized_model.load_state_dict(dequantized)
    quantized_model.eval()
    with torch.no_grad():
        int8_normalized = quantized_model(golden_input).cpu()

    expected_shape = [batch_size, metadata["output_dim"]]
    if list(fp32_normalized.shape) != expected_shape:
        raise HardwarePreflightError(
            f"model output shape must be {expected_shape}, got {list(fp32_normalized.shape)}"
        )
    pred_fp32, output_units = _inverse_target_normalization(
        fp32_normalized,
        bundle.target_normalization_stats,
    )
    pred_int8, _ = _inverse_target_normalization(
        int8_normalized,
        bundle.target_normalization_stats,
    )
    error = compute_error_report(pred_fp32, pred_int8)
    if error["relative_error_pct"] >= MAX_RELATIVE_ERROR_PCT:
        raise HardwarePreflightError(
            "INT8 quantization relative error "
            f"{error['relative_error_pct']:.4f}% exceeds "
            f"{MAX_RELATIVE_ERROR_PCT:.1f}% threshold"
        )
    size = compute_size_stats(quantization_info)
    memory = estimate_core_sram_kb(
        input_dim=metadata["input_dim"],
        seq_len=metadata["seq_len"],
        hidden_units=metadata["hidden_units"],
        output_dim=metadata["output_dim"],
        weight_kb=float(size["int8_size_kb"]),
        dtype_bytes=1,
    )
    if not memory["fits_budget"]:
        raise HardwarePreflightError(
            "INT8 core SRAM estimate "
            f"{memory['total_core_sram_kb']:.2f} KB exceeds "
            f"{memory['budget_kb']:.2f} KB budget"
        )
    report = {
        "schema_version": 2,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": bundle.checkpoint_sha256,
        },
        "model": {
            **metadata,
            "architecture": "DenseCfCLinearRegressor",
            "cfc_dropout": float(bundle.config.get("cfc_dropout", 0.0)),
            "total_params": int(size["total_params"]),
        },
        "golden": {
            "seed": seed,
            "input_shape": list(golden_input.shape),
            "output_shape": list(pred_fp32.shape),
            "output_units": output_units,
        },
        "quantization": {
            "method": "per-output-channel symmetric INT8 weights; FP32 biases",
            "size": size,
            "passes_error_gate": True,
        },
        "error": error,
        "memory": {
            "sram_budget_kb": SRAM_BUDGET_KB,
            "int8_core_estimate": memory,
            "int8_deployable_estimate": bool(memory["fits_budget"]),
        },
        "portability": {
            "status": "WARN",
            "reason": "Python preflight does not replace compiled C golden parity.",
            "next_gate": "Compile and run test_cfc_pc.c and test_features_pc.c.",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "golden_input": golden_input.cpu(),
            "pred_fp32": pred_fp32,
            "pred_int8_dequantized": pred_int8,
            "quantization_info": quantization_info,
        },
        output_dir / "golden_tensors.pt",
    )
    with (output_dir / "hardware_preflight_report.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DenseCfC hardware preflight")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("log/hardware_preflight"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        report = run_preflight(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    except HardwarePreflightError as error:
        print(f"Hardware preflight rejected checkpoint: {error}")
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
