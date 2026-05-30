"""INT8 quantization feasibility test for AutoNCP CfC model.

Tests manual weight quantization (simulates ESP32 deployment):
1. Load trained model
2. Quantize each weight tensor to INT8 with per-tensor scale/zero-point
3. Run dequantized inference and compare to FP32 baseline
"""
import argparse, json, sys
from pathlib import Path

import torch

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src" / "deep learning"))

from train import build_cfc_regressor


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    config = ckpt["config"]
    input_dim = len(config["feature_order"]) * 12
    output_dim = 5
    model = build_cfc_regressor(
        input_dim=input_dim, output_dim=output_dim,
        hidden_units=config["hidden_units"],
        model_family=config["model_family"],
        cfc_dropout=config["cfc_dropout"],
    )
    model.load_state_dict(sd)
    model.eval()
    return model


def quantize_weights_int8(state_dict: dict) -> tuple[dict, dict]:
    """Quantize weight tensors to INT8 with per-tensor symmetric quantization.

    Returns (quantized_state_dict, quantization_info) where:
      state_dict stores INT8 weights (as torch.int8) + scale factors
      quantization_info records scale/range per parameter
    """
    q_sd = {}
    q_info = {}

    for name, param in state_dict.items():
        if "sparsity_mask" in name or "num_batches_tracked" in name:
            q_sd[name] = param
            q_info[name] = {"type": "passthrough", "bits": 32}
            continue

        if not param.is_floating_point():
            q_sd[name] = param
            q_info[name] = {"type": "non_float", "dtype": str(param.dtype)}
            continue

        # Symmetric per-tensor INT8 quantization
        w = param.detach()
        w_max = w.abs().max().item()
        if w_max < 1e-10:
            scale = 1.0
        else:
            scale = w_max / 127.0  # INT8 symmetric range: [-127, 127]

        w_q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
        q_sd[name] = w_q
        q_info[name] = {
            "type": "int8",
            "scale": float(scale),
            "range": [-127 * scale, 127 * scale],
            "param_count": int(w.numel()),
            "bits": 8,
        }

    return q_sd, q_info


def dequantize_weights(q_sd: dict, q_info: dict) -> dict:
    """Dequantize INT8 weights back to FP32 for inference."""
    dq_sd = {}
    for name, param in q_sd.items():
        info = q_info[name]
        if info.get("type") == "int8":
            scale = info["scale"]
            dq_sd[name] = param.float() * scale
        else:
            dq_sd[name] = param
    return dq_sd


def compute_size_stats(q_info: dict) -> dict:
    total_params = 0
    int8_params = 0
    fp32_params = 0
    for name, info in q_info.items():
        n = info.get("param_count", 0)
        total_params += n
        if info.get("type") == "int8":
            int8_params += n
        elif info.get("bits", 32) == 32:
            fp32_params += n
    int8_kb = int8_params * 1 / 1024 + (fp32_params * 4 / 1024)
    fp32_kb = total_params * 4 / 1024
    return {
        "total_params": total_params,
        "int8_params": int8_params,
        "fp32_params": fp32_params,
        "int8_coverage_pct": int8_params / total_params * 100 if total_params > 0 else 0,
        "fp32_size_kb": fp32_kb,
        "int8_size_kb": int8_kb,
        "compression_ratio": fp32_kb / int8_kb if int8_kb > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="log/db2_paper_cfc_finetune/autoncp_doa5_all28_20260524/checkpoints/S1_dense_cfc_head_ft.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("log/quantization"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 60)
    print("INT8 Quantization Feasibility Test (Manual per-tensor INT8)")
    print("=" * 60)

    # 1. Load
    print("\n[1/4] Loading model...")
    model = load_model(args.checkpoint, device)
    sd = model.state_dict()
    total_params = sum(p.numel() for p in sd.values())
    print(f"  Total parameters: {total_params:,}")
    print(f"  FP32 size: {total_params * 4 / 1024:.1f} KB")

    # 2. FP32 baseline
    print("\n[2/4] FP32 baseline inference...")
    torch.manual_seed(42)
    x_test = torch.randn(200, 8, 60)
    with torch.no_grad():
        pred_fp32 = model(x_test).cpu()

    # 3. Quantize + dequantize + re-infer
    print("\n[3/4] INT8 quantization...")
    q_sd, q_info = quantize_weights_int8(sd)
    dq_sd = dequantize_weights(q_sd, q_info)

    model.load_state_dict(dq_sd)
    model.eval()
    with torch.no_grad():
        pred_int8 = model(x_test).cpu()

    size_stats = compute_size_stats(q_info)
    print(f"  INT8-quantized params: {size_stats['int8_params']:,} ({size_stats['int8_coverage_pct']:.1f}%)")
    print(f"  FP32 (passthrough): {size_stats['fp32_params']:,}")
    print(f"  Size: {size_stats['fp32_size_kb']:.0f} KB (FP32) -> {size_stats['int8_size_kb']:.0f} KB (INT8)")

    # 4. Compare
    print("\n[4/4] Comparing...")
    max_err = torch.abs(pred_fp32 - pred_int8).max()
    mae = torch.abs(pred_fp32 - pred_int8).mean()
    rng = pred_fp32.max() - pred_fp32.min()
    rel_err_pct = float(max_err / (rng + 1e-8) * 100)

    # Per-param quantization error analysis
    max_scale = 0
    min_scale = float("inf")
    for name, info in q_info.items():
        if info.get("type") == "int8":
            max_scale = max(max_scale, info["scale"])
            min_scale = min(min_scale, info["scale"])

    doa_names = ["thumb_rotation", "thumb_flexion", "index_flexion", "middle_flexion", "ring_little_flexion"]
    per_doa_mae = (pred_fp32 - pred_int8).abs().mean(0)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Quantization error:")
    print(f"    Max abs error: {max_err.item():.4e}")
    print(f"    Mean abs error: {mae.item():.4e}")
    print(f"    Relative error: {rel_err_pct:.2f}%")
    print(f"    Scale range: [{min_scale:.2e}, {max_scale:.2e}]")

    print(f"\n  Per-DoA MAE:")
    for i, name in enumerate(doa_names):
        print(f"    {name:<25} {per_doa_mae[i].item():.2e}")

    # ESP32-S3 SRAM budget: ~400 KB for model weights + runtime buffers (512 KB total)
    ESP32_MAX_SRAM_KB = 400
    deployable = rel_err_pct < 5.0 and size_stats["int8_size_kb"] < ESP32_MAX_SRAM_KB
    print(f"\n  VERDICT:")
    print(f"    Relative error < 5%: {'PASS' if rel_err_pct < 5.0 else 'FAIL'} ({rel_err_pct:.2f}%)")
    print(f"    Size < 400 KB SRAM: {'PASS' if size_stats['int8_size_kb'] < 400 else 'FAIL'} ({size_stats['int8_size_kb']:.0f} KB)")
    print(f"    {'DEPLOYABLE' if deployable else 'NOT DEPLOYABLE'}")

    report = {
        "model": {"architecture": "AutoNCP", "total_params": total_params},
        "size": size_stats,
        "quantization": {
            "method": "per-tensor symmetric INT8",
            "scale_range": [float(min_scale), float(max_scale)],
            "num_quantized_tensors": sum(1 for v in q_info.values() if v.get("type") == "int8"),
        },
        "error": {
            "max_abs_error": float(max_err.item()),
            "mean_abs_error": float(mae.item()),
            "relative_error_pct": rel_err_pct,
            "per_doa_mae": [float(v) for v in per_doa_mae],
            "doa_names": doa_names,
        },
        "deployable": deployable,
    }
    report_path = str(args.output_dir / "quantization_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
