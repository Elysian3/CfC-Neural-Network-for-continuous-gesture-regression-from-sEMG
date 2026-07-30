import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_DIR = PROJECT_ROOT / "src" / "hardwareOperation"
if str(HARDWARE_DIR) not in sys.path:
    sys.path.insert(0, str(HARDWARE_DIR))

import hardware_preflight as hp
import export_weights


class _TinyGoldenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.5, -0.25, 0.75, -1.0, 1.25]))
        self.bias = torch.nn.Parameter(torch.tensor([0.1, 0.2, -0.1, 0.0, 0.3]))

    def forward(self, x):
        activation = x.mean(dim=(1, 2)).unsqueeze(1)
        return activation * self.weight + self.bias


def _valid_config():
    return {
        "model_family": "dense_cfc_linear",
        "feature_order": ("rms",),
        "hidden_units": 256,
        "cfc_dropout": 0.3,
    }


def _bundle():
    model = _TinyGoldenModel()
    return hp.CheckpointBundle(
        model=model,
        state_dict=model.state_dict(),
        config=_valid_config(),
        checkpoint_path=Path("synthetic.pt"),
        checkpoint_sha256="abc123",
    )


def _valid_export_state_dict():
    state_dict = {}
    for _c_name, state_key, in_dim, out_dim in export_weights.LAYER_SPEC:
        state_dict[f"{state_key}.weight"] = torch.zeros((out_dim, in_dim))
        state_dict[f"{state_key}.bias"] = torch.zeros(out_dim)
    return state_dict


class HardwarePreflightTests(unittest.TestCase):
    def test_obsolete_entrypoint_files_are_absent(self):
        obsolete_paths = (
            HARDWARE_DIR / "quantize_test.py",
            HARDWARE_DIR / "load-model.cpp",
            PROJECT_ROOT / "_run_loo_all28.py",
            PROJECT_ROOT / "main.py",
        )
        self.assertEqual([path for path in obsolete_paths if path.exists()], [])

    def test_hardware_operation_does_not_duplicate_firmware_artifacts(self):
        duplicate_paths = tuple(
            HARDWARE_DIR / name
            for name in (
                "cfc_inference.c",
                "cfc_inference.h",
                "cfc_inference.o",
                "features.c",
                "features.h",
                "features.o",
                "main.c",
                "normalization.h",
                "test_cfc_pc.exe",
                "test_features_pc.exe",
                "weights.h",
            )
        )
        self.assertEqual([path for path in duplicate_paths if path.exists()], [])

    def test_export_defaults_to_canonical_firmware_directory(self):
        stats = {
            "center": np.zeros(12, dtype=np.float32),
            "scale": np.ones(12, dtype=np.float32),
        }
        with (
            patch.object(export_weights.np, "load", return_value=stats),
            patch.object(export_weights, "run_export", return_value={}) as run_export,
        ):
            export_weights.main(
                ["--checkpoint", "synthetic.pt", "--norm-stats", "stats.npz"]
            )

        self.assertEqual(
            run_export.call_args.kwargs["output_dir"],
            PROJECT_ROOT / "firmware" / "main",
        )

    def test_export_requires_training_normalization_stats_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint = root / "model.pt"
            output_dir = root / "firmware"
            output_dir.mkdir()
            weights_path = output_dir / "weights.h"
            normalization_path = output_dir / "normalization.h"
            weights_path.write_text("original weights", encoding="utf-8")
            normalization_path.write_text("original normalization", encoding="utf-8")
            torch.save(
                {
                    "model_state_dict": _valid_export_state_dict(),
                    "config": {"feature_order": ("rms",), "hidden_units": 256},
                },
                checkpoint,
            )

            with self.assertRaisesRegex(ValueError, "training normalization"):
                export_weights.run_export(
                    checkpoint_path=checkpoint,
                    output_dir=output_dir,
                )

            self.assertEqual(weights_path.read_text(encoding="utf-8"), "original weights")
            self.assertEqual(
                normalization_path.read_text(encoding="utf-8"),
                "original normalization",
            )

    def test_export_rejects_invalid_normalization_stats_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint = root / "model.pt"
            output_dir = root / "firmware"
            output_dir.mkdir()
            torch.save(
                {
                    "model_state_dict": _valid_export_state_dict(),
                    "config": {"feature_order": ("rms",), "hidden_units": 256},
                },
                checkpoint,
            )

            with self.assertRaisesRegex(ValueError, "12 values"):
                export_weights.run_export(
                    checkpoint_path=checkpoint,
                    output_dir=output_dir,
                    norm_center=np.zeros(2, dtype=np.float32),
                    norm_scale=np.ones(2, dtype=np.float32),
                )

            self.assertFalse((output_dir / "weights.h").exists())
            self.assertFalse((output_dir / "normalization.h").exists())

    def test_accepts_current_rms_dense_cfc_metadata(self):
        metadata = hp.validate_deployment_config(_valid_config(), output_dim=5)

        self.assertEqual(metadata["model_family"], "dense_cfc_linear")
        self.assertEqual(metadata["feature_order"], ["rms"])
        self.assertEqual(metadata["hidden_units"], 256)
        self.assertEqual(metadata["input_dim"], 12)  # RMS-only over 12 EMG channels.
        self.assertEqual(metadata["output_dim"], 5)

    def test_rejects_autoncp_non_rms_and_wrong_output_dim(self):
        with self.assertRaises(hp.HardwarePreflightError):
            hp.validate_deployment_config({**_valid_config(), "model_family": "autoncp"})

        with self.assertRaises(hp.HardwarePreflightError):
            hp.validate_deployment_config({**_valid_config(), "feature_order": ("mav", "rms")})

        with self.assertRaises(hp.HardwarePreflightError):
            hp.validate_deployment_config(_valid_config(), output_dim=10)

        with self.assertRaises(hp.HardwarePreflightError):
            hp.validate_deployment_config({**_valid_config(), "output_dim": 10})

        with self.assertRaises(hp.HardwarePreflightError):
            hp.validate_deployment_config({**_valid_config(), "use_grl": True})

        with self.assertRaises(hp.HardwarePreflightError):
            hp.validate_deployment_config({**_valid_config(), "runtime_mode": "adversarial"})

    def test_relative_error_formula_uses_max_error_over_output_range(self):
        fp32 = torch.tensor([[1.0, 3.0, 5.0, 7.0, 9.0]])
        int8 = torch.tensor([[1.5, 3.0, 4.5, 7.0, 9.0]])

        report = hp.compute_error_report(fp32, int8)

        self.assertEqual(report["formula"], hp.RELATIVE_ERROR_FORMULA)
        self.assertAlmostEqual(report["max_abs_error"], 0.5)
        # Range is 8.0, so 0.5 / 8.0 * 100 = 6.25%.
        self.assertAlmostEqual(report["relative_error_pct"], 6.25, places=5)

    def test_size_and_sram_estimates_are_explicit(self):
        q_info = {
            "w": {"type": "int8", "param_count": 1024, "bits": 8},
            "b": {"type": "int8", "param_count": 256, "bits": 8},
            "counter": {"type": "non_float", "param_count": 1, "bits": 32},
        }

        size = hp.compute_size_stats(q_info)
        sram = hp.estimate_core_sram_kb(
            input_dim=12,
            seq_len=8,
            hidden_units=256,
            output_dim=5,
            weight_kb=float(size["int8_size_kb"]),
            dtype_bytes=1,
        )

        self.assertEqual(size["total_params"], 1281)
        self.assertAlmostEqual(size["int8_size_kb"], (1024 + 256 + 4) / 1024)
        self.assertEqual(sram["budget_kb"], 400.0)
        self.assertFalse(sram["uses_psram_for_core_inference"])
        self.assertTrue(sram["fits_budget"])

    def test_preflight_report_and_golden_tensors_are_deterministic(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            with patch.object(hp, "load_checkpoint_bundle", side_effect=[_bundle(), _bundle()]):
                first = hp.run_preflight(
                    checkpoint=Path("synthetic.pt"),
                    output_dir=Path(first_dir),
                    seed=123,
                    batch_size=2,
                )
                second = hp.run_preflight(
                    checkpoint=Path("synthetic.pt"),
                    output_dir=Path(second_dir),
                    seed=123,
                    batch_size=2,
                )

            self.assertEqual(first["model"]["architecture"], "DenseCfCLinearRegressor")
            self.assertEqual(first["golden"]["input_shape"], [2, 8, 12])
            self.assertEqual(first["portability"]["status"], "WARN")
            self.assertLess(first["error"]["relative_error_pct"], 5.0)
            self.assertEqual(first, second)

            report_path = Path(first_dir) / "hardware_preflight_report.json"
            tensors_path = Path(first_dir) / "golden_tensors.pt"
            self.assertTrue(report_path.exists())
            self.assertTrue(tensors_path.exists())
            with report_path.open(encoding="utf-8") as handle:
                on_disk = json.load(handle)
            self.assertEqual(on_disk, first)


if __name__ == "__main__":
    unittest.main()
