import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_DIR = PROJECT_ROOT / "src" / "hardwareOperation"
if str(HARDWARE_DIR) not in sys.path:
    sys.path.insert(0, str(HARDWARE_DIR))

import hardware_preflight as hp


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


class HardwarePreflightTests(unittest.TestCase):
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
