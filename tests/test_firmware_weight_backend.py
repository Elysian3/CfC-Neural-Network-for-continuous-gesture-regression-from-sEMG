import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFC_SOURCE = PROJECT_ROOT / "firmware" / "main" / "cfc_inference.c"
CFC_HEADER = PROJECT_ROOT / "firmware" / "main" / "cfc_inference.h"
FIRMWARE_MAIN_SOURCE = PROJECT_ROOT / "firmware" / "main" / "main.c"
EXPORTER_DIR = PROJECT_ROOT / "src" / "hardwareOperation"
PC_CFC_TEST = EXPORTER_DIR / "test_cfc_pc.c"
PC_FEATURE_TEST = EXPORTER_DIR / "test_features_pc.c"
if str(EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORTER_DIR))

import export_weights


class FirmwareWeightBackendTests(unittest.TestCase):
    @staticmethod
    def _golden_state_dict(output_dim: int) -> dict[str, torch.Tensor]:
        """Return a deterministic, nonzero checkpoint for compiled-C parity."""
        state_dict = {}
        for layer_index, (_name, state_key, input_dim, layer_output_dim) in enumerate(
            export_weights._export_layers(output_dim)
        ):
            rows = torch.arange(layer_output_dim, dtype=torch.float32).unsqueeze(1)
            columns = torch.arange(input_dim, dtype=torch.float32).unsqueeze(0)
            pattern = torch.remainder(rows * 17 + columns * 31 + layer_index * 11, 29)
            state_dict[f"{state_key}.weight"] = (pattern - 14.0) / 700.0
            bias_pattern = torch.remainder(
                torch.arange(layer_output_dim, dtype=torch.float32) * 7 + layer_index,
                13,
            )
            state_dict[f"{state_key}.bias"] = (bias_pattern - 6.0) / 500.0
        return state_dict

    @staticmethod
    def _golden_lut_lookup(table: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Match cfc_inference.c's float LUT interpolation semantics."""
        values = np.asarray(values, dtype=np.float32)
        lut_min = np.float32(-8.0)
        lut_max = np.float32(8.0)
        lut_dx = np.float32((lut_max - lut_min) / np.float32(255.0))
        clipped = np.clip(values, lut_min, lut_max).astype(np.float32)
        idx_float = ((clipped - lut_min) / lut_dx).astype(np.float32)
        indices = np.minimum(idx_float.astype(np.intp), 254)
        fractions = (idx_float - indices.astype(np.float32)).astype(np.float32)
        interpolated = (
            table[indices] * (np.float32(1.0) - fractions)
            + table[indices + 1] * fractions
        ).astype(np.float32)
        return np.where(
            values <= lut_min,
            table[0],
            np.where(values >= lut_max, table[-1], interpolated),
        )

    @classmethod
    def _compiled_golden_reference(
        cls,
        state_dict: dict[str, torch.Tensor],
        output_dim: int,
        features: np.ndarray,
        target_center: np.ndarray,
        target_scale: np.ndarray,
        target_mu: float,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Model exported INT8 weights and C activation/inverse-normalization math."""
        parameters = {}
        for name, state_key, _input_dim, _layer_output_dim in export_weights._export_layers(
            output_dim
        ):
            quantized, scales = export_weights.quantize_per_channel(
                state_dict[f"{state_key}.weight"]
            )
            parameters[name] = (
                quantized.numpy().astype(np.float32),
                scales.astype(np.float32),
                state_dict[f"{state_key}.bias"].numpy().astype(np.float32),
            )

        lut_x = np.linspace(-8.0, 8.0, 256, dtype=np.float32)
        lecun_lut = (np.float32(1.7159) * np.tanh(np.float32(0.666) * lut_x)).astype(
            np.float32
        )
        tanh_lut = np.tanh(lut_x).astype(np.float32)
        sigmoid_lut = (np.float32(1.0) / (np.float32(1.0) + np.exp(-lut_x))).astype(
            np.float32
        )

        def fc(name: str, inputs: np.ndarray) -> np.ndarray:
            weights, scales, bias = parameters[name]
            # This uses exporter INT8 values and per-output-channel scales.
            dots = weights @ inputs.astype(np.float32)
            return (bias + dots.astype(np.float32) * scales).astype(np.float32)

        def step(inputs: np.ndarray, hidden: np.ndarray) -> np.ndarray:
            backbone = cls._golden_lut_lookup(
                lecun_lut, fc("backbone", np.concatenate((inputs, hidden)))
            )
            ff1 = cls._golden_lut_lookup(tanh_lut, fc("ff1", backbone))
            ff2 = cls._golden_lut_lookup(tanh_lut, fc("ff2", backbone))
            gate = cls._golden_lut_lookup(
                sigmoid_lut, fc("time_a", backbone) + fc("time_b", backbone)
            )
            return (ff1 * (np.float32(1.0) - gate) + gate * ff2).astype(np.float32)

        def output(hidden: np.ndarray) -> np.ndarray:
            normalized = fc("head", hidden)
            log_mu = np.log1p(np.float32(target_mu)).astype(np.float32)
            expanded = (
                np.sign(normalized)
                * np.expm1(np.abs(normalized) * log_mu).astype(np.float32)
                / np.float32(target_mu)
            )
            return (expanded * target_scale + target_center).astype(np.float32)

        hidden = np.zeros(export_weights.HIDDEN_DIM, dtype=np.float32)
        streaming_outputs = []
        for feature in features:
            hidden = step(feature, hidden)
            streaming_outputs.append(output(hidden))
        sequence_output = output(hidden)
        continued_output = output(step(features[0], hidden))
        return sequence_output, streaming_outputs, continued_output

    @unittest.skipUnless(shutil.which("gcc"), "gcc is required for compiled C golden parity")
    def test_compiled_c_matches_nonzero_exported_int8_golden_reference(self):
        """Compare canonical firmware C against an independent exported-INT8 oracle."""
        for output_dim in (5, 10, 13):
            with self.subTest(output_dim=output_dim), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                state_dict = self._golden_state_dict(output_dim)
                features = np.asarray(
                    [
                        [((t + 1) * (feature + 3)) % 19 - 9 for feature in range(12)]
                        for t in range(8)
                    ],
                    dtype=np.float32,
                ) / np.float32(11.0)
                target_center = np.linspace(-0.75, 0.75, output_dim, dtype=np.float32)
                target_scale = np.linspace(0.35, 0.65, output_dim, dtype=np.float32)
                target_mu = 31.0
                checkpoint = root / "nonzero_model.pt"
                torch.save(
                    {
                        "model_state_dict": state_dict,
                        "target_normalization_stats": {
                            "method": "mu_law",
                            "center": target_center,
                            "scale": target_scale,
                            "mu": target_mu,
                        },
                    },
                    checkpoint,
                )
                export_weights.run_export(
                    checkpoint_path=checkpoint,
                    output_dir=root,
                    norm_center=np.zeros(12, dtype=np.float32),
                    norm_scale=np.ones(12, dtype=np.float32),
                )

                sequence_expected, streaming_expected, continued_expected = (
                    self._compiled_golden_reference(
                        state_dict,
                        output_dim,
                        features,
                        target_center,
                        target_scale,
                        target_mu,
                    )
                )
                self.assertTrue(
                    all(
                        torch.count_nonzero(weight).item()
                        for key, weight in state_dict.items()
                        if key.endswith(".weight")
                    )
                )
                shutil.copy2(CFC_SOURCE, root / "cfc_inference.c")
                shutil.copy2(CFC_HEADER, root / "cfc_inference.h")
                driver = root / "compiled_golden_driver.c"
                feature_literal = ",\n        ".join(
                    "{" + ", ".join(f"{value:.9f}f" for value in frame) + "}"
                    for frame in features
                )
                driver.write_text(
                    f"""
#include <stdio.h>
#include "weights.h"
#include "cfc_inference.h"

static void emit(const char *path, int frame, const float output[CFC_OUTPUT_DIM]) {{
    for (int i = 0; i < CFC_OUTPUT_DIM; i++) {{
        printf("%s %d %d %.9g\\n", path, frame, i, output[i]);
    }}
}}

int main(void) {{
    static const float features[CFC_SEQ_LEN][CFC_INPUT_DIM] = {{
        {feature_literal}
    }};
    float output[CFC_OUTPUT_DIM];
    cfc_lut_init();
    cfc_inference_int8(features, output);
    emit("sequence", -1, output);

    cfc_reset_state();
    for (int t = 0; t < CFC_SEQ_LEN; t++) {{
        cfc_single_step(features[t], output);
        emit("stream", t, output);
    }}
    cfc_single_step(features[0], output);
    emit("continued", CFC_SEQ_LEN, output);

    cfc_reset_state();
    for (int t = 0; t < CFC_SEQ_LEN; t++) {{
        cfc_single_step(features[t], output);
        emit("reset", t, output);
    }}
    return 0;
}}
""".lstrip(),
                    encoding="utf-8",
                )
                executable = root / f"compiled_golden_{output_dim}.exe"
                compile_result = subprocess.run(
                    [
                        "gcc",
                        "-O2",
                        "-std=c11",
                        str(driver),
                        str(root / "cfc_inference.c"),
                        "-lm",
                        "-o",
                        str(executable),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
                run_result = subprocess.run(
                    [str(executable)], check=False, capture_output=True, text=True
                )
                self.assertEqual(run_result.returncode, 0, run_result.stderr)

                actual = {}
                for line in run_result.stdout.splitlines():
                    path, frame, output_index, value = line.split()
                    actual[(path, int(frame), int(output_index))] = float(value)
                expected = {
                    **{
                        ("sequence", -1, index): value
                        for index, value in enumerate(sequence_expected)
                    },
                    **{
                        ("stream", frame, index): value
                        for frame, output in enumerate(streaming_expected)
                        for index, value in enumerate(output)
                    },
                    **{
                        ("continued", 8, index): value
                        for index, value in enumerate(continued_expected)
                    },
                    **{
                        ("reset", frame, index): value
                        for frame, output in enumerate(streaming_expected)
                        for index, value in enumerate(output)
                    },
                }
                self.assertEqual(set(actual), set(expected))
                for key, expected_value in expected.items():
                    self.assertAlmostEqual(
                        actual[key],
                        float(expected_value),
                        delta=3e-5,
                        msg=f"{key}: compiled C must match the exported INT8 reference",
                    )

    def test_cfc_reads_all_weight_matrices_directly_from_flash(self):
        source = CFC_SOURCE.read_text(encoding="utf-8")
        flash_weight_symbols = (
            "cfc_backbone_weight",
            "cfc_ff1_weight",
            "cfc_ff2_weight",
            "cfc_time_a_weight",
            "cfc_time_b_weight",
            "cfc_head_weight",
        )

        self.assertNotIn("sram_", source)
        self.assertNotRegex(source, r"memcpy\s*\([^;]*cfc_\w+_weight")
        for symbol in flash_weight_symbols:
            self.assertRegex(
                source,
                re.compile(rf"fc_int8\([^;]*\b{symbol}\b", re.DOTALL),
                msg=f"{symbol} must be passed directly to fc_int8",
            )

    def test_exported_output_dimension_follows_checkpoint_head_weight(self):
        for output_dim in (5, 10, 13):
            state_dict = {}
            for _name, state_key, input_dim, layer_output_dim in export_weights.LAYER_SPEC:
                if state_key == "head":
                    layer_output_dim = output_dim
                state_dict[f"{state_key}.weight"] = torch.zeros((layer_output_dim, input_dim))
                state_dict[f"{state_key}.bias"] = torch.zeros(layer_output_dim)

            with tempfile.TemporaryDirectory() as temporary_directory:
                checkpoint_path = Path(temporary_directory) / "model.pt"
                output_dir = Path(temporary_directory) / "firmware"
                torch.save(
                    {
                        "model_state_dict": state_dict,
                        "target_normalization_stats": {
                            "method": "mu_law",
                            "center": np.arange(output_dim, dtype=np.float32),
                            "scale": np.arange(1, output_dim + 1, dtype=np.float32),
                            "mu": 255.0,
                        },
                    },
                    checkpoint_path,
                )
                report = export_weights.run_export(
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    norm_center=np.zeros(12, dtype=np.float32),
                    norm_scale=np.ones(12, dtype=np.float32),
                )

                weights_header = (output_dir / "weights.h").read_text(encoding="utf-8")
                self.assertIn(f"#define CFC_OUTPUT_DIM     {output_dim}", weights_header)
                self.assertIn(
                    f"static const int8_t cfc_head_weight[{output_dim} * 256]",
                    weights_header,
                )
                normalization_header = (output_dir / "normalization.h").read_text(
                    encoding="utf-8"
                )
                self.assertIn("#define TARGET_NORMALIZATION_AVAILABLE 1", normalization_header)
                self.assertIn(
                    f"#define TARGET_MU_LAW_DIM              {output_dim}",
                    normalization_header,
                )
                self.assertIn(
                    f"static const float target_mu_law_center[{output_dim}]",
                    normalization_header,
                )
                self.assertEqual(report["output_dim"], output_dim)
                self.assertEqual(report["output_units"], "joint_angle_original_scale")

    def test_export_rejects_checkpoint_without_target_inverse_stats(self):
        state_dict = {}
        for _name, state_key, input_dim, output_dim in export_weights.LAYER_SPEC:
            state_dict[f"{state_key}.weight"] = torch.zeros((output_dim, input_dim))
            state_dict[f"{state_key}.bias"] = torch.zeros(output_dim)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"
            output_dir = Path(temporary_directory) / "firmware"
            torch.save({"model_state_dict": state_dict}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "real-angle firmware outputs"):
                export_weights.run_export(
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    norm_center=np.zeros(12, dtype=np.float32),
                    norm_scale=np.ones(12, dtype=np.float32),
                )

            self.assertFalse(output_dir.exists())

    def test_export_rejects_same_width_stats_from_a_different_run(self):
        state_dict = {}
        for _name, state_key, input_dim, output_dim in export_weights.LAYER_SPEC:
            state_dict[f"{state_key}.weight"] = torch.zeros((output_dim, input_dim))
            state_dict[f"{state_key}.bias"] = torch.zeros(output_dim)
        feature_stats = {
            "method": "mu_law",
            "center": np.zeros(12, dtype=np.float32),
            "scale": np.ones(12, dtype=np.float32),
            "mu": 255.0,
        }
        target_stats = {
            "method": "mu_law",
            "center": np.zeros(5, dtype=np.float32),
            "scale": np.ones(5, dtype=np.float32),
            "mu": 255.0,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "model.pt"
            torch.save(
                {
                    "model_state_dict": state_dict,
                    "feature_normalization_stats": feature_stats,
                    "target_normalization_stats": target_stats,
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "feature.*disagree"):
                export_weights.run_export(
                    checkpoint_path=checkpoint_path,
                    output_dir=root / "wrong_feature",
                    norm_center=np.ones(12, dtype=np.float32),
                    norm_scale=np.ones(12, dtype=np.float32),
                )
            with self.assertRaisesRegex(ValueError, "target.*disagree"):
                export_weights.run_export(
                    checkpoint_path=checkpoint_path,
                    output_dir=root / "wrong_target",
                    norm_center=feature_stats["center"],
                    norm_scale=feature_stats["scale"],
                    target_center=np.ones(5, dtype=np.float32),
                    target_scale=target_stats["scale"],
                    target_mu=255.0,
                )

            self.assertFalse((root / "wrong_feature").exists())
            self.assertFalse((root / "wrong_target").exists())

    def test_export_rejects_target_contract_head_width_mismatch(self):
        state_dict = {}
        for _name, state_key, input_dim, output_dim in export_weights.LAYER_SPEC:
            state_dict[f"{state_key}.weight"] = torch.zeros((output_dim, input_dim))
            state_dict[f"{state_key}.bias"] = torch.zeros(output_dim)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "model.pt"
            torch.save(
                {
                    "model_state_dict": state_dict,
                    "target_contract": {"output_dim": 10},
                    "target_normalization_stats": {
                        "method": "mu_law",
                        "center": np.zeros(5, dtype=np.float32),
                        "scale": np.ones(5, dtype=np.float32),
                        "mu": 255.0,
                    },
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "target_contract output_dim"):
                export_weights.run_export(
                    checkpoint_path=checkpoint_path,
                    output_dir=root / "firmware",
                    norm_center=np.zeros(12, dtype=np.float32),
                    norm_scale=np.ones(12, dtype=np.float32),
                )

    def test_firmware_inverse_normalizes_both_inference_paths(self):
        source = CFC_SOURCE.read_text(encoding="utf-8")

        self.assertEqual(source.count("inverse_target_mulaw(output);"), 2)
        self.assertIn("expm1f(fabsf(normalized) * log_mu)", source)
        self.assertIn("target_mu_law_scale", source)
        self.assertIn("target_mu_law_center", source)

    def test_firmware_prints_checkpoint_defined_output_dimension(self):
        source = FIRMWARE_MAIN_SOURCE.read_text(encoding="utf-8")

        self.assertRegex(source, r"float output\[CFC_OUTPUT_DIM\]")
        self.assertRegex(source, r"output_idx < CFC_OUTPUT_DIM")
        self.assertNotRegex(source, r"output\[[0-9]+\]")

    def test_firmware_resets_hidden_state_once_at_startup(self):
        source = FIRMWARE_MAIN_SOURCE.read_text(encoding="utf-8")
        app_main = source.split("void app_main(void)", 1)[1]

        self.assertEqual(app_main.count("cfc_reset_state();"), 1)
        self.assertLess(app_main.index("cfc_lut_init();"), app_main.index("cfc_reset_state();"))
        self.assertLess(
            app_main.index("cfc_reset_state();"),
            app_main.index("xTaskCreatePinnedToCore("),
        )

    def test_firmware_dimensions_must_come_from_generated_weights_header(self):
        header = CFC_HEADER.read_text(encoding="utf-8")
        self.assertNotIn("#define CFC_OUTPUT_DIM     5", header)
        self.assertIn('#error "Include generated weights.h before cfc_inference.h"', header)

        for source_path in (CFC_SOURCE, FIRMWARE_MAIN_SOURCE):
            source = source_path.read_text(encoding="utf-8")
            weights_include = source.index('#include "weights.h"')
            inference_include = source.index('#include "cfc_inference.h"')
            self.assertLess(weights_include, inference_include)

        checked_in_normalization = (
            PROJECT_ROOT / "firmware" / "main" / "normalization.h"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "#define TARGET_NORMALIZATION_AVAILABLE 0",
            checked_in_normalization,
        )
        self.assertIn("Legacy PC-test fixture only", checked_in_normalization)
        cfc_source = CFC_SOURCE.read_text(encoding="utf-8")
        self.assertIn("CFC_ALLOW_NORMALIZED_OUTPUT_FIXTURE", cfc_source)
        self.assertIn("Target inverse-normalization is required", cfc_source)

    @unittest.skipUnless(shutil.which("gcc"), "gcc is required for fail-closed test")
    def test_checked_in_fixture_fails_closed_without_explicit_test_macro(self):
        with tempfile.TemporaryDirectory() as tmp:
            compile_result = subprocess.run(
                [
                    "gcc",
                    "-std=c11",
                    "-c",
                    str(CFC_SOURCE),
                    "-o",
                    str(Path(tmp) / "cfc_inference.o"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(compile_result.returncode, 0)
        self.assertIn("Target inverse-normalization is required", compile_result.stderr)

    @unittest.skipUnless(shutil.which("gcc"), "gcc is required for PC firmware parity tests")
    def test_pc_golden_programs_compile_and_pass(self):
        cases = (
            (
                PC_CFC_TEST,
                [CFC_SOURCE],
            ),
            (
                PC_FEATURE_TEST,
                [PROJECT_ROOT / "firmware" / "main" / "features.c", CFC_SOURCE],
            ),
        )
        for test_source, firmware_sources in cases:
            with self.subTest(test_source=test_source.name), tempfile.TemporaryDirectory() as tmp:
                executable = Path(tmp) / f"{test_source.stem}.exe"
                compile_result = subprocess.run(
                    [
                        "gcc",
                        "-O2",
                        "-std=c11",
                        "-DCFC_ALLOW_NORMALIZED_OUTPUT_FIXTURE=1",
                        str(test_source),
                        *(str(source) for source in firmware_sources),
                        "-lm",
                        "-o",
                        str(executable),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
                run_result = subprocess.run(
                    [str(executable)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(run_result.returncode, 0, run_result.stderr)

    @unittest.skipUnless(shutil.which("gcc"), "gcc is required for dynamic-output C test")
    def test_exported_output_dimensions_compile_and_return_original_scale(self):
        for output_dim in (5, 10, 13):
            with self.subTest(output_dim=output_dim), tempfile.TemporaryDirectory() as tmp:
                state_dict = {}
                for (
                    _name,
                    state_key,
                    input_dim,
                    layer_output_dim,
                ) in export_weights.LAYER_SPEC:
                    if state_key == "head":
                        layer_output_dim = output_dim
                    state_dict[f"{state_key}.weight"] = torch.zeros(
                        (layer_output_dim, input_dim)
                    )
                    state_dict[f"{state_key}.bias"] = torch.zeros(layer_output_dim)

                root = Path(tmp)
                checkpoint = root / "model.pt"
                target_center = np.arange(100, 100 + output_dim, dtype=np.float32)
                torch.save(
                    {
                        "model_state_dict": state_dict,
                        "target_normalization_stats": {
                            "method": "mu_law",
                            "center": target_center,
                            "scale": np.ones(output_dim, dtype=np.float32),
                            "mu": 255.0,
                        },
                    },
                    checkpoint,
                )
                export_weights.run_export(
                    checkpoint_path=checkpoint,
                    output_dir=root,
                    norm_center=np.zeros(12, dtype=np.float32),
                    norm_scale=np.ones(12, dtype=np.float32),
                )
                shutil.copy2(CFC_SOURCE, root / "cfc_inference.c")
                shutil.copy2(CFC_HEADER, root / "cfc_inference.h")
                driver = root / "driver.c"
                driver.write_text(
                    """
#include <math.h>
#include <stdio.h>
#include "weights.h"
#include "cfc_inference.h"

int main(void) {
    float sequence[CFC_SEQ_LEN][CFC_INPUT_DIM] = {{0}};
    float sequence_output[CFC_OUTPUT_DIM];
    float step_output[CFC_OUTPUT_DIM];
    cfc_lut_init();

    cfc_inference_int8(sequence, sequence_output);
    cfc_reset_state();
    for (int t = 0; t < CFC_SEQ_LEN; t++) {
        cfc_single_step(sequence[t], step_output);
    }

    for (int i = 0; i < CFC_OUTPUT_DIM; i++) {
        float expected = 100.0f + (float)i;
        if (fabsf(sequence_output[i] - expected) > 1e-5f) {
            fprintf(stderr, "sequence_output[%d]=%.8f\\n", i, sequence_output[i]);
            return 1;
        }
        if (fabsf(step_output[i] - expected) > 1e-5f) {
            fprintf(stderr, "step_output[%d]=%.8f\\n", i, step_output[i]);
            return 2;
        }
    }
    return 0;
}
""".lstrip(),
                    encoding="utf-8",
                )
                executable = root / f"dynamic{output_dim}.exe"
                compile_result = subprocess.run(
                    [
                        "gcc",
                        "-O2",
                        "-std=c11",
                        str(driver),
                        str(root / "cfc_inference.c"),
                        "-lm",
                        "-o",
                        str(executable),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
                run_result = subprocess.run(
                    [str(executable)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(run_result.returncode, 0, run_result.stderr)


if __name__ == "__main__":
    unittest.main()
