import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CFC_SOURCE = PROJECT_ROOT / "firmware" / "main" / "cfc_inference.c"


class FirmwareWeightBackendTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
