import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = PROJECT_ROOT / "firmware" / "diagnostics" / "adc_gpio_probe"
PROBE_SOURCE = PROBE_ROOT / "main" / "adc_gpio_probe.c"


class AdcGpioProbeContractTests(unittest.TestCase):
    def test_probe_scans_every_chip_gpio_with_the_official_mapping_api(self):
        """The standalone probe must discover ADC pads at runtime, not guess a pin list."""
        self.assertTrue(PROBE_SOURCE.is_file())
        source = PROBE_SOURCE.read_text(encoding="utf-8")

        self.assertIn("adc_oneshot_io_to_channel", source)
        self.assertIn("SOC_GPIO_PIN_COUNT", source)
        self.assertRegex(
            source,
            r"for\s*\(\s*int\s+gpio\s*=\s*0\s*;\s*gpio\s*<\s*SOC_GPIO_PIN_COUNT",
        )

    def test_probe_audits_eight_non_strapping_direct_input_pins_without_reconfiguring_them(self):
        """GPIO3 is a strapping pin, so the proposed ADC1 set must skip it."""
        source = PROBE_SOURCE.read_text(encoding="utf-8")

        self.assertIn("CANDIDATE_GPIOS", source)
        self.assertIn("GPIO_NUM_1", source)
        self.assertIn("GPIO_NUM_9", source)
        self.assertNotIn("GPIO_NUM_3,", source)
        self.assertIn("gpio_dump_io_configuration", source)
        self.assertIn("GPIO_IS_VALID_GPIO", source)
        self.assertIn("function-selection label", source)
        self.assertNotIn("gpio_config(", source)

    def test_probe_is_a_standalone_esp_idf_project(self):
        """The diagnostic must not change the production CfC firmware entrypoint."""
        project_cmake = PROBE_ROOT / "CMakeLists.txt"
        component_cmake = PROBE_ROOT / "main" / "CMakeLists.txt"
        self.assertTrue(project_cmake.is_file())
        self.assertTrue(component_cmake.is_file())

        self.assertIn("project(adc_gpio_probe)", project_cmake.read_text(encoding="utf-8"))
        component_config = component_cmake.read_text(encoding="utf-8")
        self.assertIn("SRCS adc_gpio_probe.c", component_config)
        self.assertIn("REQUIRES esp_adc", component_config)
        self.assertIn("esp_driver_gpio", component_config)
        self.assertNotIn("firmware/main/main.c", component_config)


if __name__ == "__main__":
    unittest.main()
