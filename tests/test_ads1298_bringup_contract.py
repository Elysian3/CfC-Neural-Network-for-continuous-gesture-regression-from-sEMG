import unittest
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRINGUP_ROOT = PROJECT_ROOT / "firmware" / "diagnostics" / "ads1298_bringup"
MAIN_ROOT = BRINGUP_ROOT / "main"
BRINGUP_SOURCE = MAIN_ROOT / "ads1298_bringup.c"
PROTOCOL_SOURCE = MAIN_ROOT / "ads1298_protocol.c"
PROTOCOL_HEADER = MAIN_ROOT / "ads1298_protocol.h"
HOST_PROTOCOL_TEST = PROJECT_ROOT / "tests" / "test_ads1298_pc_protocol_host.c"


class Ads1298BringupContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(BRINGUP_SOURCE.is_file(), "ADS1298 bring-up source is missing")
        self.source = BRINGUP_SOURCE.read_text(encoding="utf-8")

    def test_is_a_separate_esp_idf_diagnostic_project(self):
        """Bring-up must be flashable without compiling the legacy MUX/ADC firmware."""
        project_cmake = BRINGUP_ROOT / "CMakeLists.txt"
        component_cmake = MAIN_ROOT / "CMakeLists.txt"

        self.assertTrue(project_cmake.is_file())
        self.assertTrue(component_cmake.is_file())
        self.assertIn("project(ads1298_bringup)", project_cmake.read_text(encoding="utf-8"))

        component = component_cmake.read_text(encoding="utf-8")
        self.assertIn("ads1298_bringup.c", component)
        self.assertIn("ads1298_protocol.c", component)
        self.assertIn("esp_driver_spi", component)
        self.assertIn("esp_driver_gpio", component)
        self.assertNotIn("firmware/main/main.c", component)

    def test_main_includes_its_protocol_contract(self):
        """The frame type and decoder declarations must be visible to main.c."""
        self.assertIn('#include "ads1298_protocol.h"', self.source)

    def test_exact_default_ads_to_esp_pin_map_is_visible_in_source(self):
        """The first hardware run uses the reviewed, editable default wiring map."""
        expected_macros = {
            "ADS_DRDY_GPIO": "GPIO_NUM_4",
            "ADS_CS_GPIO": "GPIO_NUM_5",
            "ADS_START_GPIO": "GPIO_NUM_6",
            "ADS_SCLK_GPIO": "GPIO_NUM_7",
            "ADS_DOUT_GPIO": "GPIO_NUM_8",
            "ADS_DIN_GPIO": "GPIO_NUM_9",
            "ADS_RESET_GPIO": "GPIO_NUM_1",
            "ADS_PWDN_GPIO": "GPIO_NUM_2",
        }
        for macro, value in expected_macros.items():
            self.assertRegex(self.source, rf"#define\s+{macro}\s+{value}")

    def test_spi_and_drdy_contract_prevent_partial_or_stale_frames(self):
        """DRDY owns acquisition cadence; every normal ADS1298 frame is 27 bytes."""
        self.assertIn("SPI2_HOST", self.source)
        self.assertRegex(self.source, r"\.mode\s*=\s*1")
        self.assertIn("GPIO_INTR_NEGEDGE", self.source)
        self.assertIn("vTaskNotifyGiveFromISR", self.source)
        self.assertIn("spi_device_polling_transmit", self.source)
        self.assertRegex(PROTOCOL_HEADER.read_text(encoding="utf-8"), r"#define\s+ADS_FRAME_BYTES\s+27")
        self.assertRegex(self.source, r"#define\s+ADS_SETTLING_FRAMES\s+3")
        self.assertIn("ADS_CMD_SDATAC", self.source)
        self.assertIn("ADS_CMD_RDATAC", self.source)
        self.assertIn("ADS_CONFIG1_2KSPS_HR", self.source)
        self.assertIn("0x84", self.source)
        self.assertIn("0xC0", self.source)
        self.assertIn("ADS_CONFIG2_ELECTRODE_INPUT", self.source)
        self.assertIn("ADS_CHSET_NORMAL_ELECTRODE", self.source)
        self.assertRegex(self.source, r"#define\s+ADS_CONFIG2_ELECTRODE_INPUT\s+0x00")
        self.assertRegex(self.source, r"#define\s+ADS_CHSET_NORMAL_ELECTRODE\s+0x00")
        self.assertNotIn("ADS_CONFIG2_INTERNAL_TEST", self.source)
        self.assertNotIn("ADS_CHSET_INTERNAL_TEST", self.source)
        self.assertNotIn("ads_configure_internal_test_source", self.source)
        # ESP-IDF v6 exposes octal data fields too. Leaving them at their C
        # zero default can route unrelated SPI data pins to GPIO0.
        for field in ("data4_io_num", "data5_io_num", "data6_io_num", "data7_io_num"):
            self.assertRegex(self.source, rf"\.{field}\s*=\s*-1")

    def test_timing_and_task_partition_preserve_2ksps_acquisition(self):
        """RTOS timing and USB handoff must not run inside the SPI acquisition path."""
        self.assertRegex(self.source, r"#define\s+ADS_TPOR_MS\s+150")
        self.assertIn("ads_delay_at_least_ms", self.source)
        self.assertIn("ads_acquisition_task", self.source)
        self.assertIn("xTaskCreate", self.source)
        self.assertIn("ADS_ACQUISITION_PRIORITY", self.source)
        self.assertIn("ADS_USB_PRIORITY", self.source)
        self.assertIn("ADS_TX_QUEUE_DEPTH", self.source)

    def test_frame_decoder_is_explicit_about_24_bit_twos_complement(self):
        """Known edge values prevent silently interpreting ADS samples as unsigned."""
        self.assertTrue(PROTOCOL_SOURCE.is_file())
        self.assertTrue(PROTOCOL_HEADER.is_file())
        protocol = PROTOCOL_SOURCE.read_text(encoding="utf-8")
        header = PROTOCOL_HEADER.read_text(encoding="utf-8")

        self.assertIn("ads1298_decode_signed24", header)
        self.assertIn("ads1298_decode_signed24", protocol)
        self.assertIn("0x800000", protocol)
        self.assertIn("0xFF000000", protocol)
        self.assertIn("ADS_FRAME_BYTES", protocol)
        self.assertIn("status", protocol)
        self.assertIn("channels[8]", header)

    def test_firmware_runs_known_signed24_boundary_cases_before_sampling(self):
        """The target must reject a broken decoder even when no ADS is connected."""
        protocol = PROTOCOL_SOURCE.read_text(encoding="utf-8")
        header = PROTOCOL_HEADER.read_text(encoding="utf-8")

        self.assertIn("ads1298_protocol_self_test", header)
        self.assertIn("ads1298_protocol_self_test", protocol)
        for value in ("0x000000", "0x7FFFFF", "0x800000", "0xFFFFFF"):
            self.assertIn(value, protocol)

    def test_diagnostic_isolated_from_legacy_mux_and_adc_capture(self):
        """The ADS image must not configure the legacy MUX/ADC acquisition path."""
        forbidden = ("esp_adc", "adc_continuous", "74HC4051", "mux_select")
        for token in forbidden:
            self.assertNotIn(token.lower(), self.source.lower())

    def test_event_accounting_and_bounded_observability_are_not_optional(self):
        """Losses remain counted even though waveform USB cannot contain diagnostic text."""
        for counter in (
            "drdy_events",
            "complete_frames",
            "discarded_settling_frames",
            "missed_before_read",
            "failed_frame_attempts",
            "spi_errors",
            "host_queue_drops",
        ):
            self.assertIn(counter, self.source)
        self.assertIn("ads_record_host_queue_drop", self.source)
        self.assertIn("ADS_TX_QUEUE_DEPTH", self.source)
        self.assertNotIn("ADS_REPORT_PERIOD_MS", self.source)

    def test_pc_binary_protocol_round_trips_on_the_host(self):
        """The exact C bytes expected by the supplied PC program stay executable."""
        self.assertTrue(HOST_PROTOCOL_TEST.is_file())
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = Path(temp_dir) / "ads1298_pc_protocol_test.exe"
            compile_result = subprocess.run(
                [
                    "gcc", "-std=c11", "-Wall", "-Werror", "-I", str(MAIN_ROOT),
                    str(PROTOCOL_SOURCE), str(HOST_PROTOCOL_TEST), "-o", str(executable),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run_result = subprocess.run([str(executable)], text=True, capture_output=True,
                                        check=False)
            self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_pc_streaming_is_binary_command_controlled_and_decoupled_from_drdy(self):
        """USB output must not run in the DRDY/SPI task or leak diagnostic text into waves."""
        component = (MAIN_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

        self.assertIn("esp_driver_usb_serial_jtag", component)
        for token in (
            "usb_serial_jtag_driver_install",
            "usb_serial_jtag_read_bytes",
            "usb_serial_jtag_write_bytes",
            "ads1298_pc_command_parser_consume",
            "ads1298_pack_emg_frame",
            "ads1298_pack_sample_parameters",
            "ADS1298_PC_COMMAND_CONNECTION_STATUS",
            "xQueueCreate",
            "s_streaming",
            "ads_usb_task",
        ):
            self.assertIn(token, self.source)

        acquisition_body = self.source.split("static void ads_acquisition_task", 1)[1].split(
            "void app_main", 1
        )[0]
        self.assertNotIn("usb_serial_jtag_write_bytes", acquisition_body)
        self.assertNotIn("printf(", acquisition_body)

    def test_usb_task_yields_at_the_100hz_rtos_tick_without_losing_2ksps_capacity(self):
        """At a 100-Hz tick, a 2-ms timeout becomes zero and would starve IDLE0."""
        usb_body = self.source.split("static void ads_usb_task", 1)[1].split(
            "static void ads_acquisition_task", 1
        )[0]

        # Four packets x seven scans per 10-ms tick = 2800 scans/s, above 2 kSPS.
        self.assertRegex(self.source, r"#define\s+ADS_USB_MAX_PACKETS_PER_CYCLE\s+4U")
        self.assertIn("packet_count < ADS_USB_MAX_PACKETS_PER_CYCLE", usb_body)
        self.assertIn("received, sizeof(received), 0U", usb_body)
        self.assertIn("vTaskDelay(ADS_USB_IDLE_WAIT_TICKS)", usb_body)
        self.assertNotIn("pdMS_TO_TICKS(2U)", usb_body)

    def test_pc_rate_selection_rejects_unverified_4ksps_and_reconfigures_without_stale_drdy(self):
        """At 1 MHz SPI, only the verified 500/1000/2000-SPS modes may start a stream."""
        reconfigure_start = self.source.index("static esp_err_t ads_apply_sampling_config")
        reconfigure_end = self.source.index("static void ads_record_host_queue_drop", reconfigure_start)
        reconfigure = self.source[reconfigure_start:reconfigure_end]

        self.assertIn("ADS_RECONFIG_STOP_SETTLE_MS", self.source)
        self.assertIn("rate_index > 2U", self.source)
        self.assertLess(
            reconfigure.index("gpio_set_level(ADS_START_GPIO, 0)"),
            reconfigure.index("gpio_intr_disable(ADS_DRDY_GPIO)"),
        )
        self.assertLess(
            reconfigure.index("gpio_intr_disable(ADS_DRDY_GPIO)"),
            reconfigure.index("ulTaskNotifyTake(pdTRUE, 0U)"),
        )
        self.assertLess(
            reconfigure.index("s_settling_frames_remaining = ADS_SETTLING_FRAMES"),
            reconfigure.index("gpio_set_level(ADS_START_GPIO, 1)"),
        )


if __name__ == "__main__":
    unittest.main()
