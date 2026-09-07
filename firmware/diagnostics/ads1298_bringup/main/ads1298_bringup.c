// ADS1298 eight-channel digital bring-up for ESP32-S3.
// This is a standalone diagnostic image. It deliberately contains no DSP,
// model inference, legacy ADC, or analogue filtering code.

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "driver/usb_serial_jtag.h"
#include "ads1298_protocol.h"
#include "esp_err.h"
#include "esp_rom_sys.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

// Default board-level wiring. Confirm ESP board silk-screen before connecting.
// These macros are intentionally the only place the ESP pin assignment lives.
#define ADS_DRDY_GPIO GPIO_NUM_4
#define ADS_CS_GPIO GPIO_NUM_5
#define ADS_START_GPIO GPIO_NUM_6
#define ADS_SCLK_GPIO GPIO_NUM_7
#define ADS_DOUT_GPIO GPIO_NUM_8
#define ADS_DIN_GPIO GPIO_NUM_9
#define ADS_RESET_GPIO GPIO_NUM_1
#define ADS_PWDN_GPIO GPIO_NUM_2

#define ADS_SPI_CLOCK_HZ 1000000
#define ADS_FRAME_BYTES 27
#define ADS_SETTLING_FRAMES 3
#define ADS_TPOR_MS 150
#define ADS_REFERENCE_SETTLE_MS 150
#define ADS_RECONFIG_STOP_SETTLE_MS 3
#define ADS_ACQUISITION_STACK_BYTES 4096
#define ADS_USB_STACK_BYTES 4096
#define ADS_ACQUISITION_PRIORITY (configMAX_PRIORITIES - 2)
#define ADS_USB_PRIORITY (configMAX_PRIORITIES - 5)
#define ADS_CONTROL_QUEUE_DEPTH 8
#define ADS_TX_QUEUE_DEPTH 256
#define ADS_USB_RX_BUFFER_BYTES 256
#define ADS_USB_TX_BUFFER_BYTES 16384
#define ADS_USB_READ_BYTES 64
#define ADS_USB_WRITE_TIMEOUT_MS 20
// The project uses a 100-Hz FreeRTOS tick, so a 2-ms timeout truncates to
// zero. Yield one whole tick after a bounded output burst so IDLE0 can reset
// the task watchdog. Four full seven-scan packets per tick sustain 2.8 kSPS.
#define ADS_USB_IDLE_WAIT_TICKS 1U
#define ADS_USB_MAX_PACKETS_PER_CYCLE 4U

// tCLK = 1 / 2.048 MHz = about 0.488 us. These rounded-up delays meet the
// ADS1298 requirements: RESET low >= 2*tCLK and recovery >= 18*tCLK.
#define ADS_RESET_LOW_US 2
#define ADS_RESET_RECOVERY_US 9
// tSDECODE >= 4*tCLK = about 1.96 us. At 1 MHz each command byte is already
// 8 us; this extra 3-us gap keeps the requirement explicit between commands.
#define ADS_TSDECODE_DELAY_US 3

#define ADS_CMD_RDATAC 0x10
#define ADS_CMD_SDATAC 0x11
#define ADS_CMD_RREG 0x20
#define ADS_CMD_WREG 0x40

#define ADS_REG_ID 0x00
#define ADS_REG_CONFIG1 0x01
#define ADS_REG_CONFIG2 0x02
#define ADS_REG_CONFIG3 0x03
#define ADS_REG_CH1SET 0x05

#define ADS_CONFIG1_2KSPS_HR 0x84
#define ADS_CONFIG2_ELECTRODE_INPUT 0x00
#define ADS_CONFIG3_INTERNAL_REFERENCE 0xC0
#define ADS_CHSET_NORMAL_ELECTRODE 0x00

typedef struct {
    uint8_t rate_index;
    uint8_t pga_index;
    uint8_t config1;
    uint8_t channel_set;
    uint8_t pga_gain;
} ads_sampling_config_t;

typedef enum {
    ADS_CONTROL_START_STREAM,
    ADS_CONTROL_STOP_STREAM,
    ADS_CONTROL_SET_PARAMETERS,
    ADS_CONTROL_REPORT_PARAMETERS,
} ads_control_kind_t;

typedef struct {
    ads_control_kind_t kind;
    ads_sampling_config_t sampling;
} ads_control_message_t;

typedef enum {
    ADS_TX_SAMPLE,
    ADS_TX_SAMPLE_PARAMETERS,
} ads_tx_kind_t;

typedef struct {
    ads_tx_kind_t kind;
    ads_sampling_config_t sampling;
    ads1298_frame_t frame;
} ads_tx_message_t;

typedef struct {
    uint32_t complete_frames;
    uint32_t discarded_settling_frames;
    uint32_t missed_before_read;
    uint32_t failed_frame_attempts;
    uint32_t spi_errors;
    uint32_t host_queue_drops;
    ads1298_frame_t last_frame;
    bool have_frame;
} ads_counters_t;

static spi_device_handle_t s_ads_device;
static TaskHandle_t s_acquisition_task;
static QueueHandle_t s_control_queue;
static QueueHandle_t s_tx_queue;
static volatile uint32_t s_drdy_events;
static volatile bool s_streaming;
static portMUX_TYPE s_drdy_lock = portMUX_INITIALIZER_UNLOCKED;
static ads_counters_t s_counters;
static portMUX_TYPE s_counters_lock = portMUX_INITIALIZER_UNLOCKED;
static ads_sampling_config_t s_sampling = {
    .rate_index = 2U,
    .pga_index = 4U,
    .config1 = ADS_CONFIG1_2KSPS_HR,
    .channel_set = ADS_CHSET_NORMAL_ELECTRODE,
    .pga_gain = 6U,
};
static uint32_t s_settling_frames_remaining = ADS_SETTLING_FRAMES;

static void IRAM_ATTR ads_drdy_isr(void *arg)
{
    BaseType_t higher_priority_task_woken = pdFALSE;
    (void)arg;

    portENTER_CRITICAL_ISR(&s_drdy_lock);
    s_drdy_events++;
    portEXIT_CRITICAL_ISR(&s_drdy_lock);
    vTaskNotifyGiveFromISR(s_acquisition_task, &higher_priority_task_woken);
    if (higher_priority_task_woken == pdTRUE) {
        portYIELD_FROM_ISR();
    }
}

static void ads_delay_at_least_ms(uint32_t milliseconds)
{
    // pdMS_TO_TICKS truncates. One extra tick ensures this is never shorter
    // than the requested datasheet interval for the configured tick rate.
    vTaskDelay(pdMS_TO_TICKS(milliseconds) + 1U);
}

static esp_err_t ads_transfer(const uint8_t *tx, uint8_t *rx, size_t bytes)
{
    spi_transaction_t transaction = {
        .length = bytes * 8U,
        .tx_buffer = tx,
        .rx_buffer = rx,
    };
    return spi_device_polling_transmit(s_ads_device, &transaction);
}

static esp_err_t ads_send_command(uint8_t command)
{
    esp_err_t err = ads_transfer(&command, NULL, 1);
    esp_rom_delay_us(ADS_TSDECODE_DELAY_US);
    return err;
}

static esp_err_t ads_write_registers(uint8_t address, const uint8_t *values, size_t count)
{
    if (count == 0 || count > ADS1298_CHANNEL_COUNT) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t tx[2 + ADS1298_CHANNEL_COUNT] = {0};
    tx[0] = ADS_CMD_WREG | address;
    tx[1] = (uint8_t)(count - 1U);
    for (size_t index = 0; index < count; index++) {
        tx[2 + index] = values[index];
    }

    esp_err_t err = ads_transfer(tx, NULL, count + 2U);
    esp_rom_delay_us(ADS_TSDECODE_DELAY_US);
    return err;
}

static esp_err_t ads_read_registers(uint8_t address, uint8_t *values, size_t count)
{
    if (count == 0 || count > ADS1298_CHANNEL_COUNT) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t tx[2 + ADS1298_CHANNEL_COUNT] = {0};
    uint8_t rx[2 + ADS1298_CHANNEL_COUNT] = {0};
    tx[0] = ADS_CMD_RREG | address;
    tx[1] = (uint8_t)(count - 1U);

    esp_err_t err = ads_transfer(tx, rx, count + 2U);
    esp_rom_delay_us(ADS_TSDECODE_DELAY_US);
    if (err == ESP_OK) {
        for (size_t index = 0; index < count; index++) {
            values[index] = rx[2 + index];
        }
    }
    return err;
}

static void ads_expect_register(uint8_t address, uint8_t expected, const char *name)
{
    uint8_t actual = 0;
    ESP_ERROR_CHECK(ads_read_registers(address, &actual, 1));
    printf("ADS %s readback: 0x%02X\n", name, actual);
    if (actual != expected) {
        printf("ADS %s mismatch: expected 0x%02X, got 0x%02X\n", name, expected, actual);
        ESP_ERROR_CHECK(ESP_ERR_INVALID_RESPONSE);
    }
}

static void ads_configure_control_pins(void)
{
    const uint64_t output_mask = (1ULL << ADS_CS_GPIO) |
                                 (1ULL << ADS_START_GPIO) |
                                 (1ULL << ADS_RESET_GPIO) |
                                 (1ULL << ADS_PWDN_GPIO);
    const gpio_config_t output_config = {
        .pin_bit_mask = output_mask,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };

    ESP_ERROR_CHECK(gpio_config(&output_config));
    ESP_ERROR_CHECK(gpio_set_level(ADS_CS_GPIO, 1));
    ESP_ERROR_CHECK(gpio_set_level(ADS_START_GPIO, 0));
    ESP_ERROR_CHECK(gpio_set_level(ADS_PWDN_GPIO, 0));
    ESP_ERROR_CHECK(gpio_set_level(ADS_RESET_GPIO, 0));
}

static void ads_power_on_and_reset(void)
{
    // The app has already placed all controls in their safe states. Once the
    // module rail and 2.048-MHz oscillator are present, release PWDN/RESET,
    // satisfy tPOR, then create the specified hardware reset pulse.
    ESP_ERROR_CHECK(gpio_set_level(ADS_PWDN_GPIO, 1));
    ESP_ERROR_CHECK(gpio_set_level(ADS_RESET_GPIO, 1));
    ads_delay_at_least_ms(ADS_TPOR_MS);
    ESP_ERROR_CHECK(gpio_set_level(ADS_RESET_GPIO, 0));
    esp_rom_delay_us(ADS_RESET_LOW_US);
    ESP_ERROR_CHECK(gpio_set_level(ADS_RESET_GPIO, 1));
    esp_rom_delay_us(ADS_RESET_RECOVERY_US);
}

static void ads_initialize_spi(void)
{
    const spi_bus_config_t bus_config = {
        .mosi_io_num = ADS_DIN_GPIO,
        .miso_io_num = ADS_DOUT_GPIO,
        .sclk_io_num = ADS_SCLK_GPIO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .data4_io_num = -1,
        .data5_io_num = -1,
        .data6_io_num = -1,
        .data7_io_num = -1,
    };
    const spi_device_interface_config_t device_config = {
        .clock_speed_hz = ADS_SPI_CLOCK_HZ,
        .mode = 1,
        .spics_io_num = ADS_CS_GPIO,
        .queue_size = 1,
    };

    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &bus_config, SPI_DMA_DISABLED));
    ESP_ERROR_CHECK(spi_bus_add_device(SPI2_HOST, &device_config, &s_ads_device));
}

static void ads_install_drdy_isr(void)
{
    const gpio_config_t drdy_config = {
        .pin_bit_mask = 1ULL << ADS_DRDY_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE,
    };

    ESP_ERROR_CHECK(gpio_config(&drdy_config));
    ESP_ERROR_CHECK(gpio_install_isr_service(ESP_INTR_FLAG_IRAM));
    ESP_ERROR_CHECK(gpio_isr_handler_add(ADS_DRDY_GPIO, ads_drdy_isr, NULL));
}

static bool ads_sampling_config_from_pc(uint8_t rate_index, uint8_t pga_index,
                                        ads_sampling_config_t *sampling)
{
    static const uint8_t config1_by_rate[] = {0x86U, 0x85U, 0x84U};
    static const uint8_t channel_set_by_pga[] = {
        0x10U, 0x20U, 0x30U, 0x40U, 0x00U, 0x50U, 0x60U,
    };
    static const uint8_t pga_gain_by_index[] = {1U, 2U, 3U, 4U, 6U, 8U, 12U};

    // A 27-byte frame consumes 216 us at the conservative 1-MHz SPI clock.
    // 4 kSPS leaves only 34 us per period for ISR/RTOS work, so it is not
    // exposed until a hardware zero-drop run proves a faster transport path.
    if (sampling == NULL || rate_index > 2U || rate_index >= sizeof(config1_by_rate) ||
        pga_index >= sizeof(channel_set_by_pga)) {
        return false;
    }

    *sampling = (ads_sampling_config_t){
        .rate_index = rate_index,
        .pga_index = pga_index,
        .config1 = config1_by_rate[rate_index],
        .channel_set = channel_set_by_pga[pga_index],
        .pga_gain = pga_gain_by_index[pga_index],
    };
    return true;
}

static void ads_configure_electrode_inputs(void)
{
    const uint8_t config1 = s_sampling.config1;
    const uint8_t config2 = ADS_CONFIG2_ELECTRODE_INPUT;
    const uint8_t config3 = ADS_CONFIG3_INTERNAL_REFERENCE;
    const uint8_t channel_config[ADS1298_CHANNEL_COUNT] = {
        s_sampling.channel_set, s_sampling.channel_set,
        s_sampling.channel_set, s_sampling.channel_set,
        s_sampling.channel_set, s_sampling.channel_set,
        s_sampling.channel_set, s_sampling.channel_set,
    };
    uint8_t id = 0;

    ESP_ERROR_CHECK(ads_send_command(ADS_CMD_SDATAC));
    ESP_ERROR_CHECK(ads_read_registers(ADS_REG_ID, &id, 1));
    printf("ADS1298 ID: 0x%02X (device field: 0x%02X)\n", id, id & 0x1FU);

    ESP_ERROR_CHECK(ads_write_registers(ADS_REG_CONFIG1, &config1, 1));
    ESP_ERROR_CHECK(ads_write_registers(ADS_REG_CONFIG3, &config3, 1));
    ads_delay_at_least_ms(ADS_REFERENCE_SETTLE_MS);
    ESP_ERROR_CHECK(ads_write_registers(ADS_REG_CONFIG2, &config2, 1));
    ESP_ERROR_CHECK(ads_write_registers(ADS_REG_CH1SET, channel_config,
                                        ADS1298_CHANNEL_COUNT));

    ads_expect_register(ADS_REG_CONFIG1, config1, "CONFIG1");
    ads_expect_register(ADS_REG_CONFIG2, config2, "CONFIG2");
    ads_expect_register(ADS_REG_CONFIG3, config3, "CONFIG3");
    for (uint8_t channel = 0; channel < ADS1298_CHANNEL_COUNT; channel++) {
        ads_expect_register(ADS_REG_CH1SET + channel, s_sampling.channel_set, "CHnSET");
    }
}

static esp_err_t ads_apply_sampling_config(const ads_sampling_config_t *sampling)
{
    uint8_t channel_config[ADS1298_CHANNEL_COUNT];
    esp_err_t result = ESP_OK;

    if (sampling == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    for (size_t channel = 0; channel < ADS1298_CHANNEL_COUNT; channel++) {
        channel_config[channel] = sampling->channel_set;
    }

    // Stop first and wait longer than the slowest supported 500-SPS period.
    // Clearing notifications only after that wait prevents an old DRDY event
    // from becoming the first supposedly-new configuration frame.
    ESP_ERROR_CHECK(gpio_set_level(ADS_START_GPIO, 0));
    ads_delay_at_least_ms(ADS_RECONFIG_STOP_SETTLE_MS);
    ESP_ERROR_CHECK(gpio_intr_disable(ADS_DRDY_GPIO));
    (void)ulTaskNotifyTake(pdTRUE, 0U);

    result = ads_send_command(ADS_CMD_SDATAC);
    if (result == ESP_OK) {
        result = ads_write_registers(ADS_REG_CONFIG1, &sampling->config1, 1U);
    }
    if (result == ESP_OK) {
        result = ads_write_registers(ADS_REG_CH1SET, channel_config, ADS1298_CHANNEL_COUNT);
    }
    const esp_err_t restart_result = ads_send_command(ADS_CMD_RDATAC);
    if (result == ESP_OK && restart_result != ESP_OK) {
        result = restart_result;
    }
    if (result == ESP_OK) {
        s_sampling = *sampling;
        s_settling_frames_remaining = ADS_SETTLING_FRAMES;
    }
    ESP_ERROR_CHECK(gpio_intr_enable(ADS_DRDY_GPIO));
    ESP_ERROR_CHECK(gpio_set_level(ADS_START_GPIO, 1));
    return result;
}

static void ads_record_host_queue_drop(void)
{
    portENTER_CRITICAL(&s_counters_lock);
    s_counters.host_queue_drops++;
    portEXIT_CRITICAL(&s_counters_lock);
}

static void ads_handle_control_messages(void)
{
    ads_control_message_t control;

    while (xQueueReceive(s_control_queue, &control, 0U) == pdTRUE) {
        if (control.kind == ADS_CONTROL_START_STREAM) {
            s_streaming = true;
            continue;
        }
        if (control.kind == ADS_CONTROL_STOP_STREAM) {
            s_streaming = false;
            continue;
        }
        if (control.kind == ADS_CONTROL_SET_PARAMETERS) {
            ads_tx_message_t reply = {
                .kind = ADS_TX_SAMPLE_PARAMETERS,
                .sampling = control.sampling,
            };

            // The PC sends parameters before START. Stop output first so a
            // parameter acknowledgment always precedes samples using it.
            s_streaming = false;
            if (ads_apply_sampling_config(&control.sampling) == ESP_OK) {
                if (xQueueSend(s_tx_queue, &reply, 0U) != pdTRUE) {
                    ads_record_host_queue_drop();
                }
            }
            continue;
        }
        if (control.kind == ADS_CONTROL_REPORT_PARAMETERS) {
            const ads_tx_message_t reply = {
                .kind = ADS_TX_SAMPLE_PARAMETERS,
                .sampling = s_sampling,
            };
            if (xQueueSend(s_tx_queue, &reply, 0U) != pdTRUE) {
                ads_record_host_queue_drop();
            }
        }
    }
}

static void ads_usb_write_packet(const uint8_t *packet, size_t packet_length)
{
    size_t written = 0U;

    while (written < packet_length) {
        const int result = usb_serial_jtag_write_bytes(
            &packet[written], packet_length - written, pdMS_TO_TICKS(ADS_USB_WRITE_TIMEOUT_MS));
        if (result <= 0) {
            ads_record_host_queue_drop();
            return;
        }
        written += (size_t)result;
    }
}

static bool ads_sampling_configs_match(const ads_sampling_config_t *left,
                                       const ads_sampling_config_t *right)
{
    return left->rate_index == right->rate_index &&
           left->pga_index == right->pga_index &&
           left->pga_gain == right->pga_gain;
}

static void ads_usb_send_samples(const ads_tx_message_t *first)
{
    ads1298_frame_t samples[ADS1298_PC_MAX_SAMPLES_PER_PACKET];
    uint8_t packet[ADS1298_PC_MAX_EMG_PACKET_BYTES];
    size_t sample_count = 1U;

    samples[0] = first->frame;
    while (sample_count < ADS1298_PC_MAX_SAMPLES_PER_PACKET) {
        ads_tx_message_t next;
        // The USB task batches messages already queued during its one-tick
        // yield; it must not wait here and risk starving the idle task again.
        if (xQueuePeek(s_tx_queue, &next, 0U) != pdTRUE ||
            next.kind != ADS_TX_SAMPLE ||
            !ads_sampling_configs_match(&first->sampling, &next.sampling)) {
            break;
        }
        if (xQueueReceive(s_tx_queue, &next, 0U) != pdTRUE) {
            break;
        }
        samples[sample_count++] = next.frame;
    }

    const size_t packet_length = ads1298_pack_emg_frame(
        samples, sample_count, first->sampling.pga_gain, packet, sizeof(packet));
    if (packet_length == 0U) {
        ads_record_host_queue_drop();
        return;
    }
    ads_usb_write_packet(packet, packet_length);
}

static void ads_submit_pc_command(const ads1298_pc_command_t *command)
{
    ads_control_message_t control;

    if (command->address == ADS1298_PC_COMMAND_START && command->data_length == 1U) {
        control.kind = command->data[0] == 0U ? ADS_CONTROL_STOP_STREAM : ADS_CONTROL_START_STREAM;
    } else if (command->address == ADS1298_PC_COMMAND_CONNECTION_STATUS &&
               command->data_length == 1U) {
        control.kind = ADS_CONTROL_REPORT_PARAMETERS;
    } else if (command->address == ADS1298_PC_COMMAND_SAMPLE_PARAMETERS &&
               command->data_length >= 2U &&
               ads_sampling_config_from_pc(command->data[0], command->data[1],
                                           &control.sampling)) {
        control.kind = ADS_CONTROL_SET_PARAMETERS;
    } else {
        return;
    }

    if (xQueueSend(s_control_queue, &control, 0U) != pdTRUE) {
        ads_record_host_queue_drop();
    }
}

static void ads_usb_task(void *arg)
{
    ads1298_pc_command_parser_t parser;
    uint8_t received[ADS_USB_READ_BYTES];
    (void)arg;

    ads1298_pc_command_parser_init(&parser);
    while (true) {
        const int received_bytes = usb_serial_jtag_read_bytes(
            received, sizeof(received), 0U);
        for (int index = 0; index < received_bytes; index++) {
            ads1298_pc_command_t command;
            if (ads1298_pc_command_parser_consume(&parser, received[index], &command)) {
                ads_submit_pc_command(&command);
            }
        }

        for (uint32_t packet_count = 0U;
             packet_count < ADS_USB_MAX_PACKETS_PER_CYCLE;
             packet_count++) {
            ads_tx_message_t message;
            if (xQueueReceive(s_tx_queue, &message, 0U) != pdTRUE) {
                break;
            }
            if (message.kind == ADS_TX_SAMPLE_PARAMETERS) {
                uint8_t packet[ADS1298_PC_SAMPLE_PARAMETERS_PACKET_BYTES];
                const size_t packet_length = ads1298_pack_sample_parameters(
                    message.sampling.rate_index, message.sampling.pga_index, packet, sizeof(packet));
                if (packet_length == 0U) {
                    ads_record_host_queue_drop();
                } else {
                    ads_usb_write_packet(packet, packet_length);
                }
            } else {
                ads_usb_send_samples(&message);
            }
        }

        vTaskDelay(ADS_USB_IDLE_WAIT_TICKS);
    }
}

static void ads_acquisition_task(void *arg)
{
    uint8_t tx[ADS_FRAME_BYTES] = {0};
    uint8_t rx[ADS_FRAME_BYTES] = {0};
    (void)arg;

    while (true) {
        ads_handle_control_messages();
        const uint32_t notifications = ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (notifications > 1U) {
            portENTER_CRITICAL(&s_counters_lock);
            s_counters.missed_before_read += notifications - 1U;
            portEXIT_CRITICAL(&s_counters_lock);
        }

        const esp_err_t err = ads_transfer(tx, rx, ADS_FRAME_BYTES);
        if (err != ESP_OK) {
            portENTER_CRITICAL(&s_counters_lock);
            s_counters.failed_frame_attempts++;
            s_counters.spi_errors++;
            portEXIT_CRITICAL(&s_counters_lock);
            continue;
        }

        ads1298_frame_t decoded;
        if (!ads1298_decode_frame(rx, &decoded)) {
            portENTER_CRITICAL(&s_counters_lock);
            s_counters.failed_frame_attempts++;
            portEXIT_CRITICAL(&s_counters_lock);
            continue;
        }

        portENTER_CRITICAL(&s_counters_lock);
        if (s_settling_frames_remaining > 0U) {
            s_counters.discarded_settling_frames++;
            s_settling_frames_remaining--;
        } else {
            s_counters.complete_frames++;
            s_counters.last_frame = decoded;
            s_counters.have_frame = true;
        }
        portEXIT_CRITICAL(&s_counters_lock);

        if (s_streaming && s_settling_frames_remaining == 0U) {
            const ads_tx_message_t message = {
                .kind = ADS_TX_SAMPLE,
                .sampling = s_sampling,
                .frame = decoded,
            };
            if (xQueueSend(s_tx_queue, &message, 0U) != pdTRUE) {
                ads_record_host_queue_drop();
            }
        }
    }
}

void app_main(void)
{
    printf("\n=== ADS1298 8-channel SPI/DRDY bring-up ===\n");
    printf("Default wiring (verify board silk-screen): DRDY=%d CS=%d START=%d SCLK=%d "
           "DOUT/MISO=%d DIN/MOSI=%d RESET=%d PWDN=%d\n",
           ADS_DRDY_GPIO, ADS_CS_GPIO, ADS_START_GPIO, ADS_SCLK_GPIO,
           ADS_DOUT_GPIO, ADS_DIN_GPIO, ADS_RESET_GPIO, ADS_PWDN_GPIO);
    printf("Normal differential electrode inputs selected; ADS test source is disabled.\n");

    if (!ads1298_protocol_self_test()) {
        printf("ADS signed-24 protocol self-test failed; acquisition will not start.\n");
        ESP_ERROR_CHECK(ESP_FAIL);
    }

    ads_configure_control_pins();
    ads_power_on_and_reset();
    ads_initialize_spi();
    ads_configure_electrode_inputs();

    s_control_queue = xQueueCreate(ADS_CONTROL_QUEUE_DEPTH, sizeof(ads_control_message_t));
    s_tx_queue = xQueueCreate(ADS_TX_QUEUE_DEPTH, sizeof(ads_tx_message_t));
    if (s_control_queue == NULL || s_tx_queue == NULL) {
        ESP_ERROR_CHECK(ESP_ERR_NO_MEM);
    }

    // Install the binary USB endpoint only after the human-readable boot
    // checks above. No printf/ESP_LOG output is permitted after this point.
    usb_serial_jtag_driver_config_t usb_config = {
        .rx_buffer_size = ADS_USB_RX_BUFFER_BYTES,
        .tx_buffer_size = ADS_USB_TX_BUFFER_BYTES,
    };
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_config));

    // The high-priority task is alive before DRDY is enabled. It owns all ADS
    // SPI I/O; the lower-priority USB task only dequeues already-decoded data.
    if (xTaskCreate(ads_acquisition_task, "ads_acquire", ADS_ACQUISITION_STACK_BYTES,
                    NULL, ADS_ACQUISITION_PRIORITY, &s_acquisition_task) != pdPASS) {
        ESP_ERROR_CHECK(ESP_ERR_NO_MEM);
    }
    if (xTaskCreate(ads_usb_task, "ads_usb", ADS_USB_STACK_BYTES, NULL,
                    ADS_USB_PRIORITY, NULL) != pdPASS) {
        ESP_ERROR_CHECK(ESP_ERR_NO_MEM);
    }
    ads_install_drdy_isr();
    ESP_ERROR_CHECK(ads_send_command(ADS_CMD_RDATAC));
    ESP_ERROR_CHECK(gpio_set_level(ADS_START_GPIO, 1));

    while (true) {
        vTaskDelay(portMAX_DELAY);
    }
}
