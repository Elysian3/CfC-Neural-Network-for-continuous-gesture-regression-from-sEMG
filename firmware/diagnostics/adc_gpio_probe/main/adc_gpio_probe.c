// ADC-capable GPIO discovery probe for ESP32-S3.
// This is a standalone diagnostic project; it does not modify production firmware.

#include <stdio.h>

#include "esp_adc/adc_oneshot.h"
#include "esp_err.h"
#include "driver/gpio.h"
#include "soc/soc_caps.h"

// Proposed direct-input routing: eight channels on one ADC unit.
// GPIO3 is intentionally excluded because it is an ESP32-S3 strapping pin.
static const gpio_num_t CANDIDATE_GPIOS[] = {
    GPIO_NUM_1,
    GPIO_NUM_2,
    GPIO_NUM_4,
    GPIO_NUM_5,
    GPIO_NUM_6,
    GPIO_NUM_7,
    GPIO_NUM_8,
    GPIO_NUM_9,
};

#define CANDIDATE_GPIO_COUNT (sizeof(CANDIDATE_GPIOS) / sizeof(CANDIDATE_GPIOS[0]))

static const char *adc_unit_name(adc_unit_t unit)
{
    switch (unit) {
    case ADC_UNIT_1:
        return "ADC_UNIT_1";
    case ADC_UNIT_2:
        return "ADC_UNIT_2";
    default:
        return "ADC_UNIT_UNKNOWN";
    }
}

static uint64_t candidate_gpio_mask(void)
{
    uint64_t mask = 0;

    for (size_t index = 0; index < CANDIDATE_GPIO_COUNT; index++) {
        mask |= 1ULL << CANDIDATE_GPIOS[index];
    }
    return mask;
}

static void audit_candidate_gpio_occupancy(void)
{
    printf("\n=== Proposed direct-input GPIO audit (GPIO1,2,4--9) ===\n");
    printf("Reject any pin marked **RESERVED**.\n");
    printf("[periph_sig_ctrl] is an IOMUX function-selection label, not a conflict by itself.\n");
    printf("This audit is read-only: it does not reconfigure any GPIO.\n");

    for (size_t index = 0; index < CANDIDATE_GPIO_COUNT; index++) {
        int gpio = CANDIDATE_GPIOS[index];
        adc_unit_t unit;
        adc_channel_t channel;
        esp_err_t err = adc_oneshot_io_to_channel(gpio, &unit, &channel);

        if (err == ESP_OK) {
            printf("candidate GPIO %d -> %s_CH%d\n", gpio, adc_unit_name(unit), channel);
        } else {
            printf("candidate GPIO %d -> ADC lookup error: %s\n",
                   gpio, esp_err_to_name(err));
        }
    }

    ESP_ERROR_CHECK(gpio_dump_io_configuration(stdout, candidate_gpio_mask()));
}

void app_main(void)
{
    int adc_gpio_count = 0;

    printf("\n=== ESP32-S3 ADC-capable GPIO probe ===\n");
    printf("This lists chip ADC pads only; verify board routing before use.\n");

    audit_candidate_gpio_occupancy();

    for (int gpio = 0; gpio < SOC_GPIO_PIN_COUNT; gpio++) {
        if (!GPIO_IS_VALID_GPIO(gpio)) {
            continue;
        }

        adc_unit_t unit;
        adc_channel_t channel;
        esp_err_t err = adc_oneshot_io_to_channel(gpio, &unit, &channel);

        if (err == ESP_OK) {
            printf("GPIO %d -> %s_CH%d\n", gpio, adc_unit_name(unit), channel);
            adc_gpio_count++;
        } else if (err != ESP_ERR_NOT_FOUND) {
            printf("GPIO %d -> lookup error: %s\n", gpio, esp_err_to_name(err));
        }
    }

    printf("ADC-capable chip GPIO count: %d\n", adc_gpio_count);
    printf("Do not connect raw EMG electrodes directly to these pins.\n");
}
