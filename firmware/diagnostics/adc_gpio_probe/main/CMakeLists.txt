idf_component_register(
    SRCS adc_gpio_probe.c
    INCLUDE_DIRS .
    REQUIRES esp_adc esp_driver_gpio
)
