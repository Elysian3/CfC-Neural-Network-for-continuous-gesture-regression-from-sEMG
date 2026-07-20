idf_component_register(
    SRCS main.c cfc_inference.c features.c
    INCLUDE_DIRS .
    REQUIRES freertos driver esp_adc esp_timer esp_driver_gpio espressif__esp-dsp
)
