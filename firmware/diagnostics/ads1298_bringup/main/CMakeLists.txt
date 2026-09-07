idf_component_register(
    SRCS ads1298_bringup.c ads1298_protocol.c
    INCLUDE_DIRS .
    REQUIRES esp_driver_gpio esp_driver_spi esp_driver_usb_serial_jtag freertos
)
