# ESP32-S3 ADC GPIO probe

This standalone ESP-IDF project reports which **chip GPIO pads** map to ADC
units and channels. It does not prove that a pin is exposed on the ESP32-S3
Zero board or available after board-level peripheral conflicts are considered.

Run it from an ESP-IDF PowerShell with the board connected:

```powershell
Set-Location D:\Project Antikythera\firmware\diagnostics\adc_gpio_probe
idf.py set-target esp32s3
idf.py build
idf.py -p COM3 flash monitor
```

Record the printed GPIO-to-ADC mapping, then compare it against the board
schematic/pinout. Only connect a buffered, biased, in-range analog-front-end
output to a chosen ADC GPIO; never connect a raw sEMG electrode directly.

The probe audits the proposed direct-input pins GPIO1, GPIO2, and GPIO4--GPIO9
before listing all ADC-capable chip pads. GPIO3 is deliberately excluded: it
is an ESP32-S3 strapping pin, so an analog-front-end output could influence
boot before firmware can inspect its configuration. Reject a candidate
immediately if its configuration dump contains `**RESERVED**`. The
`[periph_sig_ctrl]` label by itself is expected for an IOMUX function and does
not prove an active output conflict. The dump cannot prove a board pin is
physically exposed or identify an unrepresented board-level connection, so the
board schematic or pinout remains the final check.
