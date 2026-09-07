# ADS1298 8-channel bring-up (ESP32-S3)

This is a standalone ESP-IDF acquisition image. It reads eight normal
differential ADS1298 electrode inputs before filtering, feature extraction, or
model inference are introduced. It does **not** build or replace the existing
12-channel acquisition firmware.

## Default wiring — verify before powering

The following are editable firmware defaults inferred for an ESP32-S3 Zero.
They are not a proof of the particular board in hand; check its silk-screen
and the module P1 labels before supplying power.

| ADS1298 P1 signal | ESP32-S3 GPIO | Role |
| --- | ---: | --- |
| GND | GND | Common digital ground |
| 3V3 | 3V3 | Module power only |
| DRDY | GPIO4 | Data-ready input, falling-edge interrupt |
| CS | GPIO5 | SPI chip select |
| START | GPIO6 | Conversion-start control |
| SCLK | GPIO7 | SPI clock |
| DOUT | GPIO8 | ADS-to-ESP data (MISO) |
| DIN | GPIO9 | ESP-to-ADS commands (MOSI) |
| RESET (module silk-screen: `REST`) | GPIO1 | Active-low reset control |
| PWDN | GPIO2 | Active-low power-down control |

Do not connect `CLKSEL`, `CLK`, or `DAISY_IN` to the ESP: on this module the
onboard 2.048-MHz oscillator supplies CLK, `CLKSEL` is pulled low, and
`DAISY_IN` is pulled low. This firmware reads each `INxP`/`INxN` pair as a
normal differential electrode input. It does not make a USB-connected module
safe for human use; patient isolation, input protection, and the electrode/RLD
network remain separate hardware requirements.

## What the image does

The firmware performs a deterministic startup sequence, uses SPI Mode 1 at
1 MHz, verifies the ADS ID and configuration readback, enables the internal
reference, disables the ADS test signal (`CONFIG2=0x00`), and selects normal
electrode input on all eight channels (MUX=000 in `CH1SET..CH8SET`). Its
power-on wait is 150 ms plus one RTOS tick, deliberately longer than the ADS
128-ms minimum even when tick conversion rounds down.

After startup, the USB Serial/JTAG port is a **binary waveform protocol**, not
a text console. It accepts the supplied Windows application's parameter and
start/stop commands. The application may select 500, 1000, or 2000 SPS and
PGA 1/2/3/4/6/8/12; the ESP changes `CONFIG1` and all eight channel PGA fields,
then replies with the actual setting before emitting samples. 4000 SPS is
intentionally rejected at the present 1-MHz SPI clock because a 27-byte ADS
frame would leave too little DRDY/RTOS timing margin.

A normal frame is exactly 27 bytes:

```text
status[3] | CH1[3] | CH2[3] | ... | CH8[3]
```

Each channel is decoded as signed 24-bit two's complement, converted to
microvolts using the active PGA, then sent as CH1 through CH8 little-endian
`float32` values. A one-scan packet is exactly:

```text
A5 22 11 33 | CH1_uV(float32 LE) | ... | CH8_uV(float32 LE) | 5A
```

The firmware batches up to seven scans per packet. USB output is separated
from the DRDY/SPI task by a bounded queue, and no normal text is printed after
the binary USB driver starts.

## Build, flash, and see the eight traces

From an ESP-IDF PowerShell with the board connected:

```powershell
Set-Location D:\Project Antikythera\firmware\diagnostics\ads1298_bringup
idf.py set-target esp32s3
idf.py build
idf.py -p COM3 flash
```

Then open the supplied Windows upper computer, select the ESP COM port, choose
**500/1000/2000 sps** and the desired range, and click **开始采集**. The ESP
returns the selected parameters, then the application draws eight live traces.
The application opens the port at 1,000,000 baud; native USB Serial/JTAG does
not use that baud rate as a physical UART clock, but leave the application
setting unchanged for protocol compatibility.

Before compiling the upper computer, correct its receive length in `Uart.cs`:

```csharp
int n = serialPort.BytesToRead;
byte[] receivedData = new byte[n];
serialPort.Read(receivedData, 0, n);
```

The supplied source currently allocates `BytesToRead - 1`, leaving the final
byte of each OS-buffer read behind until future data arrives. Continuous plots
usually recover on the next packet, but a stopped stream can lose its final
packet. This transport work is not a medical-safety clearance for a person
connected to electrodes.

## Failure signatures

- No ID/readback: inspect 3V3/GND, PWDN, RESET, CS, SCLK, DIN, and DOUT.
- No trace after clicking start: confirm the app is on the ESP's COM port and
  its rate is at most 2000 SPS.
- Trace uses the wrong horizontal scale: the parameter response did not reach
  the PC; apply the `BytesToRead` receive-length fix above and reopen the port.
- Readback mismatch or a static trace: stop and correct the digital link
  before any electrode test.
