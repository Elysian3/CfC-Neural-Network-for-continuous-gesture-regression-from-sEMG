# Host-side probe for the ADS1298 bringup binary protocol (ESP32-S3 USB CDC).
# Sends the sample-parameter + start commands, captures a few seconds of the
# binary stream, and dumps hex to a file for frame parsing.
#
# Usage:  powershell -ExecutionPolicy Bypass -File host_probe.ps1 [-Port COM3]
param(
    [string]$Port = "COM3",
    [int]$CaptureMs = 2500
)

$port = New-Object System.IO.Ports.SerialPort($Port, 1000000, 'None', 8, 'One')
$port.ReadTimeout = 500
$port.Open()
Write-Host "Opened $Port at 1,000,000 baud"

# Parameter command: 2000 SPS (rate_index=0x02), PGA index 4 (gain 6x)
$param = [byte[]](0xAA, 0x06, 0x80, 0x10, 0x02, 0x04, 0x00, 0x90, 0xBB)
$port.Write($param, 0, $param.Length)
Start-Sleep -Milliseconds 200

# Start command
$start = [byte[]](0xAA, 0x04, 0x80, 0x11, 0x01, 0x94, 0xBB)
$port.Write($start, 0, $start.Length)

$all = New-Object System.Collections.Generic.List[byte]
$deadline = [DateTime]::Now.AddMilliseconds($CaptureMs)
while ([DateTime]::Now -lt $deadline) {
    try {
        while ($port.BytesToRead -gt 0) {
            $b = $port.ReadByte()
            $all.Add([byte]$b)
        }
    } catch { }
    Start-Sleep -Milliseconds 10
}
$port.Close()

$hex = ($all | ForEach-Object { $_.ToString("X2") }) -join " "
$outFile = Join-Path $PSScriptRoot ("host_probe_capture_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Set-Content -Path $outFile -Value $hex
Write-Host "Captured $($all.Count) bytes -> $outFile"
