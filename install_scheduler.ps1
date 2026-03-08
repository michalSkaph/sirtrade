param(
    [string]$TaskName = "SirTrade-AutoRunner",
    [string]$Symbol = "BTCUSDT",
    [ValidateSet("simulation", "binance")]
    [string]$Source = "binance",
    [int]$Days = 365,
    [int]$IntervalMinutes = 15
)

$ErrorActionPreference = "Stop"

$pythonPath = "C:\Users\Lenovo\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if (-not (Test-Path $pythonPath)) {
    throw "Python nebyl nalezen na očekávané cestě: $pythonPath"
}

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$runScript = Join-Path $workspace "run_automation.py"

if (-not (Test-Path $runScript)) {
    throw "Soubor run_automation.py nebyl nalezen v: $workspace"
}

$arguments = "\"$runScript\" --source $Source --symbol $Symbol --days $Days"
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument $arguments -WorkingDirectory $workspace
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Naplánovaná úloha '$TaskName' byla vytvořena. Interval: každých $IntervalMinutes minut."
