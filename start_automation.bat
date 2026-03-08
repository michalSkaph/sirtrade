@echo off
setlocal
cd /d "%~dp0"

set "PYEXE=C:\Users\Lenovo\AppData\Local\Python\pythoncore-3.14-64\python.exe"
if not exist "%PYEXE%" (
  echo Python nebyl nalezen na %PYEXE%
  pause
  exit /b 1
)

%PYEXE% run_automation.py --source binance --symbol BTCUSDT --days 365
if errorlevel 1 (
  echo Automatizacni beh selhal.
  pause
  exit /b 1
)

echo Automatizacni beh dokoncen.
endlocal
