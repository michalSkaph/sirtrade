@echo off
setlocal
cd /d "%~dp0"

set "PYEXE="
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if not defined PYEXE (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 set "PYEXE=python"
)
if not defined PYEXE if exist "C:\Users\Lenovo\AppData\Local\Python\pythoncore-3.14-64\python.exe" set "PYEXE=C:\Users\Lenovo\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if not defined PYEXE (
    echo Python nebyl nalezen. Nainstaluj Python 3.11+ a zkus znovu.
    pause
    exit /b 1
)

echo Pouzivam: %PYEXE%
%PYEXE% -m pip install -r requirements.txt
if errorlevel 1 (
    echo Instalace zavislosti selhala.
    pause
    exit /b 1
)

%PYEXE% -m streamlit run app.py
endlocal
