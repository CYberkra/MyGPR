@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

title MyGPR Launcher
cd /d "%~dp0"
set "MYGPR_ROOT=%CD%"
set "MYGPR_APP=%MYGPR_ROOT%\app_qt.py"
if defined LOCALAPPDATA (
    set "MYGPR_LOG_DIR=%LOCALAPPDATA%\MyGPR\logs\launcher"
) else (
    set "MYGPR_LOG_DIR=%MYGPR_ROOT%\logs\launcher"
)
if not exist "%MYGPR_LOG_DIR%" mkdir "%MYGPR_LOG_DIR%" >nul 2>nul
set "MYGPR_LOG=%MYGPR_LOG_DIR%\start_mygpr_%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%.log"
set "MYGPR_LOG=%MYGPR_LOG: =0%"

echo ==========================================
echo MyGPR one-click launcher
echo ==========================================
echo Project root: %MYGPR_ROOT%
echo Log file: %MYGPR_LOG%
echo.

if not exist "%MYGPR_APP%" (
    echo ERROR: app_qt.py not found in project root.
    echo Please run this launcher from the extracted MyGPR package root.
    pause
    exit /b 1
)

if not exist "%MYGPR_ROOT%\PythonModule" (
    echo ERROR: PythonModule folder not found.
    echo This source package should include PythonModule at the project root.
    pause
    exit /b 1
)

call :select_python
if not defined MYGPR_PY_EXE (
    echo ERROR: Could not find a usable Python runtime.
    echo Set MYGPR_PYTHON to your MyGPR Python executable, for example:
    echo   set MYGPR_PYTHON=your_python_executable
    pause
    exit /b 1
)

echo Using Python: %MYGPR_PY_EXE%
echo Project root: %MYGPR_ROOT% > "%MYGPR_LOG%"
echo Python: %MYGPR_PY_EXE% >> "%MYGPR_LOG%"
"%MYGPR_PY_EXE%" --version >> "%MYGPR_LOG%" 2>&1

set "PYTHONPATH=%MYGPR_ROOT%;%PYTHONPATH%"
set "QT_AUTO_SCREEN_SCALE_FACTOR=1"
set "QT_ENABLE_HIGHDPI_SCALING=1"
set "MPLBACKEND=QtAgg"

echo Checking required runtime modules...
"%MYGPR_PY_EXE%" -c "import sys; print(sys.executable); import PyQt6; import qfluentwidgets; import numpy; import pandas; import scipy; import matplotlib; import h5py; import yaml; import pywt; print('runtime_imports_ok')" >> "%MYGPR_LOG%" 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Required Python modules are missing in this environment.
    echo See log:
    echo   %MYGPR_LOG%
    echo.
    echo This launcher does not install packages automatically. Use your existing MyGPR environment,
    echo or install dependencies with:
    echo   python -m pip install -r requirements-dev.txt
    pause
    exit /b 1
)

echo Starting MyGPR...
echo Starting MyGPR at %DATE% %TIME% >> "%MYGPR_LOG%"
"%MYGPR_PY_EXE%" "%MYGPR_APP%" >> "%MYGPR_LOG%" 2>&1
set "MYGPR_EXIT=%ERRORLEVEL%"

echo.
if not "%MYGPR_EXIT%"=="0" (
    echo MyGPR exited with error code %MYGPR_EXIT%.
    echo Log file:
    echo   %MYGPR_LOG%
    pause
    exit /b %MYGPR_EXIT%
)

echo MyGPR closed normally.
exit /b 0

:select_python
rem Priority 1: explicit user override.
if defined MYGPR_PYTHON (
    if exist "%MYGPR_PYTHON%" (
        set "MYGPR_PY_EXE=%MYGPR_PYTHON%"
        goto :eof
    )
)

rem Priority 2: local venv beside this package.
if exist "%MYGPR_ROOT%\.venv\Scripts\python.exe" (
    set "MYGPR_PY_EXE=%MYGPR_ROOT%\.venv\Scripts\python.exe"
    goto :eof
)

rem Priority 3: Python launcher; prefer an available modern Python 3 runtime.
for /f "usebackq delims=" %%P in (`py -3.13 -c "import sys; print(sys.executable)" 2^>nul`) do (
    set "MYGPR_PY_EXE=%%P"
    goto :eof
)

rem Priority 4: active PATH python.
for /f "usebackq delims=" %%P in (`python -c "import sys; print(sys.executable)" 2^>nul`) do (
    set "MYGPR_PY_EXE=%%P"
    goto :eof
)

goto :eof
