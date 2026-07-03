@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
set "MYGPR_ROOT=%CD%"
set "MYGPR_VENV=%MYGPR_ROOT%\.venv"

echo ==========================================
echo MyGPR environment installer
echo ==========================================
echo Project root: %MYGPR_ROOT%
echo Target venv:  %MYGPR_VENV%
echo.

if not exist "%MYGPR_ROOT%\requirements.txt" (
    echo ERROR: requirements.txt not found.
    pause
    exit /b 1
)

call :select_base_python
if not defined MYGPR_BASE_PYTHON (
    echo ERROR: Could not find Python 3.10+.
    echo Install Python 3.10 or newer, or set MYGPR_PYTHON to your python.exe.
    pause
    exit /b 1
)

echo Using base Python: %MYGPR_BASE_PYTHON%
"%MYGPR_BASE_PYTHON%" --version

if not exist "%MYGPR_VENV%\Scripts\python.exe" (
    echo.
    echo [1/4] Creating local .venv ...
    "%MYGPR_BASE_PYTHON%" -m venv "%MYGPR_VENV%"
    if errorlevel 1 (
        echo ERROR: Failed to create .venv.
        pause
        exit /b 1
    )
) else (
    echo.
    echo [1/4] Reusing existing .venv
)

echo.
echo [2/4] Repairing/upgrading packaging tools ...
"%MYGPR_VENV%\Scripts\python.exe" -m ensurepip --upgrade >nul 2>nul
"%MYGPR_VENV%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: Failed to prepare pip/setuptools/wheel.
    pause
    exit /b 1
)

echo.
echo [3/4] Installing MyGPR runtime dependencies ...
"%MYGPR_VENV%\Scripts\python.exe" -m pip install -r "%MYGPR_ROOT%\requirements.txt"
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo [4/4] Verifying imports ...
"%MYGPR_VENV%\Scripts\python.exe" -c "import PyQt6, qfluentwidgets, numpy, pandas, scipy, matplotlib, h5py, yaml, pywt; print('runtime_imports_ok')"
if errorlevel 1 (
    echo ERROR: Environment verification failed.
    pause
    exit /b 1
)

echo.
echo MyGPR environment is ready.
echo Start the application with:
echo   start_mygpr.bat
echo.
pause
exit /b 0

:select_base_python
if defined MYGPR_PYTHON (
    if exist "%MYGPR_PYTHON%" (
        set "MYGPR_BASE_PYTHON=%MYGPR_PYTHON%"
        goto :eof
    )
)
for %%V in (3.13 3.12 3.11 3.10) do (
    for /f "usebackq delims=" %%P in (`py -%%V -c "import sys; print(sys.executable)" 2^>nul`) do (
        set "MYGPR_BASE_PYTHON=%%P"
        goto :eof
    )
)
for /f "usebackq delims=" %%P in (`python -c "import sys; print(sys.executable)" 2^>nul`) do (
    set "MYGPR_BASE_PYTHON=%%P"
    goto :eof
)
goto :eof
