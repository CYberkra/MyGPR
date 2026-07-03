@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

echo ==========================================
echo MyGPR v0.9.20 environment check
echo ==========================================
echo This checker does not install packages automatically.
echo Required modules are listed in requirements.txt, including pywt / PyWavelets.
echo.

call :select_bootstrap_python
if not defined MYGPR_BOOTSTRAP_PYTHON (
    echo Python: NOT FOUND
    echo To create the bundled local environment, run install_mygpr_environment.bat
    echo Manual alternative: python -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%MYGPR_BOOTSTRAP_PYTHON%" "%~dp0scripts\mygpr_windows_launcher.py" --check %*
set "MYGPR_EXIT=%ERRORLEVEL%"
if not "%MYGPR_EXIT%"=="0" pause
exit /b %MYGPR_EXIT%

:select_bootstrap_python
if defined MYGPR_PYTHON (
    if exist "%MYGPR_PYTHON%" (
        set "MYGPR_BOOTSTRAP_PYTHON=%MYGPR_PYTHON%"
        goto :eof
    )
)
if defined VIRTUAL_ENV (
    if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        set "MYGPR_BOOTSTRAP_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
        goto :eof
    )
)
if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" (
        set "MYGPR_BOOTSTRAP_PYTHON=%CONDA_PREFIX%\python.exe"
        goto :eof
    )
    if exist "%CONDA_PREFIX%\Scripts\python.exe" (
        set "MYGPR_BOOTSTRAP_PYTHON=%CONDA_PREFIX%\Scripts\python.exe"
        goto :eof
    )
)
if exist "%~dp0.venv\Scripts\python.exe" (
    set "MYGPR_BOOTSTRAP_PYTHON=%~dp0.venv\Scripts\python.exe"
    goto :eof
)
for /f "usebackq delims=" %%P in (`python -c "import sys; print(sys.executable)" 2^>nul`) do (
    set "MYGPR_BOOTSTRAP_PYTHON=%%P"
    goto :eof
)
for %%V in (3.13 3.12 3.11 3.10 3) do (
    for /f "usebackq delims=" %%P in (`py -%%V -c "import sys; print(sys.executable)" 2^>nul`) do (
        set "MYGPR_BOOTSTRAP_PYTHON=%%P"
        goto :eof
    )
)
goto :eof
