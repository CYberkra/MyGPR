@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

title MyGPR v0.9.20 Launcher
cd /d "%~dp0"

echo ==========================================
echo MyGPR v0.9.20 one-click launcher
echo ==========================================
echo This launcher does not install packages automatically. Runtime modules include pywt / PyWavelets.
echo It will prefer MYGPR_PYTHON, activated Conda/venv, local .venv, Conda envs,
echo PATH python, then Windows Python Launcher.
echo.

rem Use any available Python only as a lightweight bootstrap. The Python script
rem below will search for the real MyGPR environment and will NOT install packages.
call :select_bootstrap_python
if not defined MYGPR_BOOTSTRAP_PYTHON (
    echo ERROR: Could not find Python to run the launcher helper.
    echo If you already have a MyGPR environment, run:
    echo   set MYGPR_PYTHON=C:\path\to\your\env\python.exe
    echo   start_mygpr.bat
    echo.
    echo To create the bundled local environment, run:
    echo   install_mygpr_environment.bat
    echo Manual alternative:
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%MYGPR_BOOTSTRAP_PYTHON%" "%~dp0scripts\mygpr_windows_launcher.py" %*
set "MYGPR_EXIT=%ERRORLEVEL%"
exit /b %MYGPR_EXIT%

:select_bootstrap_python
rem 1. Explicit Python is also a valid bootstrap.
if defined MYGPR_PYTHON (
    if exist "%MYGPR_PYTHON%" (
        set "MYGPR_BOOTSTRAP_PYTHON=%MYGPR_PYTHON%"
        goto :eof
    )
)

rem 2. Activated environments before system Python.
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

rem 3. Local package .venv if present.
if exist "%~dp0.venv\Scripts\python.exe" (
    set "MYGPR_BOOTSTRAP_PYTHON=%~dp0.venv\Scripts\python.exe"
    goto :eof
)

rem 4. PATH python. In Anaconda Prompt this catches the active env.
for /f "usebackq delims=" %%P in (`python -c "import sys; print(sys.executable)" 2^>nul`) do (
    set "MYGPR_BOOTSTRAP_PYTHON=%%P"
    goto :eof
)

rem 5. Last resort: Windows Python launcher.
for %%V in (3.13 3.12 3.11 3.10 3) do (
    for /f "usebackq delims=" %%P in (`py -%%V -c "import sys; print(sys.executable)" 2^>nul`) do (
        set "MYGPR_BOOTSTRAP_PYTHON=%%P"
        goto :eof
    )
)

goto :eof
