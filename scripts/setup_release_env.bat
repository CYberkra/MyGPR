@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0\.."

echo ========================================
echo Setup Release Venv
echo ========================================

if exist ".release_venv\Scripts\python.exe" (
    echo [INFO] Reusing existing .release_venv
) else (
    echo [1/3] Creating .release_venv ...
    python -m venv .release_venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .release_venv
        exit /b 1
    )
)

echo [2/3] Upgrading packaging tools ...
".release_venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip/setuptools/wheel
    exit /b 1
)

echo [3/3] Installing runtime/build dependencies ...
".release_venv\Scripts\python.exe" -m pip install -r requirements-dev.txt pyinstaller pyinstaller-hooks-contrib pywin32
if errorlevel 1 (
    echo [ERROR] Failed to install release dependencies
    exit /b 1
)

echo.
echo Release environment is ready:
echo   %cd%\.release_venv
echo.
echo Next steps:
echo   1. Run preflight: .release_venv\Scripts\python.exe scripts\preflight_check.py
echo   2. Build exe:     .release_venv\Scripts\python.exe -m PyInstaller gpr_gui.spec --clean --noconfirm
echo.
pause
