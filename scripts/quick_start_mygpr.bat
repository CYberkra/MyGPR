@echo off
setlocal EnableExtensions

echo ==========================================
echo MyGPR Quick Start
echo ==========================================

cd /d "%~dp0\.."
set "PKG_ROOT=%CD%"
echo Working directory: %PKG_ROOT%


if exist "%PKG_ROOT%\PythonModule" (
    set "PYTHONPATH=%PKG_ROOT%;%PYTHONPATH%"
) else (
    echo ERROR: PythonModule not found in this package.
    echo Copy PythonModule into the package root or start from the full MyGPR repository.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    echo Using local virtual environment: .venv
    ".venv\Scripts\python.exe" app_qt.py
) else (
    echo Local .venv not found. Falling back to system python.
    python app_qt.py
)

if errorlevel 1 (
    echo.
    echo MyGPR exited with an error.
    pause
    exit /b 1
)

endlocal
