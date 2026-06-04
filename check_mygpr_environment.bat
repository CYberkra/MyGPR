@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
set "MYGPR_ROOT=%CD%"

echo ==========================================
echo MyGPR environment check
echo ==========================================

echo Project root: %MYGPR_ROOT%
if exist "%MYGPR_ROOT%\app_qt.py" (echo app_qt.py: OK) else (echo app_qt.py: MISSING)
if exist "%MYGPR_ROOT%\PythonModule" (echo PythonModule: OK) else (echo PythonModule: MISSING)

if defined MYGPR_PYTHON if exist "%MYGPR_PYTHON%" set "PY=%MYGPR_PYTHON%"
if not defined PY if exist "%MYGPR_ROOT%\.venv\Scripts\python.exe" set "PY=%MYGPR_ROOT%\.venv\Scripts\python.exe"
if not defined PY for /f "usebackq delims=" %%P in (`py -3.13 -c "import sys; print(sys.executable)" 2^>nul`) do set "PY=%%P"
if not defined PY for /f "usebackq delims=" %%P in (`python -c "import sys; print(sys.executable)" 2^>nul`) do set "PY=%%P"

if not defined PY (
    echo Python: NOT FOUND
    pause
    exit /b 1
)

echo Python: %PY%
"%PY%" --version
set "PYTHONPATH=%MYGPR_ROOT%;%PYTHONPATH%"
"%PY%" -c "mods=['PyQt6','qfluentwidgets','numpy','pandas','scipy','matplotlib','h5py','yaml']; import importlib; missing=[]; [missing.append(m) for m in mods if importlib.util.find_spec(m) is None]; print('missing=' + ','.join(missing) if missing else 'runtime_imports_ok'); raise SystemExit(1 if missing else 0)"
if errorlevel 1 (
    echo.
    echo Environment check failed. Use your MyGPR Python environment or install requirements-dev.txt.
    pause
    exit /b 1
)

echo.
echo Environment check passed.
pause
exit /b 0
