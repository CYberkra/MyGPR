@echo off
chcp 65001 >nul
setlocal

set "PY=D:\Miniconda3\envs\turix_env\python.exe"
set "ROOT=%~dp0"

if not exist "%PY%" (
  echo [ERROR] Python not found: %PY%
  echo Check turix_env environment.
  pause
  exit /b 1
)

cd /d "%ROOT%"

echo ==========================================
echo   GPR GUI Launcher
echo   Python: %PY%
echo   Workdir: %ROOT%
echo ==========================================

"%PY%" app_qt.py

if errorlevel 1 (
  echo.
  echo [ERROR] Exit code: %errorlevel%
  pause
)

endlocal
