@echo off
setlocal EnableExtensions

echo ==========================================
echo MyGPR Quick Start with copied PythonModule
echo ==========================================

cd /d "%~dp0\.."
set "PKG_ROOT=%CD%"

if not exist "%PKG_ROOT%\PythonModule" (
    if exist "%FULL_MYGPR_ROOT%\PythonModule" (
        echo Copying PythonModule from:
        echo   %FULL_MYGPR_ROOT%\PythonModule
        echo to:
        echo   %PKG_ROOT%\PythonModule
        xcopy "%FULL_MYGPR_ROOT%\PythonModule" "%PKG_ROOT%\PythonModule" /E /I /Y >nul
    ) else (
        echo ERROR: Full PythonModule not found at %FULL_MYGPR_ROOT%\PythonModule
        pause
        exit /b 1
    )
)

set "PYTHONPATH=%PKG_ROOT%;%PYTHONPATH%"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app_qt.py
) else (
    python app_qt.py
)

if errorlevel 1 (
    echo.
    echo MyGPR exited with an error.
    pause
    exit /b 1
)

endlocal
