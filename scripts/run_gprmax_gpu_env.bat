@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."

if not defined MYGPR_GPRMAX_PYTHON (
  set "MYGPR_GPRMAX_PYTHON=E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe"
)
if not defined MYGPR_GPU_DEVICE (
  set "MYGPR_GPU_DEVICE=0"
)

set "VCVARS_CANDIDATE="
if defined MYGPR_VCVARS64 (
  set "VCVARS_CANDIDATE=%MYGPR_VCVARS64%"
) else (
  if exist "E:\sisual stdio 2022\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS_CANDIDATE=E:\sisual stdio 2022\VC\Auxiliary\Build\vcvars64.bat"
  ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS_CANDIDATE=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS_CANDIDATE=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
  ) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS_CANDIDATE=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
  ) else if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS_CANDIDATE=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  )
)

if "%~1"=="" goto usage
set "MODE=%~1"

if /i "%MODE%"=="--check" (
  call :load_env || exit /b 2
  call :check_env || exit /b 3
  echo [OK] GPU environment check passed.
  exit /b 0
)

if /i "%MODE%"=="--smoke" (
  call :load_env || exit /b 2
  call :check_env || exit /b 3
  pushd "%REPO_ROOT%"
  python scripts\diagnose_gprmax_gpu_env.py --clear-pycuda-cache --gprmax-python "%MYGPR_GPRMAX_PYTHON%" --gpu-device %MYGPR_GPU_DEVICE%
  if errorlevel 1 (
    popd
    exit /b 4
  )
  python scripts\diagnose_gprmax_gpu_env.py --gprmax-python "%MYGPR_GPRMAX_PYTHON%" --gpu-device %MYGPR_GPU_DEVICE%
  set "RC=%ERRORLEVEL%"
  popd
  exit /b %RC%
)

if /i "%MODE%"=="--" (
  set "_FORWARD_CMD=%*"
  set "_FORWARD_CMD=!_FORWARD_CMD:-- =!"
  if "!_FORWARD_CMD!"=="" (
    echo ERROR: missing command after --.
    exit /b 2
  )
  call :load_env || exit /b 2
  call :check_env || exit /b 3
  echo [INFO] Running command with GPU environment:
  echo        !_FORWARD_CMD!
  cmd /c "!_FORWARD_CMD!"
  exit /b %ERRORLEVEL%
)

:usage
echo Usage:
echo   scripts\run_gprmax_gpu_env.bat --check
echo   scripts\run_gprmax_gpu_env.bat --smoke
echo   scripts\run_gprmax_gpu_env.bat -- ^<command...^>
echo.
echo Environment overrides:
echo   MYGPR_VCVARS64
echo   MYGPR_GPRMAX_PYTHON
echo   MYGPR_GPU_DEVICE
exit /b 2

:load_env
if not defined VCVARS_CANDIDATE (
  echo ERROR: Visual Studio vcvars64.bat not found. Please open VS x64 Developer Command Prompt or configure MYGPR_VCVARS64.
  exit /b 1
)
if not exist "%VCVARS_CANDIDATE%" (
  echo ERROR: Visual Studio vcvars64.bat not found at "%VCVARS_CANDIDATE%". Please open VS x64 Developer Command Prompt or configure MYGPR_VCVARS64.
  exit /b 1
)
call "%VCVARS_CANDIDATE%" >nul 2>&1
if errorlevel 1 (
  echo ERROR: failed to load vcvars64.bat: "%VCVARS_CANDIDATE%"
  exit /b 1
)
exit /b 0

:check_env
if not exist "%MYGPR_GPRMAX_PYTHON%" (
  echo ERROR: gprMax runtime python not found: "%MYGPR_GPRMAX_PYTHON%"
  exit /b 1
)
where cl >nul 2>&1 || (
  echo ERROR: cl.exe not found after loading vcvars64.
  exit /b 1
)
where nvcc >nul 2>&1 || (
  echo ERROR: nvcc not found in PATH.
  exit /b 1
)
where nvidia-smi >nul 2>&1 || (
  echo ERROR: nvidia-smi not found in PATH.
  exit /b 1
)
where cl
cl >nul 2>&1
where nvcc
nvcc --version
nvidia-smi
"%MYGPR_GPRMAX_PYTHON%" -m gprMax --help >nul 2>&1
if errorlevel 1 (
  echo ERROR: "%MYGPR_GPRMAX_PYTHON%" -m gprMax --help failed.
  exit /b 1
)
echo [OK] gprMax runtime python help check passed.
exit /b 0
