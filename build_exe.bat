@echo off
chcp 65001 >nul
echo ========================================
echo GPR GUI 打包工具
echo ========================================
echo.

cd /d "%~dp0"

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请确保已安装 Python 3.8+
    pause
    exit /b 1
)

:: 检查 PyInstaller
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [信息] 正在安装 PyInstaller...
    pip install pyinstaller
)

:: 清理旧的构建文件
echo [1/4] 清理旧的构建文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__

:: 安装依赖
echo [2/4] 检查并安装依赖...
pip install -r requirements-dev.txt --quiet

:: 打包前自检
echo [3/5] 运行打包前自检...
python scripts\preflight_check.py
if errorlevel 1 (
    echo [错误] 打包前自检失败，请先修复问题
    pause
    exit /b 1
)

:: 使用 PyInstaller 打包
echo [4/5] 开始打包（这可能需要几分钟）...
pyinstaller gpr_gui.spec --clean --noconfirm

:: 检查结果
if exist "dist\GPR_GUI.exe" (
    echo [5/5] 打包成功！
    echo.
    echo ========================================
    echo 打包完成！
    echo ========================================
    echo.
    echo 生成的文件位置: %cd%\dist\GPR_GUI.exe
    echo.
    echo 文件大小:
    dir "dist\GPR_GUI.exe" | findstr "GPR_GUI.exe"
    echo.
    echo 是否打开输出目录？
    choice /c YN /m "打开文件夹 (Y/N)"
    if errorlevel 2 goto :end
    explorer dist
) else (
    echo [错误] 打包失败，请检查错误信息
    pause
    exit /b 1
)

:end
echo.
echo 完成！
pause
