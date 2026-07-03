# MyGPR 一键启动说明

当前版本：**v0.9.9**

本包提供根目录一键脚本：

- `安装MyGPR环境.bat` / `install_mygpr_environment.bat`：创建本地 `.venv` 并安装 `requirements.txt` 中的运行依赖。
- `启动MyGPR.bat` / `start_mygpr.bat`：启动 MyGPR 现场工作台。
- `启动MyGPR_调试日志.bat` / `start_mygpr_debug.bat`：以 `MYGPR_DEBUG=1` 启动，日志更详细。
- `检查MyGPR环境.bat` / `check_mygpr_environment.bat`：只检查 Python 和依赖，不启动 GUI。

## Python 环境选择顺序

启动脚本不自动安装依赖，只查找已有可用环境。v0.9.9 的优先级为：

1. 环境变量 `MYGPR_PYTHON` 指定的 Python。
2. 当前已激活的 Conda / venv。
3. 当前包内 `.venv\Scripts\python.exe`。
4. 可发现的 Conda / Mamba 环境。
5. 当前 `PATH` 中的 `python`。
6. Windows Python Launcher 和常见系统 Python。

如果脚本选错环境，可在启动前设置：

```bat
set MYGPR_PYTHON=<path-to-python.exe>
start_mygpr.bat
```

## 依赖检查

脚本会检查这些运行依赖：

- PyQt6
- qfluentwidgets
- numpy
- pandas
- scipy
- matplotlib
- h5py
- PyYAML (`yaml`)
- PyWavelets (`pywt`)
- pyproj

缺依赖时启动脚本不会自动安装，而是停止并提示查看日志。需要自动准备本包专用环境时，先运行 `安装MyGPR环境.bat` / `install_mygpr_environment.bat`。

## 日志位置

启动日志写入：

```text
%LOCALAPPDATA%\MyGPR\logs\launcher\start_mygpr_*.log
```

如果双击后窗口闪退，请先运行 `检查MyGPR环境.bat`，或查看上述日志。

## 当前默认界面

默认启动的是面向实际勘探定位的现场工作台：

```text
项目管理 -> 测线处理 -> 目标定位 -> 空间成果 -> 成果报告
```

正式界面默认只显示现场勘探流程。开发人员如需打开历史研发工具，可在启动前设置：

```bat
set MYGPR_ENABLE_RESEARCH_UI=1
start_mygpr.bat
```

## 15.6 寸 1080P 笔记本适配验证

v0.9.9 会读取 Qt `availableGeometry()`，即屏幕扣除 Windows 任务栏后的可用区域。可用高度小于紧凑阈值时会自动进入 compact mode，压缩顶部栏、侧栏和图像预览高度。

在目标 Windows 笔记本上建议运行：

```bat
python scripts\capture_field_workbench_windows_diagnostics.py --output windows_fit_check
```

输出目录会包含：

```text
screen_diagnostics.json
00_home_project_overview_v0.9.9.png
01_project_management_v0.9.9.png
02_line_processing_v0.9.9.png
03_target_positioning_v0.9.9.png
04_spatial_results_v0.9.9.png
05_delivery_report_v0.9.9.png
```

`screen_diagnostics.json` 会记录完整屏幕、可用屏幕、DPI、捕获尺寸和 compact mode 状态。

## 首次使用建议

1. 解压 ZIP 后进入 MyGPR 根目录。
2. 双击 `install_mygpr_environment.bat` 创建本包专用 `.venv` 并下载依赖。
3. 双击 `start_mygpr.bat` 启动软件。
4. 若启动失败，运行 `check_mygpr_environment.bat`，再查看 `%LOCALAPPDATA%\MyGPR\logs\launcher` 下的日志。
