# MyGPR 一键启动说明

本包新增了根目录一键启动脚本：

- `启动MyGPR.bat` / `start_mygpr.bat`：启动 MyGPR 主界面。
- `启动MyGPR_调试日志.bat` / `start_mygpr_debug.bat`：以 `MYGPR_DEBUG=1` 启动，日志更详细。
- `检查MyGPR环境.bat` / `check_mygpr_environment.bat`：只检查 Python 和依赖，不启动 GUI。

## Python 环境选择顺序

启动脚本按以下优先级选择 Python：

1. 环境变量 `MYGPR_PYTHON` 指定的 Python。
2. 当前包内 `.venv\Scripts\python.exe`。
3. 你当前机器此前常用的 `<path-to-python.exe>`。
4. `py -3.13` 找到的 Python。优先使用与当前虚拟环境兼容的 Python 3 版本。
5. 当前 `PATH` 中的 `python`。

如果脚本选错环境，可在启动前设置：

```bat
set MYGPR_PYTHON=<path-to-python.exe>
start_mygpr.bat
```

## 依赖检查

脚本会先检查这些运行依赖：

- PyQt6
- qfluentwidgets
- numpy
- pandas
- scipy
- matplotlib
- h5py
- PyYAML

缺依赖时不会自动安装，而是停止并提示查看日志。这样避免污染你当前正在使用的环境。

## 日志位置

启动日志写入：

```text
%LOCALAPPDATA%\MyGPR\logs\launcher\start_mygpr_*.log
```

如果双击后窗口闪退，请先运行 `检查MyGPR环境.bat`，或查看上述日志。
