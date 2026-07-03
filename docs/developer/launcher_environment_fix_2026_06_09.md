# MyGPR Windows 启动器环境选择修复

## 问题

Round5 的 `start_mygpr.bat` 会优先通过 Windows `py` 启动器选择 `Python312`。如果用户本机已有 MyGPR 环境，但该环境没有处于项目 `.venv`，或者没有在当前终端激活，启动器可能会选择干净的系统 Python，然后报：

```text
ERROR: Required Python modules are missing in this environment.
```

这不是 GUI 或算法代码错误，而是启动器选择了错误的 Python 解释器。

## 修复

新增：

```text
scripts/mygpr_windows_launcher.py
```

并重写：

```text
start_mygpr.bat
check_mygpr_environment.bat
```

新的启动逻辑：

1. 不自动安装任何包。
2. 不强制使用系统 Python 3.12。
3. 优先使用 `MYGPR_PYTHON` 指定的 Python。
4. 优先识别已激活的 Conda/venv 环境。
5. 自动检查项目 `.venv`、PATH、Conda envs、Windows py launcher、常见 Anaconda/Miniconda 路径。
6. 对每个候选 Python 检查 MyGPR 运行模块，找到可用环境后再启动。
7. 如果找不到可用环境，会打印所有检查过的 Python 和缺失模块，不会误导用户必须重装。

## 推荐手动指定已有环境

如果用户已有固定 MyGPR 环境，可以在命令行执行：

```bat
set MYGPR_PYTHON=C:\path\to\your\mygpr_env\python.exe
start_mygpr.bat
```

也可以永久设置用户环境变量 `MYGPR_PYTHON`。
