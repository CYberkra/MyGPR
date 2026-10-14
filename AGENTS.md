# AGENTS.md — MyGPR

Guidance for agentic coding tools working in this repository.
All commands below assume this directory is the working directory.

## Scope

- PyQt6 + qfluentwidgets desktop app for (UAV-)GPR data processing, v0.9.38.
- Main entry points: `app_qt.py` (GUI) and `cli_batch.py` (headless batch).
- 架构与工程约束见 [CLAUDE.md](CLAUDE.md)；设计或验收任务先查 [规格索引](docs/superpowers/specs/README.md)，区分现行合同与历史方案。

## Repo Map

- `app_qt.py` — GUI 入口（DPI PassThrough、`--smoke` 离屏截图）。
- `ui/` — Qt 前端：`main_window.py` 组装器 + `page_coordinator.py` 跨页信号链薄门面（委派 `coordinator_project.py` / `coordinator_processing.py` / `coordinator_jobs.py`）+ `dialogs.py` + `geo_utils.py` + `pages/`（八页，顶部页签）+ `widgets/` + `controllers/` + `desktop_backend_facade.py`（ui→core 统一通道）。
- `mygpr/` — 后端分层：interfaces / application / domain / infrastructure。
- `core/` — 遗留内核（仍活跃），由 mygpr infrastructure 适配器调用。
- `PythonModule/` — 算法包装器；经方法注册表动态加载，静态 grep 不到引用≠死代码。
- `tests/` + `sample_data/` — pytest 与测试夹具。
- `scripts/` — 质量检查/治理脚本；其中被 `config/schema_catalog.json` 注册的不可移动。
- 历史轮次文档与一次性脚本已于 2026-07-30 清理，可从 git 历史（a9fb92e）取回。

## Run / Test

Python 支持 **3.12–3.13**（`pyproject.toml`: `>=3.12,<3.14`）；本机 2026-10-14 核验为 3.12.0。依赖基线：

- 核心：numpy 2.5.1、pandas 2.3.3、scipy 1.18.0、matplotlib 3.11.0、Pillow 12.3.0、h5py 3.16.0、PyYAML 6.0.3、PyWavelets 1.9.0、pyproj 3.7.2、openpyxl 3.1.5、rasterio 1.5.0、fiona 1.10.1、defusedxml 0.7.1。
- GUI：PyQt6 6.11.0、PyQt6-Qt6 6.11.1、PyQt6-Fluent-Widgets 1.11.2、pyqtgraph 0.14.0、PyOpenGL 3.1.10、colorlog 6.12.0。
- 核心精确版本在 `requirements.txt`；`requirements-core.txt` 引用它，GUI 在 `requirements-gui.txt`；与 `pyproject.toml` 同步。开发工具按 `requirements-test-backend.txt`，不要用本机 `pip freeze` 覆盖跨平台依赖声明。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-gui.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe app_qt.py
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe app_qt.py --smoke
.\.venv\Scripts\python.exe -m pytest tests/ -q
Remove-Item Env:QT_QPA_PLATFORM
```

Git Bash / Linux：激活对应 `.venv/Scripts/activate` / `.venv/bin/activate` 后使用：

```bash
python app_qt.py
QT_QPA_PLATFORM=offscreen python app_qt.py --smoke
python -m pytest tests/ -q
```

截图默认在系统临时目录 `mygpr_shots/`，可用 `MYGPR_SMOKE_SHOTS_DIR` 覆盖。改动依赖时检查 requirements、`pyproject.toml`、README 与本节是否一致，并运行 `python -m pip check`。

## 当前入口边界

- 项目导入：CSV/TXT/NPY/NPZ/H5/HDF5；厂商二进制底层读取器不等于已接通项目导入向导。格式约定见 [数据格式参考](docs/user/reference/数据格式参考.md)。
- CLI 作业使用显式 `methods`，`recommended_profile` 已移除。
- 设计 SPEC 在 `docs/superpowers/specs/`；当前无根 `SPEC.md` 或 PyInstaller `.spec`。exe 打包属于待重新建立和验收的发布流程。

## Hard Rules

- 不在 `main` 上直接提交；功能分支开发，PR 合并。
- 长任务走 controller `run_command`（daemon 线程 + `_XxxCommand`）+ JobBridge，跨线程通知走 pyqtSignal；工作线程与 UI 线程互不越界。
- 大文件用 mmap/分块 I/O；文件写隐藏临时文件后原子替换。
- Windows：只读句柄 `os.fsync` 会失败，统一用 `core/storage_primitives.py` 的 `fsync_file()`。
- 删除任何 `PythonModule`/`scripts` 文件前，先对照 `core/method_registry_metadata.py` 与 `config/schema_catalog.json`。

## Artifact Policy

- 大型原始数据、完整报告输出、GUI 截图不进 Git。
- 需要评审的小证据文件放入 `docs/artifacts/` 随提交一起推送。

## 研发技能路由

按任务使用下列技能；首次应用先读对应 `SKILL.md`，不要求用户每次点名。
安装位置、验证入口和证据标准见 [研发技能工作流](docs/agents/development-skills.md)。

| 任务 | 使用技能 |
| --- | --- |
| PyQt 界面设计、布局与操作流程 | `frontend-design-polish`；实际桌面交互验收时用 `windows-desktop-e2e` |
| GPR 导入、处理算法、参数、异常 B-scan、GUI/CLI 一致性 | `mygpr-processing-validation`；需要回归测试时用 `python-testing` |
| 缺陷、失败测试、卡顿、串图、异步竞态 | `superpowers:systematic-debugging`，先复现和定位，再修复 |
| 编写或维护 Python 测试 | `python-testing`；桌面端到端操作使用 `windows-desktop-e2e` |
| 跨层接口或较大模块重构 | `codebase-design`，遵循现有架构约束 |

- 每次只加载与当前任务相关的技能；同名/同用途技能选一份，不叠加多个开发流程。
- 用户要求“仅计划/仅设计”时，只交付相应文档，不因技能流程擅自实施。
- 通用技能中的框架示例、固定覆盖率目标、默认参数及旧路径须先核对当前项目；不能直接当成项目事实或新增硬门槛。
- 科学验证看数据和可复现结果；离屏测试、截图、合成数据不能代替真实数据验证或 Windows 桌面验收。
