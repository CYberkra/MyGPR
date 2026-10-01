# AGENTS.md — MyGPR

Guidance for agentic coding tools working in this repository.
All commands below assume this directory is the working directory.

## Scope

- PyQt6 + qfluentwidgets desktop app for (UAV-)GPR data processing, v0.9.38.
- Main entry points: `app_qt.py` (GUI) and `cli_batch.py` (headless batch).
- 详细架构与工程约束见 `CLAUDE.md`（现行版，2026-07-30 重写）。

## Repo Map

- `app_qt.py` — GUI 入口（DPI PassThrough、`--smoke` 离屏截图）。
- `ui/` — Qt 前端：`main_window.py` 纯组装器 + `page_coordinator.py` 跨页信号链薄门面（委派 `coordinator_project.py` / `coordinator_processing.py` / `coordinator_jobs.py` 三个域子接线器）+ `dialogs.py`（接线器许可的唯一对话框模块）+ `geo_utils.py`（覆盖统计纯函数）+ `pages/`（七页）+ `widgets/` + `controllers/` + `desktop_backend_facade.py`（ui→core 统一通道）。
- `mygpr/` — 后端分层：interfaces / application / domain / infrastructure。
- `core/` — 遗留内核（仍活跃），由 mygpr infrastructure 适配器调用。
- `PythonModule/` — 算法包装器；经方法注册表动态加载，静态 grep 不到引用≠死代码。
- `tests/` + `sample_data/` — pytest 与测试夹具。
- `scripts/` — 质量检查/治理脚本；其中被 `config/schema_catalog.json` 注册的不可移动。
- 历史轮次文档与一次性脚本已于 2026-07-30 清理，可从 git 历史（a9fb92e）取回。

## Run / Test

```bash
source .venv/Scripts/activate
python app_qt.py
QT_QPA_PLATFORM=offscreen python app_qt.py --smoke
python -m pytest tests/ -q
```

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
