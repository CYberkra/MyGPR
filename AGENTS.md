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
