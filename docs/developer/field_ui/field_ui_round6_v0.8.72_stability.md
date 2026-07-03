# MyGPR v0.8.72：版本规范、启动器回归与算法接入稳定性

## 本轮目标

v0.8.72 是 Round5 已有算法接入后的稳定性整理版本。重点不是新增大功能，而是把版本号、启动器、交接文档和测线处理页算法桥接固化下来。

## 版本规范

- `VERSION` 更新为 `0.8.72`。
- `CHANGELOG.md` 新增 `0.8.71` 和 `0.8.72` 记录。
- `README.md`、`START_MYGPR_README.md`、`CURRENT_STATE.md`、`DEV_HANDOFF.md` 同步当前版本和启动策略。
- 发布包与截图包按 `mygpr_v0.8.72_*` 命名。

## 启动器回归

启动器继续遵守不自动安装依赖的原则，只查找已有环境。优先顺序为：

```text
MYGPR_PYTHON
已激活 Conda / venv
项目 .venv
Conda / Mamba envs
PATH python
Windows Python Launcher
常见系统 Python
```

失败时输出已检查环境和缺失模块，并提示用户用 `MYGPR_PYTHON` 绑定已有环境，或显式运行安装器 / pip 命令。

## 算法桥接稳定性

测线处理页继续通过 `core/field_processing_bridge.py` 调用已有算法体系：

```text
GPRDataSet -> field_processing_bridge -> methods_registry -> processing_engine -> PythonModule
```

本轮继续保持：

- 不接预设流程处理。
- 不接 `QUICK_PRESETS`。
- 不接 `workflow_executor`。
- 不使用“单算法处理”作为用户可见文案。
- 不暴露会让用户误解为默认改变道数或采样间距的入口。

## 验证范围

建议验证：

```bash
python -m compileall .
python scripts/check_version_consistency.py --expected 0.8.72
python -m pytest tests/test_field_project_store.py tests/test_round4_data_interfaces.py tests/test_field_processing_bridge.py tests/test_launcher_environment_selection.py -q
```

GUI 截图继续按 1920×1080 输出六个主页面。
