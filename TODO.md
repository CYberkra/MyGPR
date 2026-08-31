# MyGPR Backend TODO

当前后端基线：**v0.9.37**

## P0

1. 使用营山完整六测线数据执行后端 Golden 回归。
2. 完成多 GB 导入、取消、磁盘不足、异常断电与恢复测试。
3. 扩展厂商格式真实样例：RD3/RD7、DZT、DT1/HD、OKO、SEG-Y、ENVI。
4. 固化 `mygpr/interfaces/` 与 `config/backend_api_v1.json`，供新前端调用。

## P1

1. 继续拆分 `core/` 历史算法与基础设施适配器。
2. 完成 GIS、三维、制图和报告服务的无界面 API。
3. 增加项目自动保存、崩溃恢复和增量备份服务。
4. 完善 AutoTune 安全边界、候选流程空间和证据导出。

## 前端（v0.9.37 已交付）

- Qt GUI 可用：`app_qt.py` 启动，8 页导航 + 右侧日志面板。
- 架构门禁已建立：`ui/` → `core/` 通过 `desktop_backend_facade.py` 统一通道。
- `MyGPRMainWindow` 已拆分：组装器（`main_window.py`）+ 跨页接线器（`page_coordinator.py`）。
- **任务 F 候选 1 已完成**（PageCoordinator 重构，commit `193d3a1` + CI 修复 `95ce4b4`/`c7281b6`，RFC #6）。
- CI 已覆盖：Linux offscreen、Windows GUI、Python 3.11/3.12/3.13、干净安装。

## 任务 F（架构优化）后续

1. **候选 2：双执行器底层收敛**：实施计划见 `_handoff_20260830/任务F候选2_双执行器收敛实施计划.md`。**阶段 0（等价性基线）已完成**：`tests/test_native_convergence_baseline.py` + `fixtures/processing_convergence/descriptor_baseline.json` 钉死 36 方法数值/描述符基线（native 覆盖已达 36/36，legacy 执行器生产路径已不可达）。元数据决策已定案：display 元数据经 `metadata_bridge.py` 引用 core 单一来源，parameter_schema 固化进 `NativeAlgorithm`。待办：阶段 1 元数据收敛 → 阶段 2 拆除 Legacy/Composite 四类 → 阶段 3 收尾。
2. **候选 3：关闭**（facade 重构后仅 259 行，原"体量膨胀"不成立，不再拆分）。
3. **RFC issue #6** 已归档任务 F 决策记录。
