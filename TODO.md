# MyGPR Backend TODO

当前后端基线：**v0.9.37**

## P0

1. ~~使用营山完整六测线数据执行后端 Golden 回归~~ **已完成（2026-08-31）**：`tests/industrial/scientific_validation/test_yingshan_golden_v1.py` 三层全绿（子集指纹 / 钻孔基准 / 全文件 SHA256+头部）。全文件层需设 `MYGPR_YINGSHAN_DATA` 指向数据目录（CI 无数据时自动 skip，属设计行为）；本机数据路径见 `_handoff_20260830/`。
2. ~~完成多 GB 导入、取消、磁盘不足、异常断电与恢复测试~~ **已完成（2026-08-31）**：分块导入/取消回滚、事务回滚/前滚、进程锁、深完整性此前已覆盖；磁盘不足（ENOSPC）缺口由 `tests/industrial/reliability/test_disk_full_injection_v1.py` 补齐（导入回滚 / 工件目录+日志回滚 / 原始容器可读性三项契约）。
3. ~~扩展厂商格式真实样例：RD3/RD7、DZT、DT1/HD、OKO、SEG-Y、ENVI~~ **已完成（2026-08-31）**：DZT/DT1/OKO 解码器落地（注册表全部 `native-subset`）；六格式验收资产齐备——真实样例 DZT/DT1（GPRPy）与 SEG-Y（segyio F3）+ 真实数据派生 RD3/ENVI + OKO 合成样例，全部 SHA256 登记 `tests/fixtures/vendor_formats_v1/vendor_formats_manifest.json`。遗留：OKO 真实样例无公开渠道（合成验收已覆盖），真实样例取得后按 `MYGPR_VENDOR_SAMPLE_DATA` 加 external 回归。计划见 `_handoff_20260830/P0-3_厂商格式样例与解码器计划.md`。
4. 固化 `mygpr/interfaces/` 与 `config/backend_api_v1.json`，供新前端调用。（已有机读契约 `backend_api_v1.json` + `check_backend_api_contract.py` + CI 契约测试；后续仅随 API 变更维护）

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

1. **候选 2：双执行器底层收敛**：实施计划见 `_handoff_20260830/任务F候选2_双执行器收敛实施计划.md`。**阶段 0–2 已完成**：`tests/test_native_convergence_baseline.py` + `fixtures/processing_convergence/descriptor_baseline.json` 钉死 36 方法数值/描述符基线；`metadata_bridge.py` 统一 display 元数据来源；`legacy_adapter.py` 与 Composite 两类已删除，生产装配直连 `NativeProcessingCatalog`/`NativeProcessingExecutor`（commits `52518b6`/`0f569fd`/`81122e8`）。待办：阶段 3 收尾（CI 确认、文档）。
2. **候选 3：关闭**（facade 重构后仅 259 行，原"体量膨胀"不成立，不再拆分）。
3. **RFC issue #6** 已归档任务 F 决策记录。
