# MyGPR Backend TODO

当前后端基线：**v0.9.38**

## P0

1. ~~使用营山完整六测线数据执行后端 Golden 回归~~ **已完成（2026-08-31）**：`tests/industrial/scientific_validation/test_yingshan_golden_v1.py` 三层全绿（子集指纹 / 钻孔基准 / 全文件 SHA256+头部）。全文件层需设 `MYGPR_YINGSHAN_DATA` 指向数据目录（CI 无数据时自动 skip，属设计行为）；本机数据路径见 `_handoff_20260830/`。
2. ~~完成多 GB 导入、取消、磁盘不足、异常断电与恢复测试~~ **已完成（2026-08-31）**：分块导入/取消回滚、事务回滚/前滚、进程锁、深完整性此前已覆盖；磁盘不足（ENOSPC）缺口由 `tests/industrial/reliability/test_disk_full_injection_v1.py` 补齐（导入回滚 / 工件目录+日志回滚 / 原始容器可读性三项契约）。
3. ~~扩展厂商格式真实样例：RD3/RD7、DZT、DT1/HD、SEG-Y、ENVI~~ **已完成（2026-08-31）**：DZT/DT1 解码器落地（注册表全部 `native-subset`）；五格式验收资产齐备——真实样例 DZT/DT1（GPRPy）与 SEG-Y（segyio F3）+ 真实数据派生 RD3/ENVI，全部 SHA256 登记 `tests/fixtures/vendor_formats_v1/vendor_formats_manifest.json`。计划见 `_handoff_20260830/P0-3_厂商格式样例与解码器计划.md`。
   注：OKO GPR2 支持曾随本项落地（合成样例验收），**2026-09-01 已按用户决策整体移除**（无公开真实样例渠道，设备不使用）——解码器、注册表条目、夹具与文档引用已删除。
4. 固化 `mygpr/interfaces/` 与 `config/backend_api_v1.json`，供新前端调用。（已有机读契约 `backend_api_v1.json` + `check_backend_api_contract.py` + CI 契约测试；后续仅随 API 变更维护）

## P1

1. ~~继续拆分 `core/` 历史算法与基础设施适配器~~ **核心处理栈收敛已完成（2026-08-31）**：cli_batch/evidence_export/field_processing_bridge 全部切 `NativeProcessingExecutor` 生产路径，`processing_engine` 降级为测试对照基线（4 个手写 kernel 是 atol=0 等价性锚点，1.1.0 退役）；注册表/桥/ bindings 的进一步清理见 `_handoff_20260830/P1-1_core历史处理栈收敛计划.md` 第 5 节。
2. ~~完成 GIS、三维、制图和报告服务的无界面 API~~ **已完成（2026-08-31）**：`MyGPRBackend.build_georeference_3d`（job 系统）补上三维地理配准入口；契约 `backend_api_v1.json` 扩容至全部 18 个公开方法（含 `submit_spatial_result`/`submit_project_restore` 等），配防漂移测试；无头全链路验收（导入→传感器同步→界面标注→3D job→空间成果→报告包）纯公共 API 跑通。顺带修两个存量 bug：解释审计钩子递归（catalog.append_audit 从未触达）、`submit_spatial_result` 的 `check_cancelled` 笔误。commit `1897d3f`。
3. ~~增加项目自动保存、崩溃恢复和增量备份服务~~ **已完成（2026-08-31）**：崩溃恢复已有事务回滚/前滚 + ENOSPC 注入覆盖（P0-2）；本次补齐 `backup_project_archive` **增量链**（对比基准 manifest 只打包变更文件、恢复时自动沿 `incremental_base_archive` 回溯合并、缺基准即报错）与**保留策略** `retention_keep`（成功后按 project_id 裁剪最旧档案）；参数经 ports/Service/`submit_project_backup` 全链透传，契约签名重冻结。commit `980dbe5`。
   注：周期性"自动备份调度"属 UI 定时器职责，无头场景用外部调度调 `submit_project_backup(incremental=True, retention_keep=N)` 即可，后端不再内置调度器。
4. ~~完善 AutoTune 安全边界、候选流程空间和证据导出~~ **已完成（2026-08-31）**：调研确认约束层（`domain/autotune/constraints.py` 554 行，数据尺度派生安全域）与候选规划器已完备，但**零直接测试**；补 9 个用例钉住（dewow/agcGain/set_zero_time 越界夹紧+警告、非数值参数保持原样交失败试验、候选生成确定性且全部落在约束域内）；新增 `mygpr/application/autotune/evidence.py`（`mygpr.autotune_evidence.v1`，全 trials+推荐+偏好审计+body SHA-256 溯源落盘）并登记 schema_catalog。commit `3032989`/`3682af7`。

## P2

0. **处理链路需求实施（2026-09-01 拷问确立）**：B1 参数自动写入（删应用按钮）✅、B3 预设档全删 ✅（commit `1631f27`，breaking：cli_batch `recommended_profile` 校验报错）、B4 批量处理卡 ✅（`85af1e7`）、B7 逐步中间成果落盘+运行组归组+一键清理 ✅（`8493449`）、C1 SEG-Y 导出 ✅（`export_artifact_segy`，契约 19 方法）。全部 CI 绿。ML 阶段后置项：链模板/链推荐/非线性拓扑。

0.5 **启动耗时治理（2026-09-01）**：`-X importtime` 精确剖析后把 matplotlib 链移出 GUI 启动导入路径——`report_export_renderers` 的 `gis_map_export`（顶层拉 `matplotlib.figure`）与报告包适配器链改为函数级导入，导入树 **3.75s → 2.97s（-21%）**，matplotlib 完全退出启动期。报告/GIS 出图运行时行为不变（Agg 无头红线保持）。剩余头部：注册表链 ~1.3s（架构性，算法单一事实源启动期必需）、qfluentwidgets ~0.6s（UI 框架本体）、`pyqtgraph.opengl` ~0.29s（SpatialPage 构造期即建 3D 视图，惰性化需动 14 处引用点+数据补投，收益/风险比低，暂缓）。

1. ~~软件使用指南~~ **初版已完成（2026-08-31）**：`docs/user/用户指南.md`——安装启动、8 页导览、格式支持矩阵（P0-3 后全直读）、GUI/批处理双工作流、备份恢复与故障排查；`GUI_README.md` 过时信息（DT1/DZT"仅识别不解码"、导航页数）已同步修正。后续按用户反馈迭代章节（教程截图不进 git）。
2. ~~全面构建使用文档（Diátaxis 四象限）~~ **已完成（2026-08-31）**：按行业标准 [Diátaxis](https://diataxis.fr/) 框架重构为文档树 `docs/user/`——**tutorials**（营山零基础全流程 15 分钟）、**how-to** ×6（导入/传感器同步/处理与AutoTune/标注与空间成果/报告备份/无界面批处理）、**reference** ×3（格式矩阵含二进制布局速查/预设档算法序列/项目目录与文件位置）、**explanation**（处理证据链与数据安全模型）；入口 `docs/user/README.md` 按目的路由，旧单文件指南降为速览版并互链；15 文件相对链接互验全过。

## 前端（v0.9.37 已交付）

- Qt GUI 可用：`app_qt.py` 启动，8 页导航 + 右侧日志面板。
- 架构门禁已建立：`ui/` → `core/` 通过 `desktop_backend_facade.py` 统一通道。
- `MyGPRMainWindow` 已拆分：组装器（`main_window.py`）+ 跨页接线器（`page_coordinator.py`）。
- **任务 F 候选 1 已完成**（PageCoordinator 重构，commit `193d3a1` + CI 修复 `95ce4b4`/`c7281b6`，RFC #6）。
- CI 已覆盖：Linux offscreen、Windows GUI、Python 3.11/3.12/3.13、干净安装。

## 任务 F（架构优化）后续

1. **候选 2：双执行器底层收敛**：实施计划见 `_handoff_20260830/任务F候选2_双执行器收敛实施计划.md`。**全部阶段（0–3）已完成（2026-08-31）**：`tests/test_native_convergence_baseline.py` + `fixtures/processing_convergence/descriptor_baseline.json` 钉死 36 方法数值/描述符基线；`metadata_bridge.py` 统一 display 元数据来源；`legacy_adapter.py` 与 Composite 两类已删除，生产装配直连 `NativeProcessingCatalog`/`NativeProcessingExecutor`（commits `52518b6`/`0f569fd`/`81122e8`/`7c26667`）；阶段 3 收尾（CI 五链路全绿、GUI 冒烟、文档同步）亦已完成。后续演进见 P1-1（处理栈收敛，2026-08-31 完成）。
2. **候选 3：关闭**（facade 重构后仅 259 行，原"体量膨胀"不成立，不再拆分）。
3. **RFC issue #6** 已归档任务 F 决策记录。
4. **候选 4：UI 视觉一致性打磨**：计划与审计见 `_handoff_20260830/任务F候选4_UI视觉一致性打磨计划.md`。**已完成（2026-08-31，commit `ff375ef`）**：语义色单轨（日志 Bootstrap 四色 → 语义色同色相深底增亮变体，双轨残留归零）；六页私有分隔线工厂收敛为 `ui/widgets/separators.py`（主题自适应）；图表色板归一 `CHART_TRACK_COLORS`/`CHART_OVERLAY_COLOR`（matplotlib 默认蓝出清）。758 passed + 完整 gate 绿 + 冒烟 9 截图，行为零变化。

## 已知偶发 flake（待治理）

全量并发下偶发、单跑必过的两个用例（Windows 文件句柄/时序类，CI 历史无失败现场）：

1. `tests/test_backend_acquisition_api.py::test_acquisition_jobs_and_motion_pipeline_contract`（sensor_sync job）
2. `tests/test_hybrid_project_storage_v3.py::test_export_catalog_and_backup_snapshot_are_consistent`（备份快照缺 catalog.sqlite/L01.h5）

两处失败断言已改为**自动转储现场**（job 状态/阶段/错误详情、缺失文件清单）。下次复现时按断言消息定位根因；本会话内 3 次全量 + 组合 9 次未复现。
