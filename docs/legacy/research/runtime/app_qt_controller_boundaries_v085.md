# MyGPR V0.8.5 app_qt 收口审计

## 目标

V0.8.1–V0.8.4 已经依次拆出报告导出、处理链路、B-scan 交互、AutoTune/no-prior 同步四个 controller。V0.8.5 的目标不是新增功能，而是审计拆分后的边界，清理低风险残留，并建立回归检查。

## 本轮收口内容

- `app_qt.py` 用户主窗口仍是 GUI 装配入口，但不再直接导入 no-prior policy、no-prior guardrail、AutoTune 推荐标签等同步 helper。
- AutoTune / no-prior 同步 helper 归口到 `ui/autotune_sync_controller.py`。
- 删除了 `_relocate_basic_status_brief()` 这个无操作兼容方法及其调用。
- 清理了 `app_qt.py` 中 controller 拆分后不再使用的导入。
- 增加了 `tests/test_app_qt_controller_boundaries.py`，用于约束 V0.8 controller 边界和 `app_qt.py` 行数预算。

## 当前 controller 边界

| Controller | 文件 | 当前职责 |
|---|---|---|
| ReportExportController | `ui/report_export_controller.py` | Evidence report package、Markdown/HTML/PNG、sidecar 导出 |
| ProcessingLineageController | `ui/processing_lineage_controller.py` | B-scan 下方处理链路 stepper、历史步骤临时查看 |
| BscanInteractionController | `ui/bscan_interaction_controller.py` | 鼠标、滚轮、ROI、滑动对比、十字准线、视图范围 |
| AutoTuneSyncController | `ui/autotune_sync_controller.py` | AutoTune 页面数据同步、ROI 同步、no-prior guardrail |

## 保留边界

- `GPRGuiQt` 仍保留若干兼容 wrapper，保证旧测试和旧信号连接不需要一次性迁移。
- AutoTune worker/thread 主流程尚未拆出，后续如继续模块化，应单独做 `AutoTuneWorkflowController`。
- `app_qt.py` 仍偏大，但 V0.8.1–V0.8.5 已从约 9542 行降至约 7533 行。

## 后续建议

V0.8 架构整理基本可以收口。后续更建议进入研究主线：gprMax raw/background paired inventory、target_response 转换、AutoTune full-reference scoring 和 Evidence trial table。
