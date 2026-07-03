# MyGPR v0.9.4 测线处理页逻辑拆分记录

## 目标

继续处理 Pass 3 审计提出的 P2 问题：`field_workbench_window.py` 仍然偏大。

## 拆分内容

新增 `ui/field_panels/processing_page.py`，承载：

- `_page_processing()`
- `_refresh_processing_preview()`
- `_collect_processing_params()`
- `_run_selected_processing()`
- `_preview_processing()`
- `_apply_processing()`
- `_undo_processing()`
- `_save_processing_result()`
- `_recommend_processing_params()`
- `_processing_params_card()`
- `_populate_processing_methods()`
- `_rebuild_processing_params_panel()`
- `_make_param_widget()`
- `_set_processing_param_values()`
- `_processing_messages_card()`

## 边界

本轮不改算法注册表、不改算法结果含义、不改保存协议，只移动 UI 回调边界。

## 风险控制

- `FieldWorkbenchWindow` 通过 `ProcessingPageMixin` 组合该能力。
- 新增测试确保主窗口不再包含处理页大段回调。
- 继续保留 `active_gpr_dataset`、`processed_gpr_dataset`、`last_processing_manifest` 等状态字段在主窗口初始化区，避免迁移时破坏现有状态同步。
\n## v0.9.5 复核\n\n处理页拆分结构在 v0.9.5 继续保留；本轮新增的是首页、表格工具和预览工具拆分。
