# ProcessingLineageController v0.8.2

`ui/processing_lineage_controller.py` 是 MyGPR V0.8.x 的第二个 controller 拆分模块。它负责主 B-scan 下方处理链路 stepper，以及点击历史步骤时的临时查看状态。

## 管理范围

- `create_stepper_bar()`：创建处理链路 stepper UI。
- `sync_stepper()`：刷新 chip、active/current 状态和 tooltip。
- `on_step_clicked()`：处理历史步骤点击，临时切换 B-scan 显示。
- `set_display_override()` / `clear_display_override()`：管理 display-only 临时数据载荷。
- `get_active_plot_payload()`：决定当前绘图使用正式数据、历史 override，还是单图快照。
- `build_steps()` / `build_text()` / `build_tooltip()` / `update_display()`：生成处理链路摘要并同步主图状态 chip。

## 设计边界

本轮采用保守拆分。Controller 仍通过 `host` 读写既有主窗口属性，避免一次性改动绘图、报告导出和 AutoTune 同步路径。`GPRGuiQt` 中保留同名 wrapper，保证旧测试和旧调用路径稳定。

## 后续

V0.8.3 将继续拆分 B-scan 鼠标交互，建议新建 `ui/bscan_interaction_controller.py`，把 hover 十字线、平移、滚轮缩放、ROI 框选和滑动对比交互统一迁出主窗口类。
