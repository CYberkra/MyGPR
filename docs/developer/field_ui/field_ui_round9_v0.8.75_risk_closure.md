# MyGPR v0.8.75 风险收敛记录

本轮不新增业务功能，专门处理 v0.8.74 遗留风险：

1. 继续降低 `ui/field_workbench_window.py` 膨胀风险。
2. 将目标定位、空间成果、成果报告页面逻辑迁出到 `ui/field_panels/*_page.py` mixin。
3. 将通用卡片组件迁出到 `ui/field_panels/widgets.py`。
4. 将绘图辅助迁出到 `ui/field_panels/plots.py`。
5. 实际建立并使用 `docs/user`、`docs/developer`、`docs/audit`、`docs/legacy` 文档分区。

保留原则：
- 不移除 `time_to_depth`。
- 不接入预设流程处理。
- 不恢复“单算法处理”文案。
- 不出现“重采样类”功能分组。
