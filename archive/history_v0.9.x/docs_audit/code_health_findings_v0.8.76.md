# MyGPR v0.8.76 代码健康收敛记录

## 已处理风险

v0.8.75 审视中指出 `core/field_project_store.py` 职责偏多。本轮已将该文件从约 576 行缩减到约 123 行，并拆出专门 store 模块。

## 当前状态

- P0：未发现。
- P1：未改变算法和项目数据语义。
- P2：项目存储边界已明确；后续新增目标、空间、报告持久化能力应进入对应 store 或 service，不应回填到 `field_project_store.py`。
- P3：文档索引已更新到 v0.8.76。

## 后续守护建议

1. 为 `core/field_project_store.py` 设置行数上限，避免它重新膨胀。
2. 后续目标定位来源绑定应优先改 `field_target_store.py` 或新增 `target_annotation_store.py`。
3. 报告导出相关逻辑不要进入 `field_project_store.py`，应新增 `report_artifact_store.py` 或 services 层。
