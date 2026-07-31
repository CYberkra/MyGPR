# MyGPR v0.8.77 目标定位来源绑定记录

本轮解决目标定位页依赖 UI 临时状态的风险。目标标注不再只记录里程、深度和类型，而是同步记录标注时使用的数据来源。

## 目标来源类型

- `raw`：原始 B-scan 数据。
- `processed`：已保存处理结果。
- `display_compare`：显示与对比 / 坐标轴转换结果，例如 `time_to_depth`。

## 新增持久化字段

`targets/<line_id>_targets.csv` 增加：

- `source_mode`
- `source_data_path`
- `source_manifest_path`
- `source_method_id`
- `source_method_name`
- `source_artifact_role`
- `source_axis_transform`
- `source_input_shape`
- `source_output_shape`

## 设计边界

- 不新增“显示与对比”主导航页。
- 不接入预设流程处理。
- 不恢复“单算法处理”文案。
- `time_to_depth` 保留，并通过 `axis_transform` 追踪其坐标轴语义。

## 验证

- `tests/test_target_source_binding.py`
- `tests/test_processing_artifact_index.py`
- GUI 目标定位页截图中可见“标注来源”选择。
