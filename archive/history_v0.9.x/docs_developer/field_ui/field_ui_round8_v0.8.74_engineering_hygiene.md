# MyGPR v0.8.74 工程债整理与风险收敛记录

## 本轮定位

v0.8.74 不直接堆新功能，重点处理 v0.8.73 快速审视后暴露出的 P1/P2/P3 风险：

- 保留 `time_to_depth`，并把它明确归入“显示与对比 / 坐标轴转换”能力。
- 修复截图 `capture_summary.json` 可能引用旧版本 source root 的问题。
- 开始阻止 `ui/field_workbench_window.py` 无边界膨胀。
- 建立处理结果 artifact 索引，避免后续目标定位、空间成果和报告页依赖 UI 临时状态。
- 整理文档索引、术语约束和代码健康清单。

## time_to_depth 定位

`time_to_depth` 不删除、不隐藏。它原本属于显示与对比页能力的一部分，当前阶段暂不新增顶部主导航页，而是先作为测线处理页或成果页的子面板能力接入。

保存 `time_to_depth` 结果时，处理 manifest 会记录：

- `artifact_role = display_compare_transform`
- `axis_transform.kind = time_to_depth`
- `sample_count_changed`
- `trace_count_changed`
- `input_shape / output_shape`

后续目标定位、空间成果和报告页必须读取这些字段，而不能把它当成普通滤波结果。

## capture_summary 修复

新增：

- `ui/field_panels/capture_service.py`
- `scripts/capture_field_workbench.py`
- `tests/test_capture_summary.py`

截图摘要现在写入：

- `schema`
- `software`
- `version`
- `source_root`
- `source_root_name`
- `output_dir`
- `entrypoint`
- `capture_size`
- `screenshots`

并通过测试保证 `source_root` 指向当前项目根目录，不再复用旧版本路径。

## P2 边界治理

新增 `ui/field_panels/`，先迁出低风险内容：

- `field_ui_styles.py`：现场工作台样式常量和 1080P 尺寸基准。
- `processing_panel.py`：测线处理页文案和禁用术语守护。
- `capture_service.py`：截图和 summary 生成。

这不是大重构，而是给后续拆分测线处理、目标定位、空间成果页面建立落点。

## 处理结果索引

新增 `core/processing_artifact_index.py`，用于从项目目录读取：

```text
processed/<line_id>/
├─ *_processed_*.npy
├─ *_params.json
└─ *_processing_manifest_*.json
```

目标定位、空间成果和报告页后续应通过 artifact index 选择处理结果，而不是直接从测线处理页控件状态拿数据。

## 未做事项

- 未新增“显示与对比”主导航页。
- 未接入预设流程处理。
- 未做 PGDA-CSNet 自动目标识别模型接入。
- 未大规模拆分 `field_workbench_window.py`。
- 未改变已有算法含义。
