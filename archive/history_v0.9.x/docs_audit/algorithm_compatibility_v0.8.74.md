# MyGPR v0.8.74 算法兼容性补充说明

v0.8.74 沿用 v0.8.73 的算法兼容性回归结果，并补充 `time_to_depth` 定位说明。

## time_to_depth

- 保留：是
- 当前定位：显示与对比 / 坐标轴转换能力
- 是否作为默认流程：否
- 是否等同于降采样：否
- 是否需要 manifest 记录输出轴变化：是

保存结果时应记录：

```json
{
  "artifact_role": "display_compare_transform",
  "axis_transform": {
    "kind": "time_to_depth",
    "source_axis": "time_ns",
    "target_axis": "depth_m"
  },
  "sample_count_changed": true,
  "trace_count_changed": false
}
```

后续目标定位页、空间成果页和报告页若选择该结果作为来源，必须读取上述字段并提示用户结果语义。
