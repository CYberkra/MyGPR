# MyGPR v0.8.73 算法兼容性专项回归

## 目标

本轮对测线处理页已接入的已有算法进行专项回归。检查范围不是新增算法，而是确认新工作台通过：

```text
GPRDataSet -> field_processing_bridge -> methods_registry -> processing_engine -> PythonModule
```

调用已有算法时，参数生成、执行、输出、日志和保存链路是否稳定。

## 本轮边界

- 不接入预设流程处理。
- 不接入 `QUICK_PRESETS` 或 `workflow_executor`。
- 不在当前测线处理页暴露会让用户误解为默认改变道间距的入口。
- 不改写已有算法的科学含义。
- 优先修复桥接层和 UI 错误恢复，不把算法逻辑堆到按钮回调里。

## 回归数据

使用确定性示例 B-scan：

```text
line_id: L03
shape: 160 samples x 96 traces
length: 40.0 m
source: GPRDataSet.synthetic(...)
```

## 兼容性结果

| 算法 ID | 中文名称 | 分类 | 参数生成 | 执行 | 输入输出尺寸 | 保存风险 | 本轮状态 |
|---|---|---|---|---|---|---|---|
| `dewow` | 去低频漂移 dewow | 校正预处理 | 正常 | 正常 | 160×96 -> 160×96 | 低 | 通过 |
| `subtracting_average_2D` | 平均背景去除 | 背景抑制 | 正常 | 正常 | 160×96 -> 160×96 | 低 | 通过 |
| `median_background_2D` | 中值背景去除 | 背景抑制 | 正常 | 正常 | 160×96 -> 160×96 | 低 | 通过 |
| `svd_bg` | SVD 背景抑制 | 背景抑制 | 正常 | 正常 | 160×96 -> 160×96 | 低 | 通过 |
| `frequency_filter_1d` | 一维频率滤波 | 频率滤波 | 正常 | 正常 | 160×96 -> 160×96 | 低 | 通过 |
| `sec_gain` | SEC 增益 | 增益补偿 | 正常 | 正常 | 160×96 -> 160×96 | 低 | 通过 |
| `agcGain` | AGC 增益 | 增益补偿 | 正常 | 正常 | 160×96 -> 160×96 | 低 | 通过 |
| `trace_median_filter` | 道向中值滤波 | 去噪增强 | 正常 | 正常 | 160×96 -> 160×96 | 低 | 通过 |
| `wavelet_2d` | 二维小波去噪 | 去噪增强 | 正常 | 正常 | 160×96 -> 160×96 | 低 | 通过 |
| `time_to_depth` | 时间-深度转换 | 迁移与深度 | 正常 | 正常 | 160×96 -> 40×96 | 中 | 通过，需注意采样维度变化 |

## 本轮代码侧改进

1. `core/field_processing_bridge.py` 新增 `COMPATIBILITY_CHECK_METHOD_IDS`，固定 v0.8.73 优先回归算法清单。
2. 新增 `FieldMethodCompatibilityRecord`、`check_method_compatibility()` 和 `run_priority_compatibility_checks()`，用于后续自动化兼容性报告。
3. 处理 manifest 增加：
   - `status`
   - `sample_count_changed`
   - `trace_count_changed`
   - `warnings`
4. 测线处理页错误恢复增强：
   - 算法失败后不覆盖当前结果预览。
   - 算法失败后保存按钮禁用。
   - 右侧处理信息显示失败原因。
   - 底部日志说明算法名、错误类型和处理状态。
5. 保存逻辑增加失败保护，避免错误结果写入项目目录。

## 风险分级

- P0：无。
- P1：`time_to_depth` 会改变采样维度，当前可执行、可显示，但保存前需要用户理解输出含义。
- P2：复杂运动补偿、迁移类算法需要更多真实轨迹和采集参数专项回归。
- P3：部分算法的中文参数名仍可继续精修。

## 验证命令

```bash
python -m compileall . -q
python scripts/check_version_consistency.py --expected 0.8.73
python -m pytest tests/test_field_project_store.py tests/test_round4_data_interfaces.py tests/test_field_processing_bridge.py tests/test_algorithm_compatibility.py tests/test_launcher_environment_selection.py -q
```
