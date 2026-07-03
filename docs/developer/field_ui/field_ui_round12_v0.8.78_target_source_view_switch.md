# MyGPR v0.8.78 目标定位图像来源真实切换

## 本轮目标

让目标定位页的 B-scan 显示真正跟随“标注来源”切换，而不是只记录来源字段。

## 已完成

1. 新增 `core/target_source_data.py`。
2. 支持将 `TargetSourceBinding` 解析为 `TargetSourceDataView`。
3. 原始数据来源显示 raw B-scan。
4. 已保存处理结果来源读取 `processed/<line_id>/*.npy`。
5. `time_to_depth` 等显示与对比 / 坐标轴转换来源使用深度轴显示。
6. 点击 B-scan 新增目标时，里程和深度按当前来源的距离轴、深度轴换算。

## 边界

- 本轮不做正式导入向导。
- 本轮不接入预设流程处理。
- 本轮不新增“显示与对比”顶部导航页。
- `time_to_depth` 保留，作为显示与对比 / 坐标轴转换能力。

## 验证

- `tests/test_target_source_data.py` 覆盖 raw / processed / time_to_depth 三类来源。
- GUI 截图需检查目标定位页来源下拉和 B-scan 垂直轴标签。
