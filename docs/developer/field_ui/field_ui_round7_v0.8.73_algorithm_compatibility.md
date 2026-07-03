# MyGPR v0.8.73：测线处理算法兼容性专项回归

v0.8.73 基于 v0.8.72 稳定版继续推进，重点是检查第五轮接入的已有算法体系是否真正可执行、可保存、可回归。

## 本轮完成

- 版本号推进到 `0.8.73`。
- 建立算法兼容性专项测试 `tests/test_algorithm_compatibility.py`。
- 建立算法兼容性报告 `docs/algorithm_compatibility_v0.8.73.md`。
- 在 `core/field_processing_bridge.py` 中加入兼容性记录结构和优先算法回归入口。
- 测线处理页算法失败时不覆盖当前预览，并禁用保存。
- 处理 manifest 增加输出尺寸变化标记，方便后续报告和项目审计。

## 继续保持的产品边界

- 不显示“单算法处理”文案。
- 不接入预设流程处理。
- 不在当前页面暴露会让用户误解为默认改变道间距的入口。
- 不把算法实现写进 UI 回调。

## 本轮验证重点

优先回归算法：

```text
dewow
subtracting_average_2D
median_background_2D
svd_bg
frequency_filter_1d
sec_gain
agcGain
trace_median_filter
wavelet_2d
time_to_depth
```

全部通过基础执行和有限输出检查。`time_to_depth` 会改变采样维度，但保留 trace 数量，本轮将其标记为可执行但需要保存前复核。

## 下一轮建议

v0.8.74 建议做“目标定位数据流增强”：让目标定位页可选择处理结果作为标注来源，并把目标标注与对应处理结果 manifest 建立稳定关联。
