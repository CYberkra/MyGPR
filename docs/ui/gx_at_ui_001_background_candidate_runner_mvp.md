# GX-AT-UI-001 Background Candidate Runner MVP

本轮把 AutoTune 参数推荐页从纯 UI-local 预览推进到最小真实后端候选比较。

## 范围

仅接入背景抑制步骤的真实候选 runner：

- 不处理基线
- 均值背景扣除
- 中位数背景扣除
- SVD 背景抑制 rank sweep

## 数据链路

`app_qt.py` 在同步 AutoTune 页数据状态时，除文件名、shape、类型、阶段外，也传入当前 B-scan 数组：

```python
set_loaded_dataset(..., data_array=data)
```

`ui/autotune_tuning_page.py` 在点击“推荐”时：

- 如果当前步骤是“背景抑制”且已有 data_array，调用 `core.autotune_background_runner.run_background_candidates`
- 如果无 data_array 或其他步骤，回退 UI 预览候选

## 输出

- Candidate ranking
- Trial Table
- 推荐候选名称/参数/分数
- backend mode / backend message
- 风险提示和 claim boundary

## 评分说明

这是最小后端 MVP，不是论文级 production AutoTune scoring。评分为保守启发式：

- 背景残差降低
- ROI 能量保持
- CNR 增益

推荐结果仍需人工复核，不能宣称全局最优。

## 未改动

- 不修改生产 AutoTune 评分逻辑。
- 不写 Evidence。
- 不运行 gprMax。
- 不修改 GX-008/GX-009 模型。
- 不改变主 B-scan 数据。
