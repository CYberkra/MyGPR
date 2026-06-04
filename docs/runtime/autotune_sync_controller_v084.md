# AutoTune Sync Controller v0.8.4

`ui/autotune_sync_controller.py` 是 V0.8.x 架构拆分的一部分，负责 AutoTune UI 同步和 no-prior guardrail 的低风险状态逻辑。

## 负责范围

- 将当前 B-scan 数据元信息同步到 `AutoTuneTuningPage`。
- 管理图上手动 ROI 框选开关。
- 将手动 ROI 转换为 AutoTune ROI bounds。
- 重置 AutoTune UI 结果摘要。
- 给 AutoTune 推荐结果附加 no-prior 风险标签。
- 构建 no-prior QC policy。
- 记录 no-prior guardrail 事件。
- 执行 UI 层 no-prior action guard。

## 不负责范围

- 不运行 AutoTune 算法。
- 不管理 QThread worker 生命周期。
- 不改 AutoTune scoring。
- 不导出 Evidence artifact。
- 不运行 gprMax。

## 兼容策略

`GPRGuiQt` 中保留旧方法名 wrapper，例如：

```python
self._sync_auto_tune_page_dataset_state(...)
```

内部转发到：

```python
self.autotune_sync_controller._sync_auto_tune_page_dataset_state(...)
```

这样可以降低 V0.8.x 分阶段拆分时的回归风险。

## 后续建议

V0.8.4 之后，`app_qt.py` 中仍保留 AutoTune worker/thread 编排。后续如果要继续拆，可单独做 `AutoTuneRunController`，但建议等 gprMax paired scoring 与真实 AutoTune 指标需求明确后再拆。
