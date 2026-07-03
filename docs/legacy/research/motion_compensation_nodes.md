# UAV-GPR 运动补偿节点说明

## 当前主线

MyGPR 当前把 `motion_compensation_v2` 作为论文与科研验证的推荐统一入口。它同时处理高度归一化、姿态/APC 足迹、轨迹距离与等距重采样，并统一输出 `runtime_warnings`、`quality_flags`、`trace_metadata_updates` 或 `trace_metadata_out`。

四个用户可见原子节点仍保留原 method id 和中文名称，用于分项验证、ablation、教学展示和调试：

| method_id | 中文名称 | 当前实现语义 |
| --- | --- | --- |
| `trajectory_smoothing` | 轨迹平滑 | 保留现有平滑算法，输出统一 warnings / quality_flags / trace metadata updates。 |
| `motion_compensation_speed` | 速度误差补偿 | 调用共享等距重采样 helper，输出重采样后的 `trace_metadata_out`。 |
| `motion_compensation_attitude` | 姿态/APC足迹修正 | 调用共享姿态/APC footprint helper，优先使用 `height_agl_m`，回退 `flight_height_m`。 |
| `motion_compensation_height` | 飞行高度归一化 | 调用共享高度 helper，优先 `height_agl_m`，空气路径默认 `0.299792458 m/ns`。 |

这些原子节点不再代表旧 V1 物理契约。旧 V1 的思想和早期 benchmark 仍可作为代码兼容/回归参考，但不作为普通 UI、默认 preset 或论文主线证据。

## 共享物理假设

- 高度字段优先级：`height_agl_m` -> `flight_height_m` fallback。
- 空气路径速度默认：`0.299792458 m/ns`。
- 高度时移必须有真实 `time_window_ns` / `total_time_ns`；缺失时跳过 time-shift 并记录 warning，不伪造时间窗。
- 时移使用 `max_shift_samples` 和 `max_shift_ns` 共同约束，并受数据长度上限保护。
- 振幅归一使用 `max_amplitude_scale` 约束，防止高度异常导致过度增益。
- 姿态/APC 计算会输出 `footprint_x_m`、`footprint_y_m` 和重算后的 `trace_distance_m`。
- 等距重采样会输出完整 `trace_metadata_out`，使后续处理步骤看到新的 trace 数。

## 推荐用法

论文主线和正式处理优先使用：

```text
motion_compensation_v2
```

四个原子节点适合用于：

- 分项证明每个几何修正模块的作用。
- 做 ablation，展示缺少某一步时的影响。
- 教学或组会中解释运动补偿组成。
- 调试 sidecar 字段质量、trace metadata 更新和 GUI/CLI 运行链路。

`motion_compensation_vibration` 已移出运动补偿核心链，定位为“周期条带伪影抑制（实验）/ artifact suppression”。它不是运动补偿物理模块，也不应放入默认 UAV-GPR 运动补偿 profile。
