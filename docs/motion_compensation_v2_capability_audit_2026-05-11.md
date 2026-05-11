# motion_compensation_v2 真实能力检查（2026-05-11）

## 结论

`motion_compensation_v2` 当前已经具备可运行的 UAV-GPR 运动补偿后端能力，但仍应定位为 **V2 可验证基线**，不是最终论文级完整运动补偿模型。

它现在能真实修改 B-scan 的部分是：

- 基于离地高度的双程空气路径时移校正。
- 基于离地高度的保守振幅归一化。
- 可选等距道距重采样。

它现在主要更新元数据、但不直接进行完整成像几何校正的部分是：

- 姿态/APC 足迹估计。
- `local_x_m/local_y_m/footprint_x_m/footprint_y_m/trace_distance_m` 更新。

因此，当前 V2 适合放在标准流程的运动补偿位置，作为背景抑制、去噪、增益和迁移之前的几何/高度预校正层；但后续若要支撑论文/专利，应继续推进 V3 或增强版。

## 已确认能力

### 1. 输入字段

当前可使用的 per-trace metadata：

- `trace_index`
- `trace_timestamp_s`
- `trace_distance_m`
- `local_x_m`
- `local_y_m`
- `local_z_m`
- `roll_deg`
- `pitch_deg`
- `yaw_deg`
- `height_agl_m`
- `flight_height_m`
- `height_source`
- `height_confidence`
- `alignment_status`

高度源优先级：

1. `height_agl_m`
2. `flight_height_m` fallback，并输出 warning / quality flag

RTK altitude 不会被直接当作 AGL，这是正确的。

### 2. 高度时移校正

实现使用空气传播速度：

```text
c0 = 0.299792458 m/ns
time_shift_ns = 2 * (height_m - reference_height_m) / c0
```

输出：

- `time_shift_ns`
- `time_shift_samples`
- `raw_time_shift_samples_min`
- `raw_time_shift_samples_max`
- `time_shift_clamped`

若超过当前有效限幅，V2 会输出：

- `quality_flags: ["time_shift_clamped"]`
- `runtime_warnings: [{"code": "time_shift_clamped", ...}]`
- `max_shift_samples_effective`
- `max_shift_limit_source`

2026-05-11 后续增强：默认 profile 不再使用固定 `max_shift_samples=20`。现在默认使用 `max_shift_samples=0` + `max_shift_ns=20`，由采样间隔换算有效样点限幅，并再受数据长度比例上限保护。用户仍可显式设置 `max_shift_samples > 0` 来施加样点级硬限。

### 3. 高度振幅归一化

当前使用：

```text
amplitude_scale = (height_m / reference_height_m)^2
```

并由 `max_amplitude_scale` 限制最大放大/衰减，属于保守几何归一化，不等同于显示增益或解释增益。

### 4. 姿态/APC 足迹元数据

当具备 `local_x_m/local_y_m/roll_deg/pitch_deg/yaw_deg` 时，V2 会：

- 按 yaw 旋转 APC X/Y offset。
- 用 roll/pitch 和高度估计地面足迹偏移。
- 输出更新后的 `local_x_m/local_y_m/footprint_x_m/footprint_y_m/trace_distance_m`。

当前这一步主要是轨迹语义更新，不是完整 3D 成像补偿。

### 5. 等距道距重采样

当 `resample_spacing_m > 0` 时，V2 会：

- 按 `trace_distance_m` 对 B-scan 列方向插值。
- 同步重采样 per-trace metadata。
- 输出 `trace_metadata_out`。

## 本轮修正

本轮检查发现 gprMax airborne sidecar 与当前解析器存在不完全兼容：

- gprMax RTK sidecar 使用 `longitude_deg/latitude_deg`。
- gprMax RTK sidecar 已提供 `local_x_m/local_y_m/local_z_m`。
- 原解析器只认 `longitude/latitude`，且会忽略 explicit local XY。

已修正：

- RTK parser 支持 `longitude_deg/latitude_deg`。
- RTK parser 支持 `local_x_m/local_y_m/local_z_m`。
- sidecar integration 优先使用 explicit local XY，并据此更新 `trace_distance_m`。
- 普通经纬度 RTK 对齐不会无条件覆盖已有 `trace_distance_m`。
- 高度时移被 clamp 时会输出 runtime warning。

## gprMax 实际调用结果

测试场景：

```text
output/gprmax_multi_scenario_reports/20260511_004847/
scenarios/airborne_height_variation_cylinder_v1
```

输入：

- `mygpr_bscan.csv`
- `trace_timestamps.csv`
- `rtk.csv`
- `imu.csv`
- `altimeter.csv`

调用 `motion_compensation_v2` 后结果：

```json
{
  "input_shape": [3817, 96],
  "output_shape": [3817, 96],
  "height_source_used": "height_agl_m",
  "height_correction_applied": true,
  "time_shift_correction_applied": true,
  "amplitude_correction_applied": true,
  "resampling_applied": false,
  "height_summary": {
    "min_m": 0.085005,
    "max_m": 0.154995,
    "mean_m": 0.120000,
    "std_m": 0.024619
  },
  "raw_time_shift_samples_min": -49.477437,
  "raw_time_shift_samples_max": 49.477430,
  "time_shift_samples_min": -20.0,
  "time_shift_samples_max": 20.0,
  "trace_distance_start_end_m": [0.0, 0.38],
  "runtime_warning_codes": ["time_shift_clamped"]
}
```

这说明 V2 能正确读取合成 RTK/IMU/高度计 sidecar，并且确实对 B-scan 做了高度时移和振幅归一化。早期固定 `max_shift_samples=20` 对该高采样率 gprMax 场景会发生不合理截断；当前默认策略已改为 `max_shift_ns=20` 的数据自适应限幅。

## 当前不足

### P1：缺少论文级运动补偿验证报告

当前只有单元测试和一次实际调用检查，还缺少专门报告：

- 补偿前后 air-ground reflection 平直度。
- 目标 apex 时序稳定性。
- 目标 ROI 保真度。
- 背景/噪声变化。
- 时移、振幅倍率、姿态足迹曲线可视化。

### P1：姿态/APC 仍是简化模型

当前只支持简单 APC offset 和 roll/pitch/yaw 足迹估计，还没有：

- 外参校准文件。
- mount yaw/pitch/roll offset。
- RTK 天线点到雷达相位中心的完整 3D 刚体变换。
- 双天线 Tx/Rx 分离相位中心建模。

### P1：未做 radar horizon 高度估计

缺高度计时目前只能 fallback 或跳过，不会从 air-ground reflection 自动估计高度。

### P2：RTK 质量门控不足

目前能解析或保留部分 RTK 字段，但 V2 尚未把这些字段纳入质量门控：

- fix type
- satellites
- HDOP
- 轨迹跳变速度

### 已增强：时间戳/轨迹 gap 风险提示

V2 现在会在 `input_quality` 和 `runtime_warnings` 中报告：

- `trace_timestamp_nonmonotonic`
- `trace_timestamp_gap`
- `trace_distance_nonmonotonic`
- `trace_distance_gap`
- `trace_speed_outlier`

这些 warning 用于提示时间同步、断航迹、空间插值或重采样风险，避免把大间隙悄悄插值成连续测线。

### 已增强：默认时移限幅数据自适应

固定样点数阈值不够物理稳定。高采样率数据中，厘米级高度变化也可能对应几十个 sample。当前已新增 `max_shift_ns`，并根据 `total_time_ns / sample_count` 自动换算样点上限；后续报告中仍应展示 clamp 比例。

## 建议下一步

1. 新建 `motion_compensation_v2_benchmark_report.py`。
2. 用 `airborne_height_variation_cylinder_v1` 生成补偿前后 HTML 报告。
3. 加入 air-ground reflection 平直度、目标 ROI 保真度、时移曲线、振幅倍率曲线。
4. 在报告中展示有效时移限幅、clamp 比例、时间戳 gap 和轨迹 speed outlier。
5. 设计 V3 外参配置：RTK/IMU/APC lever arm + mount angle。
