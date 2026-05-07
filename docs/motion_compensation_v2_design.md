#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motion compensation V2 design for MyGPR."""

# 运动补偿 V2 重建方案

本文档用于替代当前分散的运动补偿 V1 思路。V2 的目标不是再加一个孤立算法，而是建立统一的 UAV-GPR 几何校正层：把 RTK、IMU、NAR15 高度计、雷达主 CSV 对齐到每道 trace，然后输出可被背景抑制、去噪、迁移、自动选参共同使用的 B-scan 与 `trace_metadata`。

## 为什么需要重建

当前 V1 已有价值，但不足以作为科研主线：

- `motion_compensation_height` 主要依赖 `flight_height_m`，且当前时移常数沿用 benchmark 的 `0.1 m/ns`，不应继续当作空气段物理波速。
- `motion_compensation_attitude` 主要返回元数据更新，不修改 B-scan，也没有统一接入高度计/RTK 质量。
- `motion_compensation_speed`、`trajectory_smoothing`、`vibration` 与高度/姿态模块分散，缺少一个总的质量判定和 provenance。
- 父项目目前采集线程能保存高度 sidecar，但 RTK、IMU、高度计和主 CSV 的统一 per-trace 合并还没有成为稳定数据契约。

## V2 输入契约

### 主数据

- `data`: `np.ndarray[samples, traces]`
- `header_info`:
  - `a_scan_length`
  - `num_traces`
  - `total_time_ns`
  - `trace_interval_m`（若可得）
  - `radar_center_frequency_hz`（若可得）

### 每道 trace 基础字段

- `trace_index`
- `trace_timestamp_s`
- `trace_distance_m`（可由采集间隔或 RTK 重建）

### RTK 字段

- `timestamp_s`
- `longitude`
- `latitude`
- `ellipsoidal_height_m` 或 `altitude_m`
- `rtk_fix_type`
- `satellites`
- `hdop`

RTK 高程语义必须标注清楚。默认只用于三维轨迹/地理参考，不直接作为 AGL。

### IMU/HWT905 字段

- `timestamp_s`
- `roll_deg`
- `pitch_deg`
- `yaw_deg`
- 可选：`gyro_x/y/z`、`accel_x/y/z`、`mag_x/y/z`

### 高度计/NAR15 字段

- `timestamp_s`
- `height_agl_m`
- `height_source`
- 可选：`snr`、`target_count`、`valid`

父项目现有 NAR15 快照字段 `distance_m` 可以规范映射为 `height_agl_m`。

### 安装几何字段

- `rtk_to_apc_offset_m`: RTK 天线参考点到雷达相位中心的 lever arm。
- `imu_to_apc_offset_m`: IMU 到雷达相位中心的 lever arm。
- `mount_yaw_offset_deg`、`mount_pitch_offset_deg`、`mount_roll_offset_deg`。
- `antenna_separation_m`（双天线/双站时）。

这些字段必须进入配置或校准文件，不能硬编码。

## V2 处理步骤

### 1. 时间轴统一

- 主 CSV 每道必须有 trace timestamp；若没有，按采集开始时间和采集间隔估计，但在 metadata 中标记为 `estimated`。
- RTK/IMU/高度计按 timestamp 插值到每道 trace。
- 超出 sidecar 时间范围的 trace 标记 `out_of_sensor_range`，不能静默外推。

### 2. 质量门控

- RTK：fix 类型、satellites、HDOP、跳变速度。
- IMU：roll/pitch/yaw 是否有限，姿态是否超阈值。
- 高度计：高度是否正数、跳变是否过大、有效目标数/SNR 是否可靠。
- 主数据：非有限值、饱和、坏道、异常能量。

质量门控输出 `motion_quality_flags`，供 GUI、CLI、导出报告使用。

### 3. 坐标与轨迹

- 经纬度投影到 local ENU/XY，原点使用第一条有效 RTK 或用户指定 survey origin。
- 计算 `platform_x_m/y_m/z_m`。
- 用姿态旋转 lever arm，得到 `apc_x_m/y_m/z_m`。
- 用 roll/pitch/yaw 与高度估计雷达地面足迹 `footprint_x_m/y_m`。
- 计算真实 `trace_distance_m` 和速度。

### 4. 离地高度融合

优先级：

1. 有效 NAR15/激光/高度计 AGL。
2. 雷达 air-ground interface 自动拾取高度。
3. 高度计与雷达拾取融合，并输出差异。
4. RTK altitude 只在存在地形模型/地面高程模型时换算 AGL。

V2 必须显式输出：

- `height_agl_m`
- `height_source`
- `height_confidence`
- `height_disagreement_m`（多源可用时）

### 5. 时移校正

- air-path 高度变化使用空气传播速度 `c0 = 0.299792458 m/ns` 的双程时间近似。
- 地下深度转换/迁移使用介质速度 `v_ground_m_per_ns` 或介电常数模型。
- V1 的 `wave_speed_m_per_ns=0.1` 只能作为历史 preset 的数值兼容，V2 不应沿用为默认 air-path 常数。

输出：

- `time_shift_ns`
- `time_shift_samples`
- `time_shift_clamped`
- `height_reference_m`

### 6. 振幅归一化

- 默认只做保守高度能量归一化，并设置最大增益/衰减 clamp。
- 报告中区分“几何归一化”和“显示增益”。
- 若高度数据置信度低，跳过振幅归一化，仅保留 warning。

### 7. 道距重采样

- 当后续算法要求均匀 trace spacing 时，将数据重采样到统一 `trace_distance_m`。
- 大间隙不得简单插值掩盖，应标记 gap。
- 重采样后必须同步更新所有 per-trace metadata。

### 8. 输出契约

V2 返回：

```python
corrected_data, meta = method_motion_compensation_v2(...)
```

`meta` 必须包含：

- `method`: `motion_compensation_v2`
- `skipped`
- `warnings`
- `quality_flags`
- `trace_metadata_updates`
- `display_trace_metadata`（若 GUI 需要展示未重采样坐标）
- `height_summary`
- `trajectory_summary`
- `attitude_summary`
- `provenance`

## GUI/CLI 行为

- GUI advanced sidecar 区应支持 RTK、IMU、高度计三个输入，而不是只有 RTK/IMU。
- CLI config 应支持：

```json
{
  "sidecars": {
    "rtk": "path/to/rtk.csv",
    "imu": "path/to/imu.csv",
    "altimeter": "path/to/height.csv"
  },
  "motion_compensation": {
    "calibration": "config/uav_sensor_mount_v1.json",
    "height_source": "auto",
    "resample_spacing_m": 0.05
  }
}
```

- 运行报告必须列出哪些传感器真正参与了补偿。

## 与父项目的对接

父项目 `D:\CDUT-UavGPR-Controller` 当前可参考：

- `src/lib/rtk_module.py`：RTK NMEA/GNGGA/GNRMC 存储字段。
- `src/lib/sensors/hwt905_module.py`：HWT905 roll/pitch/yaw、gyro、accel、mag、pressure height 解析。
- `src/lib/nar15_module.py`：NAR15 目标信息与 `distance_m` 高度快照。
- `src/lib/workers.py`：连续采集时按 trace 保存高度快照到 metadata JSON。
- `src/lib/acquisition_metadata.py`：当前 `lineData_metadata.json` 读取逻辑。
- `src/lib/gpr_processing_adapter.py`：父项目调用 MyGPR batch 的适配层。

下一步最好让父项目导出一个统一 sidecar，而不是让 MyGPR 猜多个文件关系：

```json
{
  "schema": "uav_gpr_trace_metadata_v2",
  "trace_timestamp_s": [],
  "rtk": {},
  "imu": {},
  "altimeter": {},
  "mount": {}
}
```

## 测试计划

### 单元测试

- RTK/IMU/高度计时间插值。
- RTK 经纬度到 local XY。
- lever arm + roll/pitch/yaw 足迹计算。
- AGL 高度优先级和 source/confidence。
- air-path time shift 使用 `c0`。
- 重采样后 metadata 长度一致。

### 集成测试

- 主 CSV + RTK sidecar + IMU sidecar + altimeter sidecar 完整运行。
- 缺 RTK 时仅高度补偿可运行并告警。
- 缺高度计时可从 radar horizon 估计或跳过高度补偿。
- sidecar 长度/时间范围不匹配时返回 warning，不静默错配。

### 基准与科研验证

- 使用 synthetic case：已知高度波动、姿态波动、目标位置，检查补偿后 air-ground interface 更平、目标位置误差更小。
- 使用无人机实测 CSV：比较人工 baseline 与 V2/auto-tune 后的 B-scan、评分、warnings。

## 分阶段实现

1. 新增 sidecar schema：支持 altimeter，统一字段名。
2. 新增 `core/motion_compensation_v2.py` 或 `PythonModule/motion_compensation_v2.py` 的纯后端实现。
3. 在 CLI 中接入 altimeter sidecar 和 calibration config。
4. 在 GUI advanced settings 中增加高度计 sidecar 与校准配置。
5. 将 V2 放入 `RECOMMENDED_RUN_PROFILES` 的 UAV 标准流程。
6. 自动选参 comparison 页默认使用 V2 标准流程。
7. 再考虑轨迹感知 SAR/back-projection 成像。

## 核心约束

- 不把 RTK altitude 当离地高度。
- 不把 V1 的 `0.1 m/ns` 当空气传播速度。
- 不静默外推 sidecar。
- 不只改 B-scan 不更新 metadata。
- 不只更新 metadata 不说明 B-scan 是否被真正校正。
- 每次补偿必须能导出 provenance，保证科研复现。
