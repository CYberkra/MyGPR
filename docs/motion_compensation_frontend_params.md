# Motion Compensation Frontend Parameters

This note records the motion-compensation parameters that must remain visible in the MyGPR frontend. The source of truth for callable parameters is `core/methods_registry.py`; workflow defaults come from `core/workflow_data.py`.

## motion_compensation_height

Method label: 飞行高度归一化

| Parameter | UI label | Source | Frontend tier | Default | Passed to backend | Notes |
|---|---|---|---|---:|---|---|
| `reference_height_mode` | 参考高度 | `methods_registry.py` | Common | `mean` | Yes | Choice: `mean`, `min`, `manual`. Render as a dropdown. |
| `manual_height` | 手动参考高度 (m) | `methods_registry.py` | Common | `10.0` | Yes | Enabled only when `reference_height_mode=manual`; otherwise disabled to avoid implying it is active. |
| `compensate_amplitude` | 振幅校正 | `methods_registry.py` | Common | `True` | Yes | Render as a checkbox/switch. |
| `compensate_time_shift` | 时移校正 | `methods_registry.py` | Common | `True` | Yes | Render as a checkbox/switch. |
| `wave_speed_m_per_ns` | 传播速度假设 (m/ns) | `methods_registry.py` | Common | `0.1` | Yes | Critical for height time-shift. UAV air-path demos should use `0.299792458`; soil-equivalent velocity should be interpreted carefully. |

Recommended demo parameters for UAV air-path height shift:

```json
{
  "reference_height_mode": "mean",
  "compensate_amplitude": true,
  "compensate_time_shift": true,
  "wave_speed_m_per_ns": 0.299792458
}
```

## motion_compensation_v2

Method label: UAV 运动补偿 V2

| Parameter | UI label | Source | Frontend tier | Default | Passed to backend | Notes |
|---|---|---|---|---:|---|---|
| `height_reference_mode` | 参考高度 | `methods_registry.py` / `workflow_data.py` | Common | `mean` | Yes | Dropdown: `mean`, `min`, `manual`. |
| `height_source` | 高度来源 | `methods_registry.py` / `workflow_data.py` | Common | `auto` | Yes | Dropdown: `auto`, `height_agl_m`, `flight_height_m`. |
| `compensate_time_shift` | 高度时移校正 | `methods_registry.py` / `workflow_data.py` | Common | `True` | Yes | Checkbox/switch. |
| `compensate_amplitude` | 高度振幅归一 | `methods_registry.py` / `workflow_data.py` | Common | `True` | Yes | Checkbox/switch. |
| `max_shift_samples` | 最大时移样点 (0=按时间窗) | `methods_registry.py` / `workflow_data.py` | Common | `0.0` | Yes | Numeric input with registry range validation. |
| `max_shift_ns` | 最大时移时间 (ns) | `methods_registry.py` / `workflow_data.py` | Common | `20.0` | Yes | Numeric input with registry range validation. |
| `max_amplitude_scale` | 最大振幅倍率 | `methods_registry.py` / `workflow_data.py` | Common | `2.0` | Yes | Numeric input with registry range validation. |
| `resample_spacing_m` | 等距重采样间距 (m) | `methods_registry.py` / `workflow_data.py` | Common | `0.0` | Yes | `0` disables equal-distance resampling. |
| `apc_offset_x_m` | APC X偏移 (m) | `methods_registry.py` / `workflow_data.py` | Common | `0.0` | Yes | Numeric input. |
| `apc_offset_y_m` | APC Y偏移 (m) | `methods_registry.py` / `workflow_data.py` | Common | `0.0` | Yes | Numeric input. |
| `apc_offset_z_m` | APC Z偏移 (m) | `methods_registry.py` / `workflow_data.py` | Common | `0.0` | Yes | Numeric input. |
| `max_abs_tilt_deg` | 最大姿态角 (deg) | `methods_registry.py` / `workflow_data.py` | Common | `20.0` | Yes | Numeric input. |

For `sample_data/uav_gpr_motion_demo_v3_clear_effect/main.csv` or equivalent clear-effect demos, prefer the dataset manifest's recommended V2 parameters. The current v2 API already uses `air_wave_speed_m_per_ns=0.299792458` internally for air-path height correction, so the frontend focus is exposing the remaining V2 controls consistently.
