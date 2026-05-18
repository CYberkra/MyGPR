#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运动补偿原子节点与 V2 统一设计文档
"""

# 运动补偿原子节点与 V2 统一设计

## 概述

本文档描述了 MyGPR 中运动补偿原子节点的当前设计，这些节点现在与 UAV 运动补偿 V2 共享核心逻辑，但保留了 V1 的代码用于兼容性和回归测试。

## 设计目标

1. **统一物理假设**：高度归一化、速度误差补偿、姿态/APC 足迹修正与 UAV 运动补偿 V2 使用同一套核心算法和物理假设
2. **保留兼容性**：旧的 V1 代码仍被保留，用于回归测试和旧的工作流
3. **简化 UI**：用户可见的节点只有普通的 5 个，没有 Legacy 选项
4. **自动切换逻辑**：根据输入数据自动在 V2 核心逻辑和 V1 兼容模式之间切换

## 核心共享模块

我们将核心共享逻辑提取到了 `core/motion_compensation_core.py`，包括：

### 共享帮助函数

- `_clone_metadata`: 安全克隆 trace_metadata
- `_field_1d`: 提取并验证一维字段
- `_numeric_field_or_none`: 安全提取数值字段
- `_warning`: 统一警告添加
- `_select_height`: 高度源选择（优先使用 height_agl_m）
- `_compute_reference_height`: 计算参考高度
- `_resolve_shift_sample_limit`: 解析时移样本限制
- `_apply_time_shift`: 应用时移校正
- `_compute_trace_distance`: 计算轨迹距离
- `_rotate_xy`: 坐标旋转
- `_build_attitude_updates`: 构建姿态更新
- `_resample_bscan_columns`: B-scan 列重采样
- `_metadata_for_output`: 准备输出元数据

### 公共 API 函数

- `apply_height_correction`: 高度归一化（V2 核心）
- `apply_attitude_apc_correction`: 姿态/APC 足迹修正（V2 核心）
- `apply_speed_resampling`: 速度误差补偿（V2 核心）

## 用户可见节点

用户在 UI 中看到的 5 个运动补偿节点：

1. **[运动补偿] 飞行高度归一化** (`method_motion_compensation_height`)
2. **[运动补偿] 速度误差补偿** (`method_motion_compensation_speed`)
3. **[运动补偿] 轨迹平滑** (`method_trajectory_smoothing`)
4. **[运动补偿] 姿态/APC 足迹修正** (`method_motion_compensation_attitude`)
5. **[运动补偿] UAV 运动补偿 V2** (`method_motion_compensation_v2`)

## 自动模式切换逻辑

原子节点的主函数会根据输入数据自动选择使用 V2 核心逻辑还是 V1 兼容模式。

### 高度归一化 (`motion_compensation_height.py`)

```python
use_v1 = False
if abs(wave_speed_m_per_ns - 0.1) < 1e-9:
    use_v1 = True
elif trace_metadata is None:
    use_v1 = True
else:
    has_flight_height = "flight_height_m" in trace_metadata
    has_agl_height = "height_agl_m" in trace_metadata
    if has_flight_height and not has_agl_height:
        use_v1 = True
    elif not has_flight_height and not has_agl_height:
        use_v1 = True
```

### 姿态/APC 足迹修正 (`motion_compensation_attitude.py`)

```python
use_v1 = False
if trace_metadata is None:
    use_v1 = True
else:
    has_height_agl = "height_agl_m" in trace_metadata
    has_flight_height = "flight_height_m" in trace_metadata
    has_roll = "roll_deg" in trace_metadata
    has_pitch = "pitch_deg" in trace_metadata
    has_yaw = "yaw_deg" in trace_metadata
    if has_flight_height and not has_height_agl:
        use_v1 = True
    elif not (has_roll and has_pitch and has_yaw):
        use_v1 = True
```

### 速度误差补偿 (`motion_compensation_speed.py`)

```python
use_v1 = False
if trace_metadata is None:
    use_v1 = True
elif interpolation_mode != "linear":
    use_v1 = True
elif "height_agl_m" not in trace_metadata:
    use_v1 = True
```

### 轨迹平滑 (`trajectory_smoothing.py`)

保持原实现不变，因为该方法是独立的平滑算法，与高度/速度/姿态的核心物理假设关系较小。

## 统一的物理假设

### 高度归一化的物理假设

1. **高度源优先级**：优先使用 `height_agl_m`，没有时回退到 `flight_height_m`
2. **空气传播速度**：使用真实物理值 `c0 = 0.299792458 m/ns`（不再使用旧 V1 的 `0.1 m/ns` 作为默认值）
3. **时移限制**：有安全钳制机制，防止过度时移
4. **质量标志**：输出 `quality_flags`，表示高度质量和校正状态

### 速度误差补偿的物理假设

1. **轨迹距离**：正确计算并更新 `trace_distance_m`
2. **元数据同步**：重采样后会同步更新所有 trace_metadata
3. **均匀道间距**：重采样到目标道间距

### 姿态/APC 足迹修正的物理假设

1. **Lever arm 应用**：正确应用 APC 偏移
2. **姿态旋转**：使用 roll/pitch/yaw 计算足迹
3. **距离更新**：更新 `trace_distance_m`

## 测试覆盖范围

我们运行了所有运动补偿相关的测试，共 42 个，全部通过：

- `tests/test_motion_compensation_v2.py` (13 个测试)：V2 总节点的完整功能
- `tests/test_motion_compensation_height.py` (11 个测试)：高度归一化的新旧行为
- `tests/test_motion_compensation_attitude.py` (7 个测试)：姿态/APC 修正
- `tests/test_motion_compensation_speed.py` (7 个测试)：速度误差补偿
- `tests/test_motion_runtime_metadata_contract.py` (4 个测试)：元数据契约

## 向后兼容性

### 保留的 V1 函数

- `method_motion_compensation_height_v1`: 高度归一化（V1 版本）
- `method_motion_compensation_speed_v1`: 速度误差补偿（V1 版本）
- `method_motion_compensation_attitude_v1`: 姿态/APC 修正（V1 版本）

### 触发 V1 兼容模式的场景

1. 当 trace_metadata 中只有 `flight_height_m` 而没有 `height_agl_m` 时
2. 当显式传入 `wave_speed_m_per_ns=0.1` 时（高度归一化）
3. 当缺少必要的姿态字段（roll_deg/pitch_deg/yaw_deg）时
4. 当 trace_metadata 为 None 时
5. 当插值模式不是 "linear" 时（速度补偿）

## 推荐使用方式

### 科研主线/论文

推荐使用 **UAV 运动补偿 V2** 总节点，因为它是完整的统一流程。

### 分项验证/教学/消融研究

可以使用四个原子节点，它们现在默认会使用 V2 核心逻辑（当有 height_agl_m 时），保持与 V2 总节点相同的物理假设。

### 回归测试/旧工作流

旧的 V1 函数仍可直接调用，但不会在 UI 中暴露，只用于代码层面的兼容性测试。
