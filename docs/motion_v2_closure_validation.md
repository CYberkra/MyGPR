#!/usr/bin/env markdown
# Motion V2 收口可用性验证

## 结论

基于 `output/mygpr_uav_motion_effect_demo_v1`，四个用户可见原子运动补偿步骤与统一入口
`motion_compensation_v2` 均能读取同一套 UAV-GPR sidecar，并产生可见、可量化的补偿效果。

本次验证对象是合成演示数据，用于确认软件链路、元数据契约、三维预览和 Evidence 导出可用；
它不能替代真实外业数据的地质结论。

## 复现命令

```bash
python scripts/validate_motion_v2_closure.py ^
  --dataset output/mygpr_uav_motion_effect_demo_v1 ^
  --output output/motion_v2_closure_validation
```

## 验证流程

1. 读取 `main.csv`、`rtk.csv`、`imu.csv`、`altimeter.csv`。
2. 通过 `trace_timestamp_s` 对齐辅助传感器，确认 RTK/IMU/高度计没有被忽略。
3. 执行四个原子步骤：
   `trajectory_smoothing -> motion_compensation_attitude -> motion_compensation_speed -> motion_compensation_height`。
   姿态/APC 足迹更新必须先于等距道距重采样，避免后续姿态更新破坏 speed compensation 建立的等距 trace axis。
4. 执行统一入口 `motion_compensation_v2`。
5. 对比 B-scan、trace metadata、三维预览 payload 和回放 Evidence 包。

## 关键指标

| 指标 | Raw | 四原子步骤 | Motion V2 |
| --- | ---: | ---: | ---: |
| 顶部界面抖动 std / sample | 0.288 | 0.186 | 0.171 |
| 道间距抖动 std / m | 0.108801 | 0.014143 | 0.008717 |

补充指标：

- Raw -> 四原子步骤 B-scan RMS 差异：0.073814
- Raw -> Motion V2 B-scan RMS 差异：0.075112
- 四原子步骤 -> Motion V2 B-scan RMS 差异：0.016069

## 契约检查

- `height_agl_m` 已加载。
- 四原子步骤输出的 `trace_metadata` 长度与 B-scan trace 数一致。
- `motion_compensation_v2` 输出的 `trace_metadata` 长度与 B-scan trace 数一致。
- `motion_compensation_v2` 输出 `footprint_x_m`、`footprint_y_m`、`trace_distance_m`。
- 高度补偿使用空气传播速度 `0.299792458 m/ns`。
- 原始、四原子步骤、Motion V2 三维预览均可生成。
- 回放 Evidence 包包含 motion 专用 B-scan、3D preview、quality flags 和 motion params。

## 输出位置

本次生成产物位于：

`output/motion_v2_closure_validation/mygpr_uav_motion_effect_demo_v1/`

重要文件：

- `motion_v2_closure_report.md`
- `motion_v2_closure_summary.json`
- `motion_v2_closure_bscan_comparison.png`
- `raw_3d_preview.png`
- `atomic_3d_preview.png`
- `motion_v2_3d_preview.png`
- `motion_v2_replay_evidence.zip`

## 当前限制

- 当前数据是合成演示数据，目标是让运动补偿效果可见，不代表真实外业泛化能力。
- Motion V2 当前仍是 V2.1 收口版本，不包含完整 V3 backlog 中的外参标定、时钟漂移、RTK 硬门控和轨迹感知迁移。
- 后续应使用真实 RTK/IMU/高度计数据复验同一脚本和同一 Evidence 导出链路。
