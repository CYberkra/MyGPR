#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UAV-GPR processing flow research note."""

# UAV-GPR 标准处理流程（MyGPR 2026-05-07 版）

本文档用于固定 MyGPR 面向“无人机实测 CSV + RTK/IMU/高度计”的默认处理流程。结论不是把地面 GPR 流程照搬到 UAV，而是把 UAV 的几何误差、姿态误差和高度变化作为前置约束纳入流程。

## 结论

你原计划的流程“运动补偿、零时校正、低频漂移抑制、背景抑制、增益、去噪、成像/迁移”方向基本正确，但顺序需要拆分和修正：

1. “运动补偿”不能只放在最前面作为一个黑盒。它应拆成两层：先做传感器对齐/轨迹建模，再在时零和必要的轻量预处理后做高度/姿态/轨迹补偿。
2. “零时校正”应早于严格的高度时移和深度解释。文献和商业流程都强调 time-zero 对深度轴正确性是基础。
3. “增益”不应默认早于主要降噪/背景抑制。增益会放大后期噪声，应区分“物理归一化/高度能量校正”和“显示或传播衰减补偿增益”。
4. UAV 场景中，RTK 高程不能直接等同于离地高度；离地高度应优先来自高度计、雷达 air-ground interface 拾取，或二者融合。
5. UAV 轨迹通常不是理想匀速直线采样，后续迁移/成像必须知道真实 trace 坐标，或先重采样到均匀沿线距离。

## 推荐默认流程

### 0. 数据载入与不可变原始数据

- 输入：无人机实测主 CSV、可选 RTK CSV、IMU CSV、NAR15/高度计 sidecar。
- 保留原始数据快照，不在原数组上原地修改。
- 建立 `header_info`：样点数、道数、时窗 ns、采样间隔、雷达配置。
- 建立 `trace_metadata`：每道 timestamp、trace_index、初始 trace_distance_m。

### 1. 采集质量控制与传感器对齐

- 校验主 CSV 道数、样点数、非有限值、重复/缺失道。
- RTK/IMU/高度计统一到秒级时间轴，并插值到每条 trace 的时间戳。
- RTK 经纬度投影到本地 ENU/XY；保留 RTK fix、卫星数、HDOP 等质量字段。
- IMU roll/pitch/yaw 与可选 gyro/accel 对齐到每道。
- 高度字段统一为 `flight_height_m`，语义必须是 AGL（above ground level），不能把 RTK altitude 自动当 AGL。
- 输出：对齐后的 `trace_metadata` 与质量告警。

### 2. 原始信号校正

- 坏道/异常道检测：缺失、饱和、热像素、极端能量、明显采集失败。
- DC shift / 基线偏置检查。
- time-zero correction：基于阈值、峰值、first-break 或校准值，将每道零时对齐。
- 对 UAV 数据，若需要从雷达图中拾取 air-ground interface，应在保留界面反射的轻量预处理版本上做，不要先用强背景消除抹掉它。

### 3. 低频漂移与轻量频带控制

- Dewow / 低频漂移抑制：去除低频 wow 或 DC bias。
- 必要时做温和 band-pass 或低通/高通，用于稳定后续拾取；避免过早强滤波导致目标形态失真。

### 4. UAV 几何/运动补偿

这一阶段是 MyGPR 后续重点，建议称为 `motion_compensation_v2`，不要继续把高度、速度、姿态分散成彼此独立的最终算法。

- 轨迹补偿：由 RTK/ENU 与 trace timestamp 构造每道真实位置，识别不均匀采样和速度突变。
- 姿态/APC 补偿：用 roll/pitch/yaw 和天线相位中心 lever arm 计算天线实际位置/足迹。
- 高度补偿：用高度计或 radar-picked ground interface 做 AGL 对齐，修正 air-path time shift。
- 道距重采样：当后续算法假设等距 B-scan 时，重采样到均匀 `trace_distance_m`；同时保留原始坐标和重采样 provenance。
- 质量输出：标记缺传感器、时间不同步、RTK fix 差、高度离群、姿态超限、重采样大间隙。

### 5. 背景/杂波抑制

- 平均/中值背景扣除、SVD 背景、CCBS、F-K 等方法在运动补偿后更可靠，因为水平条纹和 air-ground interface 已尽可能对齐。
- 对 UAV 数据建议支持“两次背景”策略：
  - 初次温和背景：用于提升高度拾取或稳定 air-ground interface。
  - 几何补偿后二次背景：用于正式抑制天线耦合、空气-地面强界面和水平杂波。

### 6. 去噪与目标保持

- Hankel-SVD、Wavelet、Wavelet-SVD、SVD-subspace、滑动/中值类去噪。
- 去噪必须有保真约束：不能只追求图像干净，还要保留局部异常、双曲线 apex、层位边界和 first-break 稳定性。
- 自动选参评分应显式惩罚过度平滑、目标能量丢失和局部显著性下降。

### 7. 增益

- 高度振幅归一化属于运动补偿阶段的物理/工程校正。
- 显示增益、SEC、AGC、power/exponential gain 应在主要杂波/噪声处理之后使用。
- 需要输出“用于解释/导出”的数据和“用于显示”的增益数据，避免把显示增强误认为物理幅度改善。

### 8. 成像、迁移与深度转换

- 若只做 2D B-scan 解释：可使用 Stolt/Kirchhoff migration，但必须满足等距 trace、合理速度模型、时零正确。
- 若使用 UAV 真实非直线轨迹：优先规划 SAR back-projection / beamforming 或能吃任意轨迹坐标的成像方法。
- time-to-depth 需要速度模型；空气段用近似光速，地下段用介质速度或介电常数模型，不能沿用 V1 的 `0.1 m/ns` 作为通用 air-path 常数。

### 9. 科研证据导出

- 每次处理链应导出：输入数据 hash、流程版本、每步参数、传感器质量摘要、warnings、ROI、评分、输出 B-scan 图。
- 自动选参与人工 baseline 对比必须绑定同一输入、同一 ROI、同一显示尺度和同一导出配置。

## MyGPR 默认 profile 建议

### 快速预览

`set_zero_time -> dewow -> motion_compensation_v2(light) -> background -> display_gain`

用于现场快速确认数据是否可用，不作为最终科研图。

### 稳健科研 B-scan

`QC/sidecar alignment -> set_zero_time -> dewow -> motion_compensation_v2 -> background/SVD/CCBS -> bandpass/F-K -> denoise -> SEC/AGC display gain -> optional migration -> export`

这是 MyGPR 的默认研究链。

### UAV-SAR / 轨迹成像

`QC/sidecar alignment -> positioning/pose processing -> time-zero/window -> background -> height correction -> second background/SVD -> antenna-position calculation -> SAR back-projection or trajectory-aware migration -> export`

该链是后续高级成像目标，不应被当前 V1 的普通 B-scan 迁移替代。

## 当前 MyGPR/父项目现状

- 父项目已有 RTK 模块、HWT905 九轴姿态计、NAR15 高度计。
- 父项目连续采集线程当前已能按道保存 `flight_height_m`、`height_source`、`height_timestamp_s` 到 metadata JSON。
- MyGPR 已有 RTK/IMU sidecar 解析、trace metadata 对齐、`motion_compensation_height/speed/attitude/vibration` 等 V1 实验模块。
- 当前缺口是：V1 模块分散、物理语义不统一、RTK/IMU/高度计没有形成一个单一可信的 `motion_compensation_v2` 运行契约。

## 权威依据

- RGPR 基础教程列出常见 GPR 处理步骤：first break/time-zero、DC shift、time-zero correction、dewow、frequency filter、gain、spatial/f-k/background 等，并强调处理历史应可追踪以支持可复现研究：<https://emanuelhuber.github.io/RGPR/02_RGPR_tutorial_basic-GPR-data-processing/>
- ERDC/CRREL 2022 自动化 GPR 后处理报告将标准化流程概括为 static data removal、time-zero correction、distance normalization、data filtering、stacking，并以 GSSI 软件对照验证：<https://erdc-library.erdc.dren.mil/items/c95bac4f-900c-4218-a7b7-3e5b7b25c38b>
- Geolitix GPR processing 文档强调 time-zero 对深度正确性必需，air-launched/drone GPR 的高度变化不是简单 time-zero，应做 horizon/height flatten；同时说明 background、dewow、gain、migration 的用途：<https://docs.geolitix.com/layers/gpr-processing.html>
- Alani 等 2019 road GPR signal-processing review 给出典型流程：raw signal correction、time-zero、elevation correction、energy normalization、dewow、ringing/background、gain、band-pass、migration/depth conversion，并提醒 gain 通常应在若干去噪步骤之后使用：<https://www.mdpi.com/2076-3263/9/2/96>
- Garcia-Fernandez 等 2019 UAV UWB-GPR 系统论文说明 UAV-GPR 需要高精度 RTK、IMU、laser rangefinder，并在处理链中进行定位数据处理、time-zero/window、background、height correction、second background、SAR processing：<https://www.mdpi.com/2072-4292/11/20/2357>
- Noviello 等 2020 小型 UAV 雷达成像论文指出准确 UAV 定位能避免 defocusing 和 localization errors，轨迹质量取决于机载导航传感器和地面辅助系统：<https://www.mdpi.com/2072-4292/12/20/3463>
- Garcia-Fernandez 等 2022 UAV-GPR-SAR 改进研究指出 height information、antenna tilt、SVD clutter filtering 都会影响聚焦和检测；RTK height 不应直接视为离地高度：<https://www.sciencedirect.com/science/article/pii/S0924271622001113>
- Catapano 等 2022 down-looking UAV-GPR overview 总结 UAV-GPR 面临非均匀轨迹、平台动态、运动补偿、RTK/PPK/IMU/LIDAR 高度等问题，并指出标准迁移算法常假设直线等距轨迹，UAV 场景常需 MoCo 或 SAR back-projection：<https://www.mdpi.com/2072-4292/14/14/3245>
- Bähnemann 等 2021 airborne GPSAR landmine 论文强调 airborne GPR 图像质量依赖精确 motion estimation，并使用 dual RTK GNSS + IMU 融合获得高频位置/姿态：<https://arxiv.org/abs/2106.10108>
