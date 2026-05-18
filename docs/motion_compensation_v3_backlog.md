# Motion Compensation V3 Backlog

## 结论

本轮不实现完整 V3。当前工程继续把 `motion_compensation_v2` 作为推荐统一入口；四个用户可见原子运动补偿节点共享 V2 物理假设，用于分项验证、ablation 和调试；`motion_compensation_vibration` 已从运动补偿核心链移出，定位为周期条带伪影抑制实验功能。

## V3 暂缓项

- 外参标定：RTK 天线、IMU、雷达主机、Tx/Rx 相位中心之间的 lever arm 与 mount angle。
- Tx/Rx 分离：双天线或等效相位中心间距对目标位置、双曲线 apex 和迁移的影响。
- 时钟误差：雷达、RTK、IMU、高度计之间的 clock offset 与 drift 估计。
- RTK 质量门控：fix type、satellite count、HDOP、轨迹跳变速度和加速度异常的硬门控。
- Radar horizon 高度估计：缺少高度计时，从 air-ground reflection 估计 AGL 的 fallback。
- 轨迹感知迁移：将补偿后的三维航迹、姿态和高度模型接入迁移/成像，而不是只更新 B-scan 与 metadata。

## 当前边界

- `motion_compensation_v2`：当前推荐入口，负责高度时移、振幅归一化、姿态/APC 足迹元数据、等距道距重采样和质量告警。
- 原子运动补偿节点：`trajectory_smoothing`、`motion_compensation_attitude`、`motion_compensation_speed`、`motion_compensation_height`，共享 V2 的 AGL 优先级、空气路径速度、clamp、warnings 和 trace metadata 输出契约。验证/ablation 顺序应先更新姿态/APC 足迹，再做等距道距重采样，最后做高度时移与振幅归一。
- `motion_compensation_vibration`：周期条带伪影抑制（实验），可用于高级增强或去噪验证，不再作为运动补偿核心能力证明。

## 进入 V3 前置条件

- 至少一套真实 CSV + RTK + IMU + 高度计同步数据。
- gprMax 高度变化场景能稳定输出运动补偿前后 benchmark。
- 质量指标覆盖地表反射平直度、目标 ROI 保真度、apex 稳定性、轨迹间距变化和时移 clamp 比例。
- V3 每一项能力都有独立测试和可复现实验报告，不与 UI 或 AutoTune 大改混在同一提交。
