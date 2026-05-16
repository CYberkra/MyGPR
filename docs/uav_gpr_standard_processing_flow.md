# UAV-GPR 标准处理流程（MyGPR 2026-05-10 最终基线）

本文档固定 MyGPR 面向 UAV-GPR 的默认数据处理流程。这里的 UAV-GPR 指无人机平台搭载 GPR，并可接入 RTK、IMU、NAR15/激光/毫米波高度计等导航与姿态传感器的数据处理场景。

结论：MyGPR 不应照搬地面 GPR 的普通“零时、去漂移、背景、增益、去噪、迁移”链路。UAV 场景的核心难点是飞行轨迹不均匀、离地高度变化、姿态变化、传感器同步误差和空气-地表强界面，因此最终流程必须把“导航/轨迹/高度/姿态”作为前置几何基准，而不是把运动补偿当成一个普通滤波按钮。

## 一句话流程

`数据导入/QC -> RTK/IMU/高度计同步与轨迹建模 -> 零时/坏道/DC 等基础校正 -> dewow/温和频带控制 -> UAV 几何与运动补偿 -> 背景/杂波抑制 -> 保真去噪 -> 解释增益/显示增益 -> 迁移或轨迹感知成像 -> 可复现实验导出`

## 当前软件预设对应关系

MyGPR 现在保留两类内置模板，不应混用它们的定位：

- `mygpr_standard` / `MyGPR 标准流程`：原 MyGPR 经典五步链，顺序固定为 `零时校正 -> dewow/低频漂移 -> 平均背景抑制 -> SEC 增益 -> SVD 子空间去噪`。它用于快速复现旧版处理习惯和用户熟悉的基础流程。
- `high_quality_uav_gpr` / `高质量 UAV-GPR`：面向项目真实 UAV-GPR SFCW CSV 的完整流程，包含 DC 去偏、频带控制、运动补偿、速度模型和几何-深度上下文。它是后续科研链和商用默认链的主要基础。

当前 Workflow Studio 的执行语义仍是“按节点顺序执行”。画布连线用于可视化、每步 Preview/Evidence 和一致性检查，暂不作为 DAG 执行顺序。

## 最终默认流程

### 0. 数据导入与原始快照

- 输入主 B-scan CSV、雷达头信息、RTK sidecar、IMU sidecar、NAR15/高度计 sidecar。
- 原始数据必须不可变保存，不允许在 raw array 上原地修改。
- 建立基础元数据：样点数、道数、时间窗、采样间隔、雷达中心频率、天线间距、采集时间、输入文件 hash。
- 建立每道 `trace_metadata`：trace index、timestamp、初始 distance、传感器插值状态、质量标记。

### 1. 采集质量控制与传感器同步

这一阶段是 UAV-GPR 与普通地面 GPR 最大的分界。它主要处理元数据和轨迹，不应大幅修改雷达幅值。

- 检查主数据：空道、坏道、重复道、非有限值、饱和道、异常能量道、静止采样段。
- 将 RTK、IMU、高度计统一到同一时间轴，按 trace timestamp 插值到每一道。
- RTK 经纬高转换到本地 ENU/XY 坐标；保留 fix quality、卫星数、HDOP、差分状态等质量字段。
- IMU roll/pitch/yaw、角速度、加速度插值到每一道。
- 高度字段统一为 AGL（above ground level）。RTK altitude 不能自动等同为离地高度；优先使用 NAR15/激光高度计、地表界面拾取，或二者融合。
- 计算天线相位中心 APC：用 RTK 天线、IMU、GPR 天线之间的 lever arm 加上姿态旋转，得到每道 GPR 天线实际位置。
- 生成质量 mask：RTK 弱、IMU 跳变、姿态超限、高度离群、时间不同步、道距过大、静止/转弯过采样。

### 2. 雷达基础校正

这一阶段修正“雷达记录本身”的基础问题，是后续几何补偿和深度解释的前提。

- 删除或标记坏道、空道、静止道；必要时保留被剔除 trace 的 provenance。
- DC shift / baseline offset 校正。
- time-zero correction：基于校准值、阈值、峰值、first-break 或 air-ground interface，把系统延迟和无效早期时间移除。
- time window / range gate：只裁掉明确无效或超出研究目标的时间段，避免过大零时参数直接切掉有效数据。
- 对 UAV 下视数据，air-ground interface 的拾取应在“轻量预处理版本”上做，不能先用强背景消除把地表界面抹掉。

### 3. 低频漂移与温和频带控制

- Dewow / 低频漂移抑制，用于消除 wow、基线扭曲和低频拖尾。
- 可选 `frequency_filter_1d`：band-pass / low-pass / high-pass / notch。截止频率必须由采样率、时间窗和天线频带约束，不能凭经验写死。
- 此阶段只能做温和稳定化，不应使用会明显改变目标形态的强滤波。

### 4. UAV 几何与运动补偿

这是 MyGPR 的核心 UAV-GPR 专用阶段，建议统一落到 `motion_compensation_v2` 及其后续版本，而不是分散成互不知情的高度、速度、姿态小算法。

- 轨迹补偿：基于 RTK/ENU 与 timestamp 计算每道真实位置、速度、航向、道距，识别非均匀采样和转弯过采样。
- 姿态/APC 补偿：用 roll/pitch/yaw 与 lever arm 计算 GPR 天线相位中心位置，而不是直接把 RTK 天线位置当作雷达位置。
- 高度补偿：用 AGL 高度或雷达地表界面估计修正空气段 travel-time，使 air-ground interface 在合理基准上对齐。
- 高度幅值归一化：只补偿由离地高度变化造成的空气传播和耦合差异；它属于几何/物理校正，不等同于后面的显示增益。
- 道距重采样：当后续 B-scan、F-K、传统 migration 假设等距采样时，重采样到均匀 `trace_distance_m`；同时保留原始坐标和重采样来源。
- 输出 warning：缺传感器、时间错位、RTK 质量差、高度缺失、姿态超限、重采样间隙过大、可疑过补偿。

### 5. 背景与杂波抑制

正式背景抑制应放在几何/高度对齐之后，因为 UAV 高度变化会让 air-ground interface 和水平杂波上下漂移，过早强背景会造成错误扣除。

- 常规方法：平均/中值背景扣除、滑动背景、SVD 背景、CCBS。
- 方向性杂波：F-K / 2D FFT，用于压制斜向空气波、平台振动纹理或规则干扰。
- 推荐两阶段策略：
  - 轻量背景：可在高度拾取前使用，只为增强界面可拾取性。
  - 正式背景：运动补偿后执行，用于抑制天线耦合、地表强界面、水平条纹和重复背景。

### 6. 去噪与结构保真

- 可选方法：Hankel-SVD、Wavelet、Wavelet-SVD、SVD-subspace、局部中值/均值类滤波。
- 目标不是“越干净越好”，而是提升 SNR 同时保留双曲线 apex、层位连续性、裂缝/空洞边缘、局部异常体反射和弱深部目标。
- 自动选参评分必须惩罚过度平滑、有效反射能量丢失、异常体边缘扩散和深部噪声过曝。

### 7. 增益

增益分为“解释/导出用增益”和“显示增强用增益”，二者必须在软件中区分。

- 默认解释链：优先 SEC、energy-decay、power/exponential 这类随时间/深度变化但相对可解释的增益。
- AGC：适合快速查看弱反射、报告截图和视觉增强，但会破坏相对幅值信息，不应作为默认科研幅值链的唯一结果。
- 增益应在主要背景抑制和去噪之后执行，否则会放大本该先压制的噪声、地表强界面和平台干扰。
- 对比报告中可以并列输出 SEC、AGC、无增益/常数增益结果，但默认导出的解释数据应保留增益方法和参数标记。

### 8. 迁移、成像与深度转换

- 普通 2D B-scan 解释：只有在 trace 等距、time-zero 可信、速度模型明确时，才使用 Stolt/Kirchhoff/F-K migration。
- UAV 真实非直线轨迹：优先使用能处理任意轨迹坐标的 SAR back-projection / delay-and-sum / beamforming，或先做严格轨迹重采样后再迁移。
- 深度转换必须使用介质速度或相对介电常数模型。空气段近似光速，地下段不能沿用固定经验速度作为所有场景默认值。
- 成像输出应记录速度模型、介电常数、轨迹来源、APC 参数、是否等距重采样。

### 9. 导出与科研证据链

- 每步导出 before/after B-scan、参数、评分、warning、ROI、输入 hash、流程版本。
- 自动选参与人工 baseline 对比必须使用同一输入、同一 ROI、同一显示尺度、同一真值区域定义。
- 对 gprMax 正演数据，必须记录真实结构、目标区、背景区、噪声区，并输出 manual vs auto 的结构保持、背景抑制、深部补偿、目标显著性指标。
- gprMax 论文级验证默认使用 `airborne_*` 场景族，不再使用贴地 toy 场景作为 UAV-GPR 证据。每个 airborne 场景必须显式记录空气层、天线离地高度、Tx/Rx 航迹、直达波、air-ground 反射、地下目标/背景 ROI 和晚时窗噪声 ROI。
- `airborne_height_variation_cylinder_v1` 必须逐道定义 Tx/Rx 高度，并同步输出 `trace_timestamps.csv`、`rtk.csv`、`imu.csv`、`altimeter.csv`，用于后续运动补偿验证。该场景不能用 `#src_steps/#rx_steps` 伪装非线性高度变化；当前默认用 gprMax `#python` + `current_model_run` 在单个 `.in` 中逐道定义高度，便于使用 `-n`、`-restart` 和未来 MPI task farm。
- MyGPR 的质量页现在增加了三维地理参考预览，按“航迹 + 剖面带”输出 `VTK + CSV + JSON`，用于把轨迹、飞行高度和剖面空间关系一起保留下来；这不是完整体素重建，但足以作为 UAV-GPR 的地理参考证据和导出备份。

## MyGPR 默认实现链

当前 MyGPR 的“高质量 UAV-GPR”默认链应对齐为：

`set_zero_time -> dewow -> frequency_filter_1d -> motion_compensation_v2 -> subtracting_average_2D / SVD / CCBS -> fk_filter -> wavelet_svd / Hankel-SVD -> sec_gain -> optional migration / trajectory-aware imaging -> export`

说明：

- `motion_compensation_v2` 之前允许做 time-zero、dewow、温和频带控制，因为这些步骤稳定波形和零时，不会破坏轨迹语义。
- 强背景抑制、强去噪、增益都应在 `motion_compensation_v2` 之后。
- 当只有 CSV、没有 RTK/IMU/高度计时，流程仍保持同一顺序，但 `motion_compensation_v2` 应降级为 warning + no-op、地表界面估计，或仅做可解释的高度/道距补偿。
- `motion_compensation_v2` 的默认高度时移限幅应使用物理时间窗 `max_shift_ns` 自动换算为样点数，避免同一固定 sample 阈值在真实 SFCW 数据和高采样率 gprMax 数据中含义完全不同。
- 运动补偿输出必须保留时间戳 gap、道距 gap、速度异常、低高度置信度、外推对齐等风险提示，供 GUI 质量页、证据包和论文报告复核。
- AGC 作为显示增强或报告对比项保留；SEC/energy-decay 类方法作为默认解释增益方向。

MyGPR 默认解释链使用 SEC/energy-decay 风格增益，AGC 保留为显示增强和对比分支。针对 gprMax 验证和未来真实 UAV 数据，增益族选择应通过 `core.gain_selection` 在 SEC、AGC、线性/手动 TGC、无增益之间评分，评分依据包括目标保持、背景抑制、假异常、过曝、热点和相对幅值保真，而不是固定相信某一种增益永远更好。

## 为什么这就是最终基线

- Geolitix 将 set time zero、remove idle/empty traces、trace shifting、resample traces equidistantly 固定在处理列表顶部，因为这些步骤影响 positioning；它还明确提到 drone GPR 的高度变化应通过 horizon/height flatten 处理，而不是简单 time-zero。
- UAV-GPR 综述指出 UAV 飞行轨迹常表现为非直线、变速、离地高度变化，导致非均匀测点和散焦；标准 migration 对直线等距轨迹有假设，而 back-projection 可以处理任意轨迹和非均匀间距。
- UAV UWB-GPR/SAR 实测系统的处理链将定位数据处理、time-zero/window、background、height correction、second background、APC 计算、SAR processing 串联起来；这支持我们把传感器同步/轨迹建模放在雷达增强处理之前。
- 传统 GPR 处理资料也支持 time-zero、DC、dewow、频带控制、增益、空间/F-K/背景、迁移等步骤，但这些资料不处理 UAV 的姿态、高度和轨迹问题，因此只能作为信号处理子链参考。

## 参考依据

- Geolitix GPR Processing：<https://docs.geolitix.com/layers/gpr-processing.html>
- Geolitix GPS Positioning：<https://docs.geolitix.com/layers/gpr-position-gps.html>
- RGPR Basic GPR data processing：<https://emanuelhuber.github.io/RGPR/02_RGPR_tutorial_basic-GPR-data-processing/>
- ERDC/CRREL TR-22-18 自动化 GPR 后处理：<https://erdc-library.erdc.dren.mil/items/c95bac4f-900c-4218-a7b7-3e5b7b25c38b>
- Garcia-Fernandez 等，UWB-GPR on Board a UAV for Landmine and IED Detection：<https://www.mdpi.com/2072-4292/11/20/2357>
- Noviello/Catapano 等，An Overview on Down-Looking UAV-Based GPR Systems：<https://www.mdpi.com/2072-4292/14/14/3245>
