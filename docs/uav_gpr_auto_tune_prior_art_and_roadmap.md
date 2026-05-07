#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UAV-GPR/GPR auto-tune prior-art report and MyGPR roadmap."""

# UAV-GPR/GPR 自动选参前沿调研与 MyGPR 长期路线

检索日期：2026-05-07

## 0. 执行结论

本报告把“自动选参”限定为：软件或算法根据输入 GPR 数据、元数据、ROI 或 ground truth 指标，自动选择处理方法或处理参数；不把“软件内置默认值”“用户套用固定 recipe”“AGC 这种算法名称里的 automatic”直接等同为自动选参。

截至本次公开资料检索，**没有发现一套公开论文或开源软件明确完成了 MyGPR 目标中的 UAV-GPR 全流程、可解释、可导出证据的自动选参系统**。最接近的公开产品线索是 [Geolitix GPR Processing](https://docs.geolitix.com/layers/gpr-processing.html)：它明确声称能分析输入 GPR 数据并自动生成处理方案，且提到 drone GPR/air-launched antenna 的高度变化处理。但 Geolitix 同时要求用户设置 radar velocity，且公开文档没有披露完整算法、候选空间、评分函数、ground-truth 评估或人工/自动参数对比报告，因此更适合归为“产品级黑盒自动处理”，不能直接视为公开可复现的科研自动选参方案。

UAV-GPR 文献目前主要集中在平台集成、RTK/IMU/高度控制、SAR 成像、运动误差补偿、目标检测和应用验证。例如 UAV-GPR 综述指出 UAV 飞行速度、高度和非直线轨迹会造成图像失焦与定位误差，需要硬件定位和信号处理补偿；但该综述描述的是“标准处理和高级成像算法”，不是逐算法参数自动优化框架（[Remote Sensing 2022 UAV-GPR overview](https://www.mdpi.com/2072-4292/14/14/3245)）。普通 GPR 领域则已经有很多“局部自动化”：自动 first-break/time-zero、自动/半自动速度估计、PSO-VMD 去噪参数搜索、adaptive clutter removal、自动目标检测和深度学习识别等。这些是 MyGPR 可以吸收的模块，但大多停留在单算法或单任务层面。

因此，MyGPR 的长期论文级机会不是简单宣称“我们也有自动选参”，而是做成：**面向 UAV-GPR 标准处理链的、数据尺度受约束的、多目标可解释自动调参框架，并用 gprMax ground truth + 人工 baseline + 真实外业数据三层证据验证**。这条路线有较明确的创新空间。

## 1. 自动选参分级

| 等级 | 定义 | 代表情况 | 是否接近 MyGPR 目标 |
| --- | --- | --- | --- |
| L0 | 固定默认参数或固定 recipe | 软件预设、教程参数、经验参数表 | 否，只是 baseline |
| L1 | 单参数或单步骤自动估计 | 自动 time-zero、自动 first-break、自动速度估计、AGC 算法内部增益 | 部分相关 |
| L2 | 单阶段方法族或参数组合自动选择 | 在多种去噪/背景抑制候选间评分选择 | 接近当前 MyGPR 雏形 |
| L3 | 多步骤 pipeline 自动优化 | 自动决定运动补偿、零时、dewow、背景、增益、去噪、迁移的组合与参数 | MyGPR 中期目标 |
| L4 | 闭环 benchmark/ground-truth 驱动自动选参 | 用正演真值、人工专家 baseline、真实数据验收持续校准评分 | MyGPR 论文级目标 |

本报告使用这个分级区分“自动处理”“自动检测”“自动选参”。很多论文和软件都使用 automatic 这个词，但含义差异很大。

## 2. 检索方法

检索优先级：

1. UAV-GPR / drone GPR / airborne GPR 中是否已有自动选参或自动处理系统。
2. 普通 GPR 中是否已有可迁移的自动处理、自动参数估计或处理链自动化。
3. 商业软件、开源软件、专利与技术转移页面。
4. 与 MyGPR 当前方法直接相关的步骤：运动补偿、零时矫正、低频漂移抑制、背景抑制、增益、去噪、成像/迁移。

主要检索词包括：

- `UAV ground penetrating radar automatic parameter selection`
- `drone GPR automatic processing parameters`
- `UAV-GPR automatic processing pipeline`
- `airborne GPR automatic gain background removal parameter optimization`
- `ground penetrating radar automatic parameter selection processing`
- `ground penetrating radar automatic time zero correction first break picking`
- `ground penetrating radar automatic velocity estimation hyperbola fitting`
- `GPR automatic denoising parameter optimization PSO VMD`
- `GPR software automatic processing parameters Geolitix GPRPy RGPR EKKO_Project`
- `patent UAV ground penetrating radar automatic processing parameters`

检索结论中的“未发现”仅表示：在公开网页、论文摘要、软件文档和专利页面中未发现明确证据；它不是法律层面的自由实施意见，也不能排除未公开商业实现或未检索到的非英文/非索引资料。

## 3. UAV-GPR 方向调研

### 3.1 UAV-GPR 文献：重点在平台、定位和运动误差，不是完整自动调参

| 来源 | 主要内容 | 自动化等级 | 与 MyGPR 的关系 |
| --- | --- | --- | --- |
| [An Overview on Down-Looking UAV-Based GPR Systems, Remote Sensing 2022](https://www.mdpi.com/2072-4292/14/14/3245) | 综述 UAV-GPR 原型、应用和主要问题，强调飞行不稳定、速度/高度变化、定位精度、clutter 和电磁干扰；把信号处理分为标准 GPR 处理与高级成像算法 | L1-L2，主要是运动/成像相关处理 | 证明 UAV-GPR 自动化必须重视 RTK/IMU/高度和 motion compensation，但未给出全流程自动选参 |
| [Autonomous Airborne 3D SAR Imaging System for Subsurface Sensing, Remote Sensing 2019](https://www.mdpi.com/2072-4292/11/20/2357) | UAV 搭载 UWB-GPR，结合 RTK、激光测距和 SAR 处理生成高分辨率 3D 图像；强调精确定位和筛除会导致失焦的数据 | L2，自动飞行/成像链较强 | 对 MyGPR 运动补偿、轨迹质量控制和成像验证有参考价值；不是处理参数自动选优报告 |
| [A Lightweight and Low-Power UAV-Borne GPR Design for Landmine Detection, Sensors 2020](https://www.mdpi.com/1424-8220/20/8/2234/htm) | UAV 轻量化 SFCW GPR 硬件，指出 radargram 质量不仅受硬件影响，也受 GPR parameter settings 影响 | L0-L1 | 说明参数设置重要，但没有提出自动选参 |
| [A Real-Time Permittivity Estimation Method for SFGPR by FWI, Remote Sensing 2023](https://www.mdpi.com/2072-4292/15/21/5188) | 对 air-coupled / stepped-frequency GPR 进行介电常数实时估计，自动提取天线到反射面的距离，减少人工测量误差 | L1 | 可作为速度/介电常数估计方向的候选模块，不是全流程调参 |
| [Under the Sand: airborne GPSAR system, arXiv 2021](https://arxiv.org/abs/2106.10108) | 完整自动 airborne GPSAR 系统，集成导航、雷达成像和视觉，目标是地雷探测 | L2-L3，偏成像系统自动化 | 证明“自动 airborne GPR system”存在，但重点不是 MyGPR 式 preprocessing 参数搜索 |

UAV-GPR 的公开研究中，“自动”更常指自动飞行、自动定位、自动成像或自动目标检测。它们对 MyGPR 很重要，但不能替代“处理链参数自动选优”。MyGPR 后续应把 UAV 元数据作为自动选参的一等输入：RTK 轨迹质量、IMU 姿态、NRA15/高度计高度、速度变化、离地高度变化、时间同步质量，都应该参与候选空间约束和评分。

### 3.2 UAV-GPR 软件/产品：Geolitix 是最接近的公开产品线索

| 产品/软件 | 公开能力 | 自动化等级 | 判断 |
| --- | --- | --- | --- |
| [Geolitix GPR Processing](https://docs.geolitix.com/layers/gpr-processing.html) | 文档明确说 Geolitix 能分析输入 GPR 数据并生成处理方案；导入时可选择自动处理；多数处理步骤可通过数据分析自动化；用户仍需设置 radar velocity；还提供 drone GPR/air-launched antenna 的 flatten to horizon 和 automated horizon picking | L3-ish 产品黑盒 | 目前最接近 MyGPR 目标的公开软件证据，但不可复现、算法细节不公开、没有公开人工/自动参数对比和 ground truth 评估 |
| [SPH Engineering drone-mounted GPR](https://www.sphengineering.com/integrated-systems/technologies/gpr) | 商业 UAV-GPR 集成方案，包括 GPR、SkyHub、altimeter、UgCS 和 processing software；展示 Prism2/Geolitix 等处理结果 | L0-L2，取决于所用后处理软件 | 证明 UAV-GPR 产品生态成熟，但页面没有披露 MyGPR 式自动选参 |
| [GPR-SLICE drone-mounted Radarteam data tutorial](https://www.huntergeophysics.com/2019/03/14/gpr-slice-tutorial-video-processing-drone-mounted-radarteam-gpr-data-in-gpr-slice/) | 针对 drone/UAV/aircraft 采集数据的处理教程；低频深部数据需要用户调整 slices 和 gridding 参数 | L0-L1 | 更像人工流程和经验参数，不是数据驱动自动选参 |

结论：如果组会需要回答“有没有别人已经做了类似功能”，最严谨表述是：

> 公开资料中，Geolitix 已经有商业级 GPR 自动处理能力，且覆盖 drone/air-launched 场景的一部分处理问题；但公开资料未显示其具有可复现的、逐算法参数候选搜索、人工 baseline 对比、ground-truth 指标验证和论文级证据导出。UAV-GPR 学术文献则更多解决运动/定位/成像/检测问题，未发现完整 preprocessing pipeline 自动选参框架。

### 3.3 UAV-GPR 专利/技术转移

| 来源 | 主要权利要求/内容 | 与自动选参关系 |
| --- | --- | --- |
| [LLNL US11614534B2 UAV ground penetrating radar array](https://patents.google.com/patent/US11614534B2/en) 与 [WO2021155343A2](https://patents.google.com/patent/WO2021155343A2/en) | UAV GPR array、multistatic acquisition、calibration/pre-compensated transmit signal、local/remote object detection | 偏硬件阵列、波形/校准和目标检测；不是 MyGPR preprocessing 参数自动选优 |
| [LLNL Drone-based GPR Array technology page](https://ipo.llnl.gov/technologies/national-security-and-defense/drone-based-ground-penetrating-radar-array) | 声称降低阵列 SWaP、自动/半自动运行、自动 waveform generation、软件定义雷达和 swarm 架构 | 与自动系统有关，但不是公开处理链参数搜索 |
| [CN113075738A UAV-based GPR measurement system](https://patents.google.com/patent/CN113075738A/en) | 双无人机发射/接收、GPS、altimeter、速度传感器、水平仪、磁补偿器、带阻滤波器 | 偏系统结构与抗干扰，不是自动选参 |

这些专利说明 UAV-GPR 系统、阵列、校准、抗干扰和目标检测已有专利布局；未看到直接覆盖“基于 B-scan 质量指标自动选择 preprocessing pipeline 参数”的公开权利要求。后续若 MyGPR 进入论文/专利阶段，应请学校或专利代理做正式 freedom-to-operate 检索，本报告不能替代法律检索。

## 4. 普通 GPR 方向调研

### 4.1 开源/商业软件

| 软件/来源 | 公开能力 | 自动化等级 | 对 MyGPR 的启发 |
| --- | --- | --- | --- |
| [ERDC Automated GPR post-processing software in R, 2022](https://erdc-library.erdc.dren.mil/items/c95bac4f-900c-4218-a7b7-3e5b7b25c38b) | 用 R 脚本自动化 GSSI 数据后处理，包括 static data removal、time-zero correction、distance normalization、filtering、stacking；目标是减少 SME 后处理负担 | L2，固定标准流程自动执行 | 与 MyGPR 很接近，但公开摘要显示是脚本化标准流程，不是多候选参数评分搜索 |
| [GPRPy paper / GitHub](https://github.com/NSGeophysics/GPRPy) | 开源 Python GPR 处理与可视化，GUI 可自动生成脚本，支持 profile processing、velocity analysis、3D interpolation | L0-L1 | 强在可复现脚本和开源生态；没有证据显示自动调参 |
| [RGPR basic processing tutorial](https://emanuelhuber.github.io/RGPR/02_RGPR_tutorial_basic-GPR-data-processing/) | R 包含 firstBreak/time0、dewow、frequency filter、gain、median filter、f-k filter、background subtraction 等处理；教程中的窗口、频率和 gain 参数由用户给定 | L0-L1 | 可作为算法与脚本化处理参考；不是自动选参 |
| [EKKO_Project product page](https://www.sensoft.ca/products/ekko-project/) 与 [Processing Module User Guide](https://www.sensoft.ca/wp-content/uploads/2015/11/Processing-Module-Users-Guide.pdf) | 商业 GPR 处理/报告软件；Processing Module 支持 recipe、processing stream、AGC、SEC2、background subtraction 等；AGC 文档说明 window width 和 maximum gain 等参数 | L0-L1 | 说明商业软件重视 recipe、可比较处理前后、参数记录；没有公开数据驱动自动选参 |
| [GPR-SLICE](https://gpr-slice.es/index_en.html) | 商业 2D/3D GPR 处理、slice/volume 成像，兼容多厂家；另有 GPRSIM 正演模拟 | L0-L1 | 对 MyGPR 的 benchmark 和报告可视化有参考；公开页面未显示自动参数搜索 |
| [Geolitix](https://docs.geolitix.com/layers/gpr-processing.html) | 自动处理、自动 slicing、目标导向处理选项、horizon picking | L3-ish 产品黑盒 | 最值得重点对标的商业产品 |

普通 GPR 软件里，最接近 MyGPR “科研闭环”的不是某个具体算法，而是三个方向的组合：

1. Geolitix 的自动处理产品化体验。
2. ERDC R 脚本的标准流程自动化和减少 SME 后处理负担。
3. GPRPy/RGPR 的开源、可复现、脚本化处理历史。

MyGPR 如果要做出论文/专栏价值，需要把这三者合并并进一步超过：自动候选搜索、参数可解释、人工 baseline 对比、gprMax ground truth 指标、UAV 元数据约束。

### 4.2 单算法/单任务自动化文献

| 方向 | 来源 | 自动化等级 | 可迁移点 |
| --- | --- | --- | --- |
| 去噪参数自动搜索 | [Particle Swarm Optimization-Based VMD for GPR Denoising, Remote Sensing 2022](https://www.mdpi.com/2072-4292/14/13/2973) | L1-L2 | 明确指出 VMD 的 K 和 penalty parameter 依赖人工经验，并用 PSO 自动搜索；这是“GPR 处理参数自动优化”的强相关证据 |
| 自适应滤波 | [Depth-Adaptive Filtering Method for GPR Tree Roots Detection, arXiv/IEEE TIM 2023](https://arxiv.org/abs/2305.18775) | L1 | 通过 STFT/WLR 生成深度自适应滤波窗口，减少对土壤信息的依赖；可启发 MyGPR 频域/深度自适应候选 |
| 地面杂波自适应去除 | [Adaptive Ground Clutter Removal Algorithm, Sensing and Imaging 2006](https://research.jku.at/de/publications/adaptive-ground-clutter-removal-algorithm-for-ground-penetrating-/) | L1 | 估计地面杂波位置和变化形态，强调保留 buried object impulse response；可作为背景抑制评分中“目标保留”的依据 |
| 自动速度/迁移参数 | [Finetuning GPR velocity analysis from hyperbola fitting using migration, Near Surface Geophysics 2023](https://www.earthdoc.org/content/journals/10.1002/nsg.12250?TRACK=RSS) | L1 | 速度是深度转换和 migration 的关键参数，hyperbola fitting + migration collapse 可用于自动/半自动速度估计 |
| 自动 hyperbola 检测和速度确定 | [Hyperbola Detection with RetinaNet and Comparison of Hyperbola Fitting Methods, Remote Sensing 2022](https://www.mdpi.com/2072-4292/14/15/3665) | L1-L2 | 文摘显示在 bounding boxes 内比较 10 种自动速度确定方法，可为 MyGPR 的目标型 ROI 和 migration velocity 评分提供思路 |
| 介电常数/高度自动估计 | [Real-Time Permittivity Estimation by FWI, Remote Sensing 2023](https://www.mdpi.com/2072-4292/15/21/5188) | L1 | 自动提取天线到反射面的距离并估计介电常数，可用于 zero-time/velocity/migration 的物理约束 |
| 自动目标检测/识别 | [USF Systems and methods for detecting buried objects patent](https://digitalcommons.usf.edu/usf_patents/970/) 及大量深度学习检测论文 | L1-L2 | 可以作为评价指标或 ROI 生成器，但“自动检测目标”不等于“自动处理参数调优” |

关键判断：普通 GPR 里已经有很多可迁移的自动参数估计思想，尤其是 PSO/VMD、自动 hyperbola/velocity、adaptive clutter/filtering。但它们基本都是“单方法局部最优”。MyGPR 的论文价值应落在“跨步骤、跨目标、多指标、UAV 元数据约束”的系统化框架上。

### 4.3 普通 GPR 专利

| 来源 | 内容 | 与 MyGPR 的关系 |
| --- | --- | --- |
| [US7034740B2 Method and apparatus for identifying buried objects using GPR](https://www.patentbuddy.com/Patent/7034740) | 检测空间相关性并构建 buried object 图像结构 | 目标识别/图像构建，不是处理链调参 |
| [USF US10175350 Systems and methods for detecting buried objects](https://digitalcommons.usf.edu/usf_patents/970/) | 基于 GPR 信号计算参数、绘制空间函数、确定 hyperbola apex 估计 buried object 位置 | 可作为自动目标检测/ROI 参考 |
| [US11029402B2 Wideband GPR system and method](https://pubchem.ncbi.nlm.nih.gov/patent/US-11029402-B2) | 宽带 GPR 系统和 full waveform digitization | 偏硬件/采集，不是自动选参 |
| [US11169256B2 Precise infrastructure mapping using FWI of GPR signals](https://patents.google.com/patent/US11169256B2/en) | 用 GPR full-waveform inversion 做基础设施映射 | 更接近反演/成像，不是 preprocessing 参数搜索 |

专利侧结论：公开可见专利更集中于硬件系统、目标检测、成像/反演、阵列和采集控制。MyGPR 的“可解释自动调参 + ground-truth benchmark + 人工 baseline 证据导出”在公开资料中仍有差异化空间。

## 5. 对 MyGPR 当前自动选参的诊断

当前 `core/auto_tune.py` 是“单方法 rule-based candidate search + scoring”架构，不是全流程 optimizer。它的优点是已经有候选生成、粗筛/细化、ROI、评分函数、Pareto/profile、失败 trial 和选择置信度；但它还不足以支撑论文级自动调参。

### 5.1 关键缺陷

1. **候选空间仍然不够数据尺度敏感。**
   `core/methods_registry.py` 中存在大量固定候选：`subtracting_average_2D.ntraces` 到 501，`median_background_2D.ntraces` 到 301，`svd_subspace.rank_end` 到 40，`agcGain.window` 到 121。`core/auto_tune.py` 里已经有 `_sanitize_int_candidates` 和部分 adaptive window builder，但“候选配置层、展示解释层、运行层”没有统一的强约束契约。因此在只有 36 条 A-scan 时，出现数百 trace 的参数是可信度硬伤，即使底层运行时做了 clamp 或算法内部退化处理，也会让报告显得不科学。

2. **缺少参数物理含义和数据维度之间的显式约束。**
   窗口类参数应绑定 sample/trace 维度，rank 类参数应绑定 `min(n_samples, n_traces)`，zero-time 裁剪应绑定 time window 和 first-break 可信区间，motion compensation 应绑定 RTK/IMU/高度计的质量。现在这些约束没有被统一建模。

3. **单步骤局部最优，容易破坏后续步骤。**
   当前每个方法主要用 before/after 指标评分。零时、背景、增益、去噪之间存在强耦合：零时裁掉有效信号会让后续所有图像“看起来干净”；背景窗口过大可能保留水平条纹或误删宽目标；AGC 可提升可见性但破坏幅值关系。论文级系统必须做 pipeline-level objective，而不是每一步独立最优。

4. **评分函数还不够 ground-truth aware。**
   现有评分重视 horizontal coherence、saliency、edge、deep contrast、hot pixel、band energy 等通用指标，但没有稳定地对“已知目标必须保留”负责。组会已经决定用 gprMax 正演，这一步应该成为自动选参升级的核心。

5. **UAV 元数据尚未成为自动选参核心输入。**
   UAV-GPR 的独特性来自 RTK、IMU、NRA15/高度计、速度和轨迹。若自动选参只看 B-scan 像素，它和普通 GPR 软件差异不大。MyGPR 必须让 motion compensation、height flatten、trace resampling、bad-trace rejection 与元数据质量模型联动。

6. **缺少专家 baseline 协议。**
   “人工参数”不能随便设，也不能只用默认值。需要一个固定协议：专家只看 B-scan 和常规元数据，不知道 gprMax 真值；记录其选择理由；自动选参与其在同 ROI、同显示尺度、同 pipeline 下比较。

7. **缺少失败/低置信度输出。**
   自动选参不应总是给出“最好参数”。在小数据量、低 SNR、无目标、元数据缺失、候选分数差距很小的时候，应输出 low confidence 或 recommend manual review。

### 5.2 36 条 A-scan 场景的立即结论

对于 `n_traces = 36` 的数据：

- 背景抑制窗口 `ntraces` 不应出现 101/201/501 这种推荐结果。
- 滑动窗口应优先使用比例型候选，例如 `max(3, round(n_traces * ratio))`，并限制为 odd integer。
- 若算法允许窗口超过数据长度并内部等价为全局平均，也必须在报告里显示“effective_ntraces=36”，不能显示原始 501。
- 如果候选被 clamp，trial 记录中应同时保存 `requested_params`、`effective_params` 和 `constraint_warnings`。
- 对小数据集应增加独立惩罚：窗口过大导致参数不可解释时，即使图像评分好，也不能作为高置信度推荐。

这应成为下一轮工程重构的第一个测试切入点。

## 6. MyGPR 论文级目标路线

### M1：数据感知参数域（必须先做）

目标：把所有候选参数变成“可解释、可验证、不可越界”的参数域。

具体任务：

- 新增 `ParameterDomain` / `CandidateConstraint` 层。
- 每个方法声明参数语义：`trace_window`、`sample_window`、`rank`、`time_ns`、`velocity`、`height_m`、`ratio`、`boolean`。
- 候选生成前先根据数据形状和元数据生成合法域。
- 运行后记录 `requested_params`、`effective_params`、`constraint_warnings`。
- 加测试：36 traces 时 background `ntraces` 不得超过 36；SVD rank 不得超过 `min(shape)`；zero-time 不得裁掉超过有效窗的安全阈值。

完成标准：任何自动选参报告中都不再出现“参数明显大于数据量但没有解释”的情况。

### M2：gprMax ground-truth benchmark

目标：让自动选参不只追求图像观感，而是对真实结构负责。

当前已有 `docs/gprmax_auto_tune_validation_plan.md`，应继续扩展：

- `cylinder_single_v1`：单双曲线，验证 apex/arms 保留。
- `cylinder_double_depth_v1`：浅强 + 深弱目标，验证深部弱目标不被误删。
- `layered_soil_interface_v1`：层状界面，验证背景抑制不破坏有效层位。
- `crack_air_filled_v1`：裂缝/倾斜弱结构，验证边缘和弱反射。
- `no_target_noise_v1`：无目标，验证不制造假异常。
- UAV 扰动扩展：高度变化、速度不均、姿态扰动、bad trace、GPS/IMU 时间偏移。

每个 scenario 输出 `scenario.json`、`model.in`、`mygpr_bscan.csv`、`ground_truth.json`、`preview.png` 和 HTML/Markdown 对比报告。

核心指标：

- `target_roi_energy_preservation`
- `apex_saliency_preservation`
- `hyperbola_arm_continuity`
- `layer_continuity_preservation`
- `background_suppression_outside_roi`
- `false_positive_penalty`
- `over_smoothing_penalty`
- `display_independent_score`

### M3：pipeline-level 多目标优化

目标：从单步骤局部评分升级到 pipeline-level 自动选参。

建议架构：

- 阶段顺序仍为：运动补偿、零时矫正、低频漂移、背景抑制、增益、去噪、成像/迁移。
- 每个阶段有合法候选域和先验约束。
- 采用分层搜索：
  - 快速模式：stage-wise constrained search。
  - 标准模式：保留每阶段 top-k 进入 beam search。
  - 研究模式：multi-objective Pareto + Bayesian/SMBO 或 evolutionary search。
- 评分分为：
  - 图像质量指标。
  - ground-truth 指标。
  - 物理一致性指标。
  - 元数据质量指标。
  - 过处理惩罚。
  - 参数可解释性惩罚。

完成标准：自动选参输出的不是一个孤立方法参数，而是一套 pipeline 参数、每步理由、每步风险、整体置信度。

### M4：UAV metadata-aware motion compensation

目标：把 MyGPR 与普通 GPR 自动调参区分开。

应把 `D:\CDUT-UavGPR-Controller` 主控项目中的 RTK、九轴姿态、NRA15/高度计接入作为核心输入：

- RTK：轨迹重采样、速度估计、横向间距、bad segment。
- IMU：pitch/roll/yaw、天线相位中心偏移、姿态导致的足迹变化。
- NRA15/高度计：离地高度、air-ground two-way delay、amplitude spreading compensation。
- 时间同步：每条 trace 的传感器插值质量和时间偏移。

motion compensation 自动选参不应只看图像，而应先判断元数据质量：

- 元数据可信：优先物理补偿。
- 元数据缺失/漂移：退化为 B-scan 结构估计，输出低置信度。
- 高度变化超过阈值：强制启用 height flatten 或提示不可比较。
- 姿态超限：标记该段数据不适合自动处理或降低权重。

### M5：人工 baseline 与科研证据导出

目标：让“自动优于人工”可被组会、论文或专栏审查。

协议：

- 人工 baseline 来源：
  - 日常处理页当前参数。
  - 若无用户参数，则使用固定经验参数 profile。
- 人工专家模拟要求：
  - 不知道 gprMax 真值。
  - 只看原始 B-scan、常规元数据和可见结构。
  - 记录选择理由。
- 对比要求：
  - 同一输入。
  - 同一 ROI。
  - 同一显示尺度。
  - 同一 pipeline。
  - 同一导出格式。

报告必须包含：

- 原始 B-scan。
- 每一步处理前 B-scan。
- 人工参数处理后 B-scan。
- 自动参数处理后 B-scan。
- 人工参数表、自动参数表、effective 参数表。
- 指标表和 delta。
- ground-truth 结构保留评价。
- 自动选参失败或低置信度案例。

### M6：论文/专栏贡献框架

可形成的论文/专栏主题：

> A Ground-Truth-Aware Auto-Tuning Framework for UAV-Borne Ground Penetrating Radar Processing

核心创新点建议：

1. UAV-GPR 标准处理链自动选参，而不是单算法去噪。
2. 数据尺度约束参数域，解决小样本 B-scan 下参数不可解释问题。
3. gprMax ground-truth 驱动指标，避免只优化视觉观感。
4. 人工专家 baseline 对比，量化减少主观调参。
5. RTK/IMU/高度计元数据进入 motion compensation 和评分。
6. 完整可复现证据包：输入 hash、参数、图像、指标、warnings、commit。

建议实验设计：

- Synthetic benchmark：gprMax 多场景、多噪声、多高度扰动。
- Ablation：
  - 无参数域约束 vs 有参数域约束。
  - 单步局部评分 vs pipeline-level scoring。
  - 无 ground truth 指标 vs 有 ground truth 指标。
  - 无 UAV metadata vs 有 UAV metadata。
- Baseline：
  - 经验参数 profile。
  - 人工专家参数。
  - 固定商业/常规 recipe。
  - 当前 MyGPR v1 auto-tune。
- Metrics：
  - 目标保留。
  - 背景抑制。
  - 假异常。
  - 深部可见性。
  - 过处理。
  - 参数合理性。
  - 运行时间。

## 7. 与已有外部工作的差异化定位

| 外部工作类型 | 已有能力 | MyGPR 应避免重复 | MyGPR 可突出 |
| --- | --- | --- | --- |
| Geolitix 类产品 | 自动处理体验强 | 不应只做黑盒按钮 | 开源/可解释/可导出证据/ground truth |
| ERDC R 自动后处理 | 标准流程脚本化 | 不应只固定 recipe | 多候选评分和专家 baseline 对比 |
| GPRPy/RGPR | 开源处理和脚本复现 | 不应只堆算法按钮 | 自动参数域、pipeline 优化、GUI/CLI 统一证据 |
| PSO-VMD 等论文 | 单算法参数优化 | 不应只写一个 denoise auto search | 跨处理链、跨结构、UAV metadata |
| UAV-GPR SAR/平台论文 | 定位/成像/运动补偿强 | 不应忽略物理元数据 | 把 RTK/IMU/高度质量纳入自动选参 |
| 目标检测/深度学习 | 自动找目标 | 不应把 detection 当 tuning | 用 detection/ROI 反哺处理参数选择 |

## 8. 推荐下一步工程任务

### 第一优先级：修正自动选参参数域

立即新增约束层和测试：

- `tests/test_auto_tune_candidate_constraints.py`
- 36 traces 的 background 窗口不得超过 trace count。
- rank 参数不得超过矩阵秩上限。
- zero-time 自动候选不得产生空数据或过度裁剪。
- 所有 trial 都能导出 requested/effective params。

这一步是可信度前提，优先级高于继续调权重。

### 第二优先级：扩展 gprMax benchmark

在当前 `cylinder_single_v1` 基础上新增至少四个场景，并把自动/人工对比报告纳入同一 schema。不要先追求复杂地质模型，先覆盖“自动选参最容易犯错”的小而明确场景。

### 第三优先级：升级评分为 ground-truth aware

把目标 ROI、apex、hyperbola arms、layer interface、crack edge 纳入指标，先让报告能解释“为什么自动更好”，再考虑更复杂的优化器。

### 第四优先级：pipeline-level optimizer

先做 beam search 或 stage-wise top-k，不急着上复杂机器学习。等 gprMax 场景、指标和 baseline 稳定后，再尝试 Bayesian optimization、NSGA-II 或强化学习式策略。

### 第五优先级：UAV metadata-aware motion compensation

真实 CSV + RTK + IMU + NRA15 数据到位后，优先验证时间同步、轨迹重采样、高度补偿和姿态补偿。没有真实数据前，用 gprMax + synthetic trajectory perturbation 做先验测试。

## 9. 可直接向组会汇报的话术

1. 公开资料中，UAV-GPR 已经有很多自动飞行、自动成像、运动误差补偿和目标检测研究，但未发现公开可复现的“UAV-GPR preprocessing 全流程自动选参 + 人工 baseline 对比 + ground truth 验证”系统。
2. 商业软件 Geolitix 已经具有 GPR 自动处理能力，是我们需要重点对标的产品；但其算法细节不公开，且公开文档仍要求用户设置关键速度参数。
3. 普通 GPR 里已有 PSO-VMD、adaptive clutter removal、自动 velocity/hyperbola 等局部自动化方法，说明自动调参思想是合理且有文献基础的。
4. MyGPR 当前自动选参还是雏形，最大问题不是某个权重，而是参数域、ground-truth 指标、pipeline 耦合和 UAV 元数据没有形成统一框架。
5. 下一阶段先修参数域越界问题，再用 gprMax 建 benchmark，最后扩展到 pipeline-level 和 UAV metadata-aware 自动选参。

## 10. 参考来源

- Geolitix, [GPR Processing](https://docs.geolitix.com/layers/gpr-processing.html)
- Catapano et al., [An Overview on Down-Looking UAV-Based GPR Systems](https://www.mdpi.com/2072-4292/14/14/3245)
- Garcia-Fernandez et al., [Autonomous Airborne 3D SAR Imaging System for Subsurface Sensing](https://www.mdpi.com/2072-4292/11/20/2357)
- Sipos and Gleich, [A Lightweight and Low-Power UAV-Borne Ground Penetrating Radar Design for Landmine Detection](https://www.mdpi.com/1424-8220/20/8/2234/htm)
- SPH Engineering, [Drone-Mounted GPR](https://www.sphengineering.com/integrated-systems/technologies/gpr)
- Hunter Geophysics, [Processing drone-mounted Radarteam GPR data in GPR-SLICE](https://www.huntergeophysics.com/2019/03/14/gpr-slice-tutorial-video-processing-drone-mounted-radarteam-gpr-data-in-gpr-slice/)
- ERDC/CRREL, [Automated ground-penetrating-radar post-processing software in R programming](https://erdc-library.erdc.dren.mil/items/c95bac4f-900c-4218-a7b7-3e5b7b25c38b)
- NSGeophysics, [GPRPy GitHub](https://github.com/NSGeophysics/GPRPy)
- Plattner, [GPRPy: Open-source ground-penetrating radar processing and visualization software](https://www.researchgate.net/publication/341092502_GPRPy_Open-source_ground-penetrating_radar_processing_and_visualization_software)
- Huber, [RGPR Basic GPR data processing](https://emanuelhuber.github.io/RGPR/02_RGPR_tutorial_basic-GPR-data-processing/)
- Sensors & Software, [EKKO_Project](https://www.sensoft.ca/products/ekko-project/) and [Processing Module User Guide](https://www.sensoft.ca/wp-content/uploads/2015/11/Processing-Module-Users-Guide.pdf)
- Liu et al., [Particle Swarm Optimization-Based Variational Mode Decomposition for GPR Data Denoising](https://www.mdpi.com/2072-4292/14/13/2973)
- Luo et al., [A Depth-Adaptive Filtering Method for Effective GPR Tree Roots Detection](https://arxiv.org/abs/2305.18775)
- Ossberger et al., [Adaptive Ground Clutter Removal Algorithm for GPR Applications](https://research.jku.at/de/publications/adaptive-ground-clutter-removal-algorithm-for-ground-penetrating-/)
- Ronning, [Finetuning GPR velocity analysis from hyperbola fitting using migration](https://www.earthdoc.org/content/journals/10.1002/nsg.12250?TRACK=RSS)
- Remote Sensing, [Hyperbola Detection with RetinaNet and Comparison of Hyperbola Fitting Methods in GPR Data](https://www.mdpi.com/2072-4292/14/15/3665)
- Li et al., [A Real-Time Permittivity Estimation Method for SFGPR by FWI](https://www.mdpi.com/2072-4292/15/21/5188)
- LLNL, [Drone-based Ground Penetrating Radar Array](https://ipo.llnl.gov/technologies/national-security-and-defense/drone-based-ground-penetrating-radar-array)
- Google Patents, [US11614534B2 UAV ground penetrating radar array](https://patents.google.com/patent/US11614534B2/en)
- Google Patents, [WO2021155343A2 UAV ground penetrating radar array](https://patents.google.com/patent/WO2021155343A2/en)
- Google Patents, [CN113075738A Ground penetrating radar measurement system based on UAV](https://patents.google.com/patent/CN113075738A/en)
- USF Patents, [Systems and methods for detecting buried objects](https://digitalcommons.usf.edu/usf_patents/970/)
- PatentBuddy, [US7034740B2 Method and apparatus for identifying buried objects using GPR](https://www.patentbuddy.com/Patent/7034740)
