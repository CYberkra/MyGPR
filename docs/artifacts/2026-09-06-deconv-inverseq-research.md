# 反褶积与 inverse-Q 补偿调研笔记（2026-09-06）

> 状态：**仅调研完成，未开工**。三路调研：ChainSurvey（商业软件对标，28m56s）、QCompSurvey（inverse-Q，34m40s）、反褶积方法细节由 ChainSurvey + QCompSurvey 交叉覆盖 + 主会话直读 Schmelzbach & Huber 2015 全文补齐（DeconvSurvey 卡死已取消）。
> 触发：工业级缺口清单（2026-09-06 会话）——反褶积是"没有它就不像商业 GPR 软件"的缺口。

## 1. MyGPR 数据与架构约束（已验证，仓库实证）

### 1.1 数据形态（关键前提）

| 事实 | 证据 |
|---|---|
| 真实数据是**仪器端已合成时域的等效 B-scan**，不是原始 SFCW 频谱 | 英山测线 CSV 头：`Number of Samples = 501`、`Time windows (ns) = 700`（docs/artifacts/yingshan-full-validation.json） |
| 20–170 MHz / 501 频点是**采集参数元数据**，导入后全链按脉冲 GPR 时域处理 | `mygpr/domain/autotune/data_context.py:18 FIELD_SFCW_BAND_MHZ=(20,170)`、`:87 frequency_points=501` |
| 采样间隔 dt ≈ 700ns/501 ≈ **1.397 ns**（采样率 ≈ 715.7 MHz） | 同上推导 |
| 全库无 SFCW→时域 IFFT 合成步骤（ifft 仅存在于 fk_filter/stolt/global_spectral 的二维谱运算） | grep 实证 |
| 道距 0.08537 m，共偏移距单覆盖，无 CMP | 同一 CSV 头 + 审计报告 |

**结论**：反褶积/inverse-Q 将作用于等效时域数据。仪器的频域合成（加窗、频带裁剪）已不可逆地塑造了有效子波——反褶积反的是"仪器合成子波+地层响应"的复合；inverse-Q 反的是地层衰减（这部分与仪器合成正交，物理上更干净）。

### 1.2 实现契约（未来落地时的验收口径）

- 方法契约：`(ny, nx) float32` → `(data, metadata)`；参数经 registry schema 注入（`mygpr/infrastructure/processing/algorithms/methods.py` NATIVE_ALGORITHMS 单一事实源）。
- 验收：`scripts/algorithm_fitness_benchmark.py` harness——退化输出（全零）0 分、data-noop 不计分；4 个确定性 fixture 可扩展。
- 处理链现状：dewow → set_zero_time → 背景抑制（6 法）→ 去噪（7 法）→ 增益（5 族）→ stolt/kirchhoff 偏移 → time_to_depth。
- **MyGPR 已有带通**：`frequency_filter_1d`（frequency.py:19 支持 bandpass/lowpass/highpass/notch，低/高频边界 + taper）——ChainSurvey 报告的"deconv 后无带通可跟"是错的，前置缺口不存在。
- 契约教训（审计报告 §3.3）：时间基准缺失抛 ValueError 而非静默猜测；新方法必需参数缺失同样应显式报错，不得静默直通。

## 2. 反褶积（Deconvolution）

### 2.1 商业软件配置（confirmed）

| 软件 | 模式 | 默认参数 | 来源 |
|---|---|---|---|
| REFLEXW | predictive / deterministic deconv | autocorr 窗 0–60ms、filterlength=60ms、lag=12ms(multiple)/20ms(ghost)；手册明言反褶积前提在 GPR 中**常不满足**、"not well suited for GPR"、需人工检查伪影 | sandmeier-geo.de gpr_2d_import_processing.pdf + deconvolution.pdf |
| EKKO_Project | predictive（Wiener+Toeplitz） | Filter Width=3×pulse width、Delay=1.5×pw、Spike Width=0.3×pw、**Whiten 默认 0.1**；约束 Filter Width > Delay + Spike Width | sensoft.ca 手册 §7.4.3 |
| RadExplorer | Predictive Deconvolution | 位于 Amplitude Correction → Deconv → Bandpass → Stolt F-K 序列中；默认值按数据属性自动 | MALÅ Radexplorer.pdf |
| Geolitix | predictive (Wiener)、逐道时域 | Prediction lag 1–80 samples、Operator length 2–120 samples、Prewhitening 0–0.1（步进 0.001） | docs.geolitix.com |
| GPRPy（开源） | **无** | 逐函数核实 gprpyTools.py 640 行 + gprpy.py 1392 行，无 deconv/inverse-Q/bandpass | GitHub NSGeophysics/GPRPy |
| RGPR（开源） | mixed-phase（Schmelzbach & Huber 2015 方法） | W=(32,112)ns、wtr=5、nf=35、mu=1e-5；deconv 后**必须再 bandpass c(5,20,160,210)** | emanuelhuber.github.io RGPR 教程 |

### 2.2 核心方法学（Schmelzbach & Huber 2015 IEEE TGRS，全文直读，confirmed）

GPR 天线通常辐射**混合相位（mixed-phase）子波**（能量居中，非最小相位）——这是标准脉冲反褶积在 GPR 上常失败的根因。该文方案（MyGPR 最合适的 v1 蓝本）：

1. **分解**：混合相位子波 w(t) = 最小相位等价 m(t) ∗ 全通滤波器 p(t)；全通相位谱可用**线性函数**近似（φ_p(f) = φ_rot + b·f）——短脉冲 GPR 子波谱平滑，近似成立。
2. **步骤一（逐道最小相位 spike 反褶积）**：解两次正态方程（先估 m⁻¹，再由其 ACF 估 m），**用相邻道合成 supertrace 稳定算子估计**（文中 11 道，35 样本算子）。
3. **步骤二（全局相位旋转）**：扫描 0°–180° 旋转角，**kurtosis 最大 = 最稀疏输出**（合成例 122°，实测例 134°/92°）；kurtosis 估计需要**大样本量**（实测数据 ~30000 样本才到 90% 收敛）——必须全测线全局估一个相位角，不能短窗/短道。
4. **验收效果**：与带限反射率的相关系数 输入 −0.05 → 仅最小相位反褶积 0.61 → 完整方案 0.76。
5. **限制（原文明言）**：需要估计窗内信号平稳；显著频变吸收/色散（衰减）会让平稳假设失效 → 建议后续引入 Gabor（时变）反褶积；kurtosis 法对带宽窄、空间相关强的数据需更多样本。
6. **字段数据参数**：100 MHz、dt=0.8ns、算子 35 样本（27.2ns）、估计窗 32–112ns、11 道合并、全局相位旋转 134°、deconv 后 15–175 MHz 带通。

**与 MyGPR 数据的对齐**：英山数据 dt=1.397ns、501×1813 道规模与该文 100MHz 实测（0.8ns、391 道）同量级偏大；20–170 MHz 带宽窄于该文 20–400 MHz，kurtosis 收敛所需样本更多、分辨率提升幅度预期更保守（inferred）。仪器合成等效子波非零相位假设需在 fixture 上验证（pending validation）。

### 2.3 UAV-GPR 文献定位（confirmed）

MDPI 2022 UAV-GPR 综述 14 个系统**无一采用**反褶积/inverse-Q（全部 time gating/background removal/gain + backprojection/f-k/MWT 成像）；Dupuy ISSW 2024 雪崩 UAV-GPR 同样无；2025 SBD-SOOT 论文证明 deconv 在 GPR 领域仍是研究热点而非常规实践。**定位结论：反褶积是对标商业软件的"高级可选项"，不是 UAV 默认链成员**。

### 2.4 处理链位置（多源一致，confirmed）

dewow/DC 最先（REFLEXW+EKKO 明文）→ 时间零位置顶（Geolitix 强制）→ 背景抑制/去噪 → **【deconvolution】** → 增益（EKKO 章节序=增益前；RGPR 实操=增益后，两派并存，EKKO 路线更稳）→ **deconv 后强制带通**（RGPR 显式示范；压制被放大的带外噪声，REFLEXW 亦警告需人工检查伪影）→ 偏移最后。

## 3. inverse-Q 补偿

### 3.1 商业软件零命中（confirmed）→ 差异化机会

REFLEXW、EKKO_Project、RadExplorer、Geolitix、GPRPy 均**无 inverse-Q**（GPR-SLICE/Leopard 不可考：Leopard 官网证书错误+Wayback 空，疑似消亡）。商业界对 GPR 衰减的答案是"温和增益 + 反褶积"，而非 inverse-Q。UAV-GPR 文献中 inverse-Q 也**零命中**（Crossref 多组合扫描）——学术上 Bano 1996、Sensors 2018 有 GPR/Q 补偿专文，但从未进入商业默认链。**inverse-Q 对 MyGPR 是差异化卖点，不是对标缺口**。

### 3.2 方法学（confirmed，多源一致）

- **正演模型**：constant-Q 衰减算子 exp(π·t·f/Q)，(t,f) 域逐道施加。Turner 1994 证明 GPR 带宽内衰减近似线性于频率、单参数 Q* 足够；Bano 1996 GJI 给出低损耗介质 tanδ ≈ 1/Q——constant-Q 模型在 20–170 MHz 带宽内**物理上站得住**。
- **标准算法**：Wang 2002（Geophysics，347 引用）稳定逆 Q 滤波——下行延拓 + 稳定化算子，振幅算子分解为 1D 时间函数 × 1D 频率函数。这是任何实现的数学骨架。
- **现代变体**：正则化反演+数据驱动谱整形（Du 2022：抑制高频过补偿与低频泄漏）、迁移型 IQF（arXiv 2023）、Q 体引导（Yang 2025）、深度学习衰减补偿（LGRS 2023/2025，非工业默认）。
- **Q 估计**：谱比法（主力，需源谱校正）、质心频移法 CFS（GPR 损耗正切活跃应用，Geophysics 2022）、上升时间法（被批评过度简化）。
- **GPR 材料 Q 先验（weak，必须现场标定）**：干粗粒土 Q≈20–100、湿/黏土 Q≈5–20、岩石 Q≈20–100、混凝土 Q≈10–40——文献可及全文查不到逐材料可靠数值表，此为从 Bano/Turner 模型上下文推断的起始先验。

### 3.3 稳定化——设计的承重墙（confirmed 失败模式 + [INFERERENCE] 数值推导）

- 失败模式（文献 confirmed）：晚时/高频分量指数放大 → 噪声底炸成振铃伪影与人工子波（REFLEXW 增益警告）；高频过补偿+低频泄漏破坏频谱（Du 2022）；无稳定化的算子数值不可用（Wang 2002 存在的意义）。
- **增益上限惯例：无文献定值**。地震界口口相传 ~40–60 dB，但 QCompSurvey 穷尽 arXiv/DOAJ/Crossref/厂商手册均查不到可引用出处（[INFERRED] folklore）。
- **本会话数值推导（[INFERENCE]，待实现时验证）**：按 exp(π·t·f/Q)，MyGPR 700ns 记录 @170 MHz 末端的无界增益：Q=100 → ~38 dB；Q=50 → ~65 dB；Q=20 → ~160 dB。**低 Q 时需求增益远超任何合理上限** → 增益上限 + 噪声底以上频带裁剪是承重设计，不是附加项。
- QCompSurvey 推荐实现管线（v1，未开工）：metadata Q 或 谱比/CFS 估 Q → (t,f) 域 constant-Q 算子（复用 global_spectral.py 的 fft 基础设施）→ Wang 式稳定化 + 显式 gain_limit 参数（超限 taper/阻尼并发告警）→ 噪声底以上才施增益 → 契约不变，metadata 记录 Q/cap/max-applied-gain → fitness benchmark 验收。**SFCW 逆衰减函数变体（Sensors 2018）是天然第二阶段**（仪器带是合成的）。

## 4. 融合结论与建议

### 4.1 两个方法的定位重排（相对既有缺口清单的修正）

| 方法 | 此前认知 | 调研后定位 |
|---|---|---|
| 反褶积 | "商业标配、最大算法缺口" | 确为商业标配（REFLEXW/EKKO/RadExplorer/Geolitix 四款内置）**但** UAV-GPR 文献应用≈0；REFLEXW 自己都说"常不适用 GPR"。定位：**商业对标的"高级可选项"**，实现 v1 蓝本唯一明确：Schmelzbach & Huber 2015 mixed-phase 方案（开源 RGPR 可直接对标参数） |
| inverse-Q | "商业标配缺口之一" | **六款商业软件零命中**——不是对标项而是差异化机会；学术有据（Turner/Bano/Wang 2002），但必须以稳定化为承重墙 |

**优先级建议（修正后）**：deconv 先做（对标缺口 + 方法蓝本唯一 + RGPR 有参考实现），inverse-Q 后做（差异化 + 设计自由度大 + 风险更高）。两者**分开注册**为两个独立方法，不复用同一注册项。

### 4.2 反褶积 v1 设计要点（仅记录，未开工）

1. 方法：逐道最小相位 spike 反褶积（supertrace 合并 11 道稳定化）+ 全局 kurtosis 相位旋转（0–180° 扫描）。
2. 参数契约（对齐 EKKO/Geolitix 口径）：估计时间窗（起止 ns）、算子长度（样本数）、合并道数、prewhitening（默认 0.1）、相位旋转搜索开关。
3. 处理链位置：背景抑制/去噪之后、增益之前；**输出后建议自动跟一次 bandpass**（frequency_filter_1d 已有，无需新增）。
4. 参数缺失显式报错（时间基准教训）；输出 metadata 记录估计的最优旋转角、kurtosis 值、窗参数。
5. 验收：gprMax 合成数据（gprMax fixture 已在 harness 体系内）+ 英山实测对比；fitness benchmark 新增 fixture 时保持 degenerate_output/data_noop 防护。

### 4.3 inverse-Q v1 设计要点（仅记录，未开工）

1. constant-Q (t,f) 域逐道补偿 + Wang 2002 稳定化；参数：Q 值（或自动估计开关+方法选择）、gain_limit_db（默认建议 40，标注 inferred）、噪声底阈值。
2. 位置：时间零位之后、deconv 之前（或替代 deconv）；GPR 材料先验 Q 表需现场标定，UI 不得给出误导性默认值。
3. 风险声明：低 Q 数据需求增益指数爆炸（本会话推导 160dB@Q=20）——参数窗必须硬钳制，超限 taper。

### 4.4 不确定性与待验证项

- 仪器合成等效子波的相位特性（最小相位？混合相位？）未知——用 gprMax 合成 + 英山数据 ACF 检查可低成本验证（pending validation）。
- 带宽窄（20–170 MHz）对 kurtosis 相位估计收敛样本量的放大效应——需在 fixture 上实测。
- inverse-Q 的 Q 估计在共偏移距 UAV 数据上无先例可循——先做固定 Q 手动档，自动估计作为 P2。
- GPR 材料 Q 数值表文献不可得——现场标定流程需要设计（未来工作）。

## 5. 来源清单

### 反褶积
- REFLEXW 官方处理指南: https://www.sandmeier-geo.de/Download/gpr_2d_import_processing.pdf（confirmed）
- REFLEXW deconv 专用指南: https://www.sandmeier-geo.de/Download/deconvolution.pdf（confirmed）
- EKKO_Project 手册 §7.4.3: https://www.sensoft.ca/products/ekko-project/（confirmed）
- RadExplorer 手册: http://www.allstartech.com.tw/pic_mala/software/Radexplorer.pdf（confirmed）
- Geolitix 处理文档: https://docs.geolitix.com/layers/gpr-processing.html（confirmed）
- RGPR mixed-phase 教程: https://emanuelhuber.github.io/RGPR/10_RGPR_mixed-phase-wavelet-deconvolution/（confirmed）
- **Schmelzbach & Huber 2015, IEEE TGRS（全文直读）**: https://emanuelhuber.github.io/RGPR/public/schmelzbach-and-huber_2015_GPR-efficient-deconvolution.pdf（confirmed）
- GPRPy 源码: https://github.com/NSGeophysics/GPRPy（confirmed，无 deconv）
- MDPI UAV-GPR 综述 2022: https://www.mdpi.com/2072-4292/14/14/3245（confirmed，14 系统 0 命中）
- Dupuy ISSW 2024: https://arc.lib.montana.edu/snow-science/objects/ISSW2024_O8.7.pdf（confirmed）
- SBD-SOOT 2025: https://www.sciencedirect.com/science/article/pii/S0926985125002149（confirmed）

### inverse-Q
- Wang 2002, Geophysics: https://doi.org/10.1190/1.1468627（confirmed）
- Wang 2003: https://doi.org/10.1190/1.1543219（inferred）
- Bano 1996 GRL: https://doi.org/10.1029/96gl03010（confirmed）
- Bano 1996 GJI（tanδ≈1/Q）: https://doi.org/10.1111/j.1365-246x.1996.tb06370.x（confirmed）
- Turner 1994, Geophysics: https://doi.org/10.1190/1.1443677（confirmed）
- Du 2022 EPP（正则化 IQF）: https://doi.org/10.26464/epp2022024（confirmed）
- Yang 2025（LCFS Q 体）: https://doi.org/10.3390/app152111504（confirmed）
- Łapinkiewicz 2023（增益函数主观性）: https://doi.org/10.3390/app13158564（confirmed）
- arXiv 2308.08350（迁移型 IQF）: https://arxiv.org/abs/2308.08350（confirmed）
- Sensors 2018（SFCW 逆衰减）: https://doi.org/10.3390/s18051366（confirmed）
- SEG Wiki Inverse Q: https://wiki.seg.org/wiki/Inverse_Q_filtering（confirmed）
- EKKO/REFLEXW 功能清单（无 inverse-Q）: 同上反褶积来源（confirmed）
- 地震增益上限 40–60 dB: 不可考（[INFERRED] folklore，多引擎负搜索）
- 本会话 dB 推导（38/65/160 dB）: [INFERENCE]，实现时须数值验证
