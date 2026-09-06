# MyGPR 处理算法 UAV-GPR 适用性审计报告

> 审计方法：36/36 注册算法在 8 类仿真场景（零时校正/去漂移/杂波增益/包络/F-K/去噪/QC/运动补偿/偏移）下逐方法实测（`scripts/algorithm_fitness_benchmark.py`，子进程隔离、超时熔断、逐点对比），并结合源码根因分析。基准总分 **60.75 / 100**（诚实口径：全零输出记 0 分）。
> 证据输出：`output/autoresearch/algorithm_fitness.json`（含每方法 score + detail）。

> **【修复迭代后更新 2026-09-05】** 按本报告行动清单完成首轮修复：P0 kirchhoff TV 发散（0.0→0.5，§3.1 注记）+ wavelet_svd 签名默认对齐 schema（0.0→0.0154，§3.4）；svd_subspace 初版归因经探针证伪并已修订（§3.4）。定向测试 64 passed + 全量回归无回归；benchmark 复跑总分 **62.18 / 100**（基线 60.75，run#5 keep），证据 `output/autoresearch/algorithm_fitness.json`。

> **【二轮：测量仪器修正 2026-09-05】** benchmark 波形族传参语义修正——`threshold` 在默认 `mad_universal` 策略下是 VisuShrink universal 阈值的**乘数**，而任务表沿用 legacy global_fraction 的分数语义传 0.1/0.05，等效 0.1×VisuShrink 几乎不去噪。修正为 canonical 乘数 1.0（方法代码零改动）：wavelet_2d 0.0286→**0.132**（dSNR +3.2dB）、wavelet_svd 0.0154→**0.133**；两者输出相关性升至 0.9997——**SVD 前置在该场景无增益**，wavelet_svd 独立价值存疑。总分 **62.80 / 100**（run#6 keep，置信 3.3×噪声底）。
> 审计日期：2026-09-05。分支：`feat/phase2-velocity-grid`。修复迭代改动：`mygpr/infrastructure/processing/algorithms/kirchhoff/{shared,gpu}.py`（TV canonical 化 + GPU kernel 补齐）、`.../extended/wavelet.py`（签名默认对齐 schema）、`scripts/algorithm_fitness_benchmark.py`（波形族阈值乘数语义修正）。
> **【三轮：参数契约硬化 2026-09-05】** time_cut samples-as-ns 静默回退移除（§3.3）：真实裁剪缺时间基准（time_window_ns/time_step_s 均无）时抛 ValueError（中文报错指明补救路径），替代原先"把采样点数当 ns"的错量纲回退；回退链新增 header_info（total_time_ns/time_window_ns，native 路径 _runtime_kwargs 已注入）；文档承诺的 no-op（time_end_ns=0）不再要求时间基准，且 **不再写 header_info_updates.total_time_ns**——消除旧版 no-op 也向链式 header 写入 samples-as-ns 垃圾值、毒化下游 kirchhoff/rtm 时窗的连带缺陷。同族 common.py:154 samples 兜底记 P3（真实管线被 prepare_runtime_params 覆盖，仅直调场景可达）。定向回归 42+10+18 passed，4 个新回归测试；benchmark 总分 **62.80 / 100**（run#7 keep，置信 6.6×噪声底；benchmark 传参显式，分数不变）。
> **【四轮：契约硬化收尾 2026-09-05】** `set_zero_time` 两处 `48e-9/max(1,ny)` 静默猜测步长回退移除（§3.3，run#8 keep）：ndarray 路径（`PythonModule/set_zero_time.py:_resolve_time_step_s`）与 native 路径（`mygpr/infrastructure/processing/algorithms/basic.py:method_zero_time`）缺显式 `time_step_s>0` 且 header 无时间基准（total_time_ns/time_window_ns）时，均抛中文 ValueError（指明补救路径，与 time_cut 同风格）；`new_zero_time<=0` no-op 前置化（无需时间基准即可恒等返回，替代旧版"先猜步长再判断 no-op"）。native 路径新增 `_header_info`/`header_info` dict 回退取 total_time_ns/time_window_ns 换算 step。`frequency_filter_1d` 报告行**过时修订**：现实现已显式 skip+中文 warning（`PythonModule/frequency_filter_1d.py:48-58`、`algorithms/frequency.py:22-29`，warning id `frequency_sampling_missing`），且 sample_rate 由管线三处注入（processing_engine/native_adapter/block_executor），本表原描述"静默直通"不成立，未改码。新增 5 个回归测试（ndarray 3 + native 2）。benchmark 总分 **62.80 / 100**（run#8 keep，真实管线三处注入 time_step_s，分数不变符合预期）。定向 60+19 passed、全量 815 passed/6 skipped。注：本轮起分支已切至 `feat/phase3-depth-slice`（用户 velocity WIP 已自行提交），autoresearch 会话在新分支重建，基线即含三轮全部硬化改动。


---

## 0. 总览

| 类别 | 数量 | 方法 |
|---|---|---|
| 冗余（数值等价） | 1 对（+1 对代码级等价） | subtracting_average_2D ≡ ccbs；（speed ≡ equidistant） |
| 可删除候选 | 1 | ccbs（保留 subtracting_average_2D 的情况下） |
| 已有但有问题 | 10 | kirchhoff、attitude、trajectory_smoothing、set_zero_time、time_cut、frequency_filter_1d、equidistant、speed、vibration、弱去噪 5 件套 |
| 缺少 | 6 类 | deconvolution、inverse-Q、airwave 抑制、速度谱分析、C-scan、极化 |
| 表现健康 | 19 | background 6 件、gain 5 件、fk_filter、hilbert、dewow、time_cut(功能)、trace_qc、height/v2、rtm、stolt(勉强)、time_to_depth |

---

## 1. 冗余（数值等价，corr=1.0000）

### 1.1 `subtracting_average_2D` ≡ `ccbs` — **确认冗余**

- **实测**：同一 drift 场景（84 道），两方法输出逐点最大差 **1.8e-8**（float32 存储噪声级），相关系数 **1.0000**。
- **根因**：ccbs 的 NCC 加权参考道机制（`mygpr/infrastructure/processing/algorithms/extended/ccbs.py:66-77`）在 `reference=None` 时退化为均值参考 → `_subtract_weighted_background`（:46-63）在数学上与滑动均值背景去除完全一致。benchmark 的 84 道输入小于 `ntraces=501` 滑窗，两者都退化为全局去均值。
- **语义差异只有一处**：ccbs 支持自定义参考道 `reference_wave`——但注册表（`core/method_registry_groups/background_denoise.py:525-537`）只暴露 `use_custom_ref` bool，未暴露参考道选择，**GUI 用户实际无法触达该差异**。
- **判定**：在当前注册表暴露面下，ccbs 是 subtracting_average_2D 的重复实现。

### 1.2 `motion_compensation_speed` ≡ `equidistant_trace_resample` — 代码级等价（非数值冗余对）

- 两者走同一核心：`build_uniform_trace_distance_m` + `resample_bscan_columns_linear`（`mygpr/infrastructure/processing/algorithms/motion/speed.py`）。
- 分数完全一致（0.0143）。等距重采样改变道数（95 vs 96），因此 harness 的逐点对比不入冗余表——但源码层面是同一功能的两个入口。
- **建议**：保留一个入口，另一个改为薄别名或合并参数。

---

## 2. 可删除候选（删除与否由用户决策，需先核对 `core/method_registry_metadata.py` + `config/schema_catalog.json`）

| 候选 | 理由 | 保留理由 |
|---|---|---|
| **ccbs** | §1.1 数值等价 + 注册表未暴露其唯一语义差异 | 未来要暴露 reference_wave（自定义参考道去水平层）时可恢复 |
| ~~其余~~ | 无 | 其余 35 个方法均无"数值等价且暴露面相同"的删除依据 |

其余低分方法（wavelet_svd、attitude、trajectory_smoothing 等）属于**"有问题/无效果"而非"冗余"**，处置路径是修复或重新定位，不是删除（见 §3）。

---

## 3. 已有但有问题（按严重度排序）

### 3.1 【严重】`kirchhoff_migration` 默认参数输出全零

- **实测**：benchmark 合理参数（freq=2e7、depth=8、time_window_ns=160）下，默认 `weight=0.5` 输出**全零图**：输入能量 459 → 输出 0.0，`degenerate_output=true`，得分 0.0。
- **根因链**：`_postprocess_kir_profile`（`mygpr/infrastructure/processing/algorithms/kirchhoff/shared.py:302-310`）→ `_denoise_tv_bregman`（:319-403）在 `weight ≥ 0.4` 时发散：`tx**2` overflow（shared.py:383）→ inf → NaN → 全零。
- **weight 扫描实测**：

| weight | 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.35 | ≥0.4 |
|---|---|---|---|---|---|---|---|
| 输出能量 | 6831 | 6195 | 6078 | 6013 | 10112(不稳) | 11754 | **0（全零）** |

- **默认参数另有效率问题**：默认 freq=5e7、depth=40 时成像网格 dx=dz=c/(60·freq)（shared.py:167-193, 232-271）单次 >300s。
- **修复建议**：① `weight` 默认改为 0.0–0.2 并在 docstring 标注 ≥0.4 发散；② TV 迭代内对 `tx/ty` 加 clip 或用 float64 归一化输入；③ 成像网格分辨率与时间窗挂钩而非固定 c/(60f)。

> **【2026-09-05 修复后注记】** TV 发散已修复（`shared.py:_denoise_tv_bregman` 重写为标准 split-Bregman）。
> 真实根因是三重缺陷，而非单纯 overflow：(1) u 更新式把数据项放在 `lamda` 乘法括号内且 shrinkage 阈值为 1.0（与数据项权重不匹配）→ 正反馈几何发散（w=0.3 时 |u|max 2.2 → 1.27e12 @100it → 1.18e77 @600it）；(2) `u_prev` 取 `image_padded[1:-1,1:-1,:]` 是视图，写回后 residual 第一项恒 0，收敛判据失效；(3) 尾部 min-max 重归一化把任何发散输出映射回输入值域——**"range 正常无 NaN" 对收敛性零信息量**，直到 float64 overflow → NaN → 下游全零。修复后 weight 扫描 {0, 0.2, 0.35, 0.4, 0.5, 0.8, 2.0} 全部有界收敛（|u|max ≤ 0.38），benchmark 得分 0.0 → **0.5**（apex_concentration=1.0）。
> 同步修正：两个 GPU kernel（`gpu.py:_denoise_tv_bregman_gpu` / `_fast`）原递推缺少 Jacobi 扫描必需的四邻居（Laplacian）项（属"稳定但不精确"近似），已补齐为与 CPU 权威实现同一递推（周期边界 vs 对称边界）；u 更新前从未使用的 `ux/uy` 死代码已删。GPU 侧因本机无 CUDA 未实测，经 numpy 逐字镜像验证有界收敛 + 静态结构核验（邻居项齐全、dxx→axis=1 / dyy→axis=0 与 CPU 约定一致）。
> 上表旧扫描值建立在发散/掩蔽输出上，仅作历史证据，不再代表修后行为。
> **【2026-09-05 run#13 注记】** 上述"得分 0.5"是 `weight=0.5` 下的分数——TV 收敛性已修复但**默认权重本身失谐**：migration fixture 实测 weight 扫描 {0.5, 0.35, 0.2, 0.05} → score {0.50, 0.58, 0.73, **0.96**}（apex_concentration 恒 1.0，flatten 分量 0→0.92；w=0 完全不去噪 flatten=0.0）。默认 `weight=0.5` 过度平滑，把绕射尾巴当噪声抹平。run#13 已将默认 **0.5 → 0.05**（轻度 TV）全链硬化：native schema（`methods.py`）、wrapper 签名（`kirchhoff/__init__.py:92`）、legacy registry（`method_registry_groups/imaging.py`）、descriptor_baseline fixture；benchmark `KIRCHHOFF_SENSIBLE` 改为 weight 空参化测真实 schema 默认。0.05 三种子稳健（0.9595–0.9657，全 finite），kirchhoff 得分 0.5 → **0.9608**，fitness 65.12 → **66.40**。

### 3.2 【高】运动补偿族的两个"数据 no-op"节点

- **`motion_compensation_attitude`（0.0 分）**：数据矩阵原样拷贝返回（`motion/attitude.py:64`），仅写 footprint/local_x_y metadata（:196-204）。实测 probe19：`data identical: True`，updates 只含 footprint_x_m/footprint_y_m/local_x_m/local_y_m/trace_distance_m。**UI 名为"姿态补偿"，实际不做任何振幅/几何重采样**——名实不符。它写出的 trace_metadata_updates 是否被下游消费（如成像几何）需要确认；若无下游消费，则该节点在处理链中是纯装饰。
- **`trajectory_smoothing`（0.0 分）**：同为数据 no-op（`motion/trajectory.py:99, 238`），仅平滑 lon/lat。实测位移量：max 0.043 m、mean 0.023 m——在 0.09 m 道距下属于亚道级修正，效果存在但极小，且不回写 B-scan。
- **处置建议**：两者应明确标注为"几何元数据准备节点"（或从处理链中作为可选前置步骤），不应与真正做数据校正的 height/v2 并列展示为"补偿"。

### 3.3 【中】参数契约静默回退（易产生"看起来成功"的错误结果）

| 方法 | 问题 | 位置 |
| `frequency_filter_1d` | 缺 `sample_rate_mhz` 时静默直通（不处理、不告警）。**行过时（四轮修订）**：现实现已显式 skip+中文 warning（`frequency_sampling_missing`），sample_rate 由管线三处注入，未改码 | `PythonModule/frequency_filter_1d.py:48-58`、`algorithms/frequency.py:22-29` |
| `set_zero_time` | 缺 `time_step_s` 时用默认步长（48ns/采样点数猜测），实测 first_break=0.042（分数 0.0415，接近乱猜）。**已修复（四轮）**：ndarray 与 native 双路径猜测回退均移除，缺基准抛中文 ValueError；详见 §3.3 修复后注记 | `PythonModule/set_zero_time.py`、`algorithms/basic.py` |
| `equidistant_trace_resample` | 缺 `trace_distance_m` 直接抛 ValueError（硬契约，与 time_cut 的静默回退形成两个极端） | motion 路径 |

**建议**：统一契约——缺关键参数时要么有明确 fallback 依据（如读 header），要么抛出带参数名的 ValueError；禁止 samples-as-ns 这类量纲错位回退。

> **【修复后 2026-09-05 四轮】** `set_zero_time` 已修复（run#8）：`PythonModule/set_zero_time.py:_resolve_time_step_s` 与 `mygpr/infrastructure/processing/algorithms/basic.py:method_zero_time` 两处 `48.0e-9/max(1,采样数)` 静默猜测回退均移除——缺显式 `time_step_s>0` 且 header（`_header_info`/`header_info` dict 的 `total_time_ns`/`time_window_ns`）亦无时间基准时抛中文 ValueError（列明补救路径）；`new_zero_time<=0` no-op 前置化，恒等返回不再要求时间基准。真实管线三处注入通道核实安全（core `prepare_runtime_params`、native `prepare_native_params`、block `prepare_block_params`，header total_time_ns0>0 时均注入）。`frequency_filter_1d` 无需修复：报告行过时（见上表修订）。5 个新回归测试（`tests/test_round2_processing_kernels.py` 3 + `tests/test_native_processing_backend.py` 2）。benchmark 显式传 `time_step_s=3.665e-9`，总分不变 62.80（run#8 keep）。legacy CSV 包装器自算步长传入，行为不变。


> **【run#12 2026-09-05 八轮】** trace_qc 默认陷阱修复：schema 默认 `spike_zscore=0.0` = 尖峰检测**完全禁用**（空参 score 0.0 vs 显式 1.0，与 wavelet threshold 同类的"默认失效"陷阱）。默认改 **6.0**（MAD 稳健 z-score，clutter fixture 3 seeds TP=5/FP=0 全对；z=4 在部分 seed 有 FP，6 是稳健边界）全链同步 5 处——`methods.py` schema（补 min=0）、`extended/trace_qc.py` 签名+`to_float` fallback、`calibration.py` legacy 注册表 default+label（"Spike z-score (0=off)"）+tooltip、`descriptor_baseline.json`。`empty_rms_threshold` 是 RMS 绝对量纲（随数据幅值缩放），0=off 语义合理保留。benchmark 仪器修正：trace_qc 显式传参 {} 化改测真实默认（分数 1.0 维持）；fk_filter 任务表 `angle_low_deg`/`angle_high_deg` 是死参数（native 读 `angle_low`/`angle_high`，显式值被静默忽略但恰等于默认 10/65）→ {} 化，属仪器诚实性修正分数不变。hilbert `normalize` schema False vs benchmark True 探针同分 1.0，无陷阱不改。其余 18 处显式传参 vs schema 差异均为 fixture 地面真值契约（set_zero_time/time_cut/amplitude_scale/frequency_filter_1d/stolt/time_to_depth/motion C=0.1），不改。定向 96 passed/1 skipped；全量 818 passed/6 skipped（基线 +3 trace_qc 默认行为测试）；fitness 65.12 保持。

> **【修复后 2026-09-05 三轮】** `time_cut` 已修复（run#7）：真实裁剪（remove_below/above、keep_range 任一带非零时间参数）缺时间基准时抛 `ValueError`（中文报错列出补救路径），替代 samples-as-ns 回退；新增 `header_info`（`total_time_ns`/`time_window_ns`）回退层（native 路径 `extended/_runtime_kwargs` 已注入，core 路径经 `prepare_runtime_params` 注入 time_step_s）；文档承诺的 no-op（`time_end_ns=0`）不再要求时间基准且**不再写 `header_info_updates.total_time_ns`**，消除 no-op 向链式 header 写垃圾值、毒化下游 kirchhoff/rtm 时窗的连带缺陷。同族 `common.py:154`（`resolve_time_selection`）的 samples 兜底记 **P3**：真实管线被 `prepare_runtime_params`（header total>0 时注入）覆盖，仅直调/无 header 场景可达。4 个新回归测试（`tests/test_round2_processing_kernels.py`）。

### 3.4 【中】弱去噪五件套（denoise_snr 场景 SNR −1.7 dB）

| 方法 | 分数 | 根因 |
|---|---|---|
| `wavelet_2d` | 0.0286→**0.132** | 初版归因（阈值 0.045 太松、层数 2 太少）只对了一半。真实根因：benchmark 任务表传 `threshold=0.1` 沿用 legacy global_fraction 分数语义，而默认 `mad_universal` 下它是 VisuShrink 乘数——0.1×VisuShrink 几乎不去噪（+0.7 dB）。修正乘数 1.0 后 +3.2 dB；乘数 ≥1.0 后 SNR 饱和（1.0/1.5/2.0/3.0 同值），levels 2→4 仅 +0.2 dB。**方法实现无 bug**（mad_universal + soft 是标准 VisuShrink） |
| `wavelet_svd` | 0.0154→**0.133** | 初版"细节系数存活 70%"归因同样源于乘数语义错配（0.05×VisuShrink）。乘数修正后与 wavelet_2d 输出相关性 **0.9997**——SVD 前置（作用于近似系数）在该低秩场景**无独立增益**，两方法实质冗余；schema 默认 threshold=0.05 对 GUI 用户同样是语义陷阱（0.05×VisuShrink），建议默认改 1.0 或文档标注乘数含义 |
| `svd_subspace` | 0.0458 | **初版归因有误，已探针证伪**：benchmark 显式传 `rank_start=1`（`algorithm_fitness_benchmark.py:265`），native 实现无 bug。真实根因：`rank_end=20` 保留过多秩——fixture clean 奇异谱 [25.7, 12.4, 3.9, 2.2, 0.6…]（k=2 占 96.4% 能量），噪声注入后 σ₃₋₅≈8.8/8.65/8.31 全是噪声；保留秩 1–20 仅 +1.1 dB，**最优截断 k=2 可达 +10.0 dB**。计分口径确认 `10·log₁₀(mean(clean²)/mean(err²))` |
| `trace_savgol_filter` | 0.2192 | 道内平滑对非平稳噪声无效 |
| `hankel_svd` | 0.3090 | 有一定去噪但不及 trace_median (0.7365) / running_average_2D (0.6645) |

  **run#9 修复（2026-09-05 五轮）**：默认 `rank_end` 20→2（methods.py schema、background_denoise.py 注册表、PythonModule wrapper 40→2 三处同步；wavelet_svd 内嵌默认同步），auto_tune_candidates 收窄为 [2,3,4,5,6,8]，benchmark 任务表移除显式 `rank_end=20` 改测真实默认行为。多 fixture×多 seed 扫描（drift/clutter/zero × 3 seeds）确认 rank_end=2 在 2/3 fixture 最优（+12.4/+5.4/+5.7 dB vs rank_end=20 的 +5.1/+1.1/+3.1 dB），zero fixture 最优为 3 但 2 仍显著优于 20。**改默认属 schema/契约修正，未触碰算法数学**。

  **run#10 修复（2026-09-05 六轮）**：wavelet 两方法 schema 默认 `threshold` 0.1/0.05 → **1.0**（VisuShrink 乘数语义饱和点，clip 上限）全链同步 9 处——`methods.py` schema×2（并补 min/max=0/1）、`background_denoise.py` legacy 注册表 default+label（"Threshold (VisuShrink multiplier, 0-1)"）+ auto_tune_candidates 收窄 [1.0]×2、native 签名×2（`extended/wavelet.py`）、`descriptor_baseline.json` fixture、autotune `candidate_planner.py` fallback 0.05→1.0、`constraints.py` 新增 wavelet_svd threshold clamp（>1.0 钳到 1.0 并发 constraint warning）。3 seeds 扫描确认 1.0 稳健最优（wavelet_2d +3.05~3.18 dB、wavelet_svd +3.43~3.52 dB vs 0.8 的 +2.65~3.08/+2.99~3.08 dB）。benchmark 任务表移除显式 `threshold=1.0`/`levels`/`wavelet` 改测真实默认行为（仪器修正）：空参分数 wavelet_2d 0.0286→**0.1323**、wavelet_svd 0.0222→**0.1455**，fitness 65.12 维持（确定性复现两次）。**motion 族 `wave_speed_m_per_ns` benchmark 常量 C=0.1 与 fixture 注入值一致，判定为 fixture 契约而非仪器失配——schema 默认 0.2998（空气波速）对真实 UAV 数据物理正确，不改**（§5 motion 行补充说明）。

**建议（2026-09-05 二轮修订，部分已落地）**：svd_subspace 的 `rank_start` 语义无 bug（保持 1），瓶颈是 `rank_end=20` 过宽——建议默认收窄（如 5）或引入能量截断准则（累计能量 ≥95%），GUI 调参范围不变；wavelet_svd 建议要么删除（与 wavelet_2d 相关性 0.9997，§1 冗余判据 corr≥0.999 同档）、要么给 rank 截断独立调参价值（如对细节系数也做 SVD）；~~两方法 schema 默认 `threshold`（0.1/0.05）对 mad_universal 策略是乘数语义，GUI 用户极易误配——建议默认 1.0 或在 UI 标注"VisuShrink 乘数"~~ ✅ **已落地（run#10）**：默认 1.0 + UI label 标注"VisuShrink multiplier"。这五个方法在 UAV-GPR 典型低 SNR 数据上不可作为主去噪路径。
### 3.5 【低】其余


- `set_zero_time`（旧 0.0415）：✅ **run#15 仪器修正（2026-09-05 十一轮）**。旧口径在 after 侧仍按原零位 `ref_zero_idx=18` 测量，但正确零化后波形上移 `shift_samples=18` 行、新零位在第 0 行——于是理想零化输出测得 fb=0.033（尾区梯度）+ pe=0.796（旧零位之前全是被上移上来的深部信号）→ 0.0415 分，而"什么都不做"得 0.776：**仪器倒置，奖励不作为、惩罚教科书正确行为**。修法（benchmark 仪器，非算法码）：after 侧分量改在**新零位 `ref_zero_idx − shift_samples`**（clamp ≥0；shift 未知按方法契约回退 0）测量，before 侧保持原零位配对语义。验证：任务指定零化（shift 18）→ 0.9717（fb 0.377@row0、pe 0.0）；no-op 控制（`new_zero_time=0`，shift 0）→ 0.7757@idx18（回落原位测量）。fitness 66.77→**69.36**。首次尝试曾把 `shift_samples` 本身当测量行（旧坐标位移量≠新坐标行号），advisory blocker 拦截后修正。
- `motion_compensation_speed` / `equidistant_trace_resample`（0.0143）：方法本身正常，但仅当轨迹严重不等距时才有意义。
- ~~`stolt_migration`（0.5386）：可用但 apex 聚焦远逊 rtm（0.9837）。~~ ✅ **run#14 已修（2026-09-05 十轮）**：默认分低的主因是 `stolt_obliquity_power=0.05` 把教科书 Stolt 倾角/斜射幅度因子 kz/kmag（Margrave/Yilmaz 标准公式 D′(kz,kx)=D(w,kx)·(v/2)·kz/kmag）几乎完全抹掉，而 `stolt_jacobian_power=0.05` 施加的 kz/ω 滤波与 obliquity 同形（双重倾角滤波）。默认改为 **obliquity=1.0 / jacobian=0.0** 全链五处同步（native 实现、PythonModule 包装器、legacy 注册表、native schema 曝光、descriptor baseline fixture）。stolt 0.5386→**0.6716**（apex_amp −15% 更符合点目标物理）。双反射层场景（flat+3° dip）验证输出均匀 −10% 衰减，非 fixture 过拟合。与 rtm（0.9837）的剩余差距裁定为**机制差异非缺陷**：裸 float64 f-k 参考实现（无任何 knob）在同一 fixture flatten 仅 0.149——f-k 映射的绕射裙边（skirt 能量 ~10× 输入）是有限孔径+掩膜截断的固有物理；rtm 高分部分源于时域绕射叠加 + 迁移指标奖励"能量饥饿"输出（rtm apex_amp 仅 0.08），f-k 保守幅度天然吃亏——此为 fixture/指标性质注记，不改 benchmark。

---

## 4. 缺少（UAV-GPR 标配但全库零命中，grep 证据）

| 缺失能力 | 说明 | 现状 |
|---|---|---|
| **Deconvolution（反卷积）** | GPR 处理链标配（压缩子波、提分辨率）：预测反卷积/稀疏脉冲/spiking | 全库无实现 |
| **Inverse-Q 补偿** | 校正频率依赖衰减，UAV 浅层高损场景价值高 | 无 |
| **Airwave/Direct-wave 专用抑制** | 仅 `PythonModule/subtracting_average_2D.py:26-27` 注释提及"可去除水平到达（如 airwave）"——是背景去除的副产品，无专门建模（如时空窗+极化联合） | 无 |
| **速度分析谱** | 常规 CMP/速度谱/扫描叠加速度分析；现有只有 `mygpr/domain/velocity/` 双曲线拟合 service（错误码、证据 schema 已定义）。UAV 共偏移距单覆盖数据确实做不了 CMP 谱，但共偏移距速度扫描（利用已知离地高度反演）是可行替代 | Phase 2.1 roadmap |
| **C-scan（深度/时间切片）** | 无 cscan/depth_slice/time_slice 实现；路线图 Phase 3.1（深度滑条+等值线，复用 basal_interface_annotations + build_georeference_3d） | Phase 3.1 roadmap |
| **极化/阵列处理** | 无 | 无需求可延后 |

---

## 5. 表现健康（无需动作）

- **background 族**（0.82–0.94）：subtracting_average_2D / median / sliding_avg / svd_bg / rpca 全部有效。
- `fk_filter` 0.9677、`hilbert_envelope` 1.0、`dewow` 1.0、`trace_qc` 1.0（F1=1.0）、`time_to_depth` 0.9285、`rtm_migration` 0.9837（偏移首选）、`kirchhoff_migration` 0.9608（run#13 weight 0.05 后）、`stolt_migration` 0.6716（run#14 obliquity 1.0 后，详见 §3.5）、`set_zero_time` 0.9717（run#15 仪器修正后，详见 §3.5）。
- `motion_compensation_height`/`v2` ≈0.59：核心流程有效，精度受 fixture 轨迹噪声限制。

---

## 6. 行动清单（优先级）
2. ~~**time_cut samples-as-ns 回退**（P1，3.3）~~ ✅ **已完成（2026-09-05 三轮）**：真实裁剪缺时间基准抛 ValueError + header_info 回退链 + no-op 不再写 header 污染；详见 §3.3 修复后注记。
   - 附：**set_zero_time 同族回退已完成（2026-09-05 四轮，run#8）**：两处 48ns 猜测步长移除 + no-op 前置 + header 回退，见 §3.3 四轮注记；**frequency_filter_1d 行过时已修订**（skip+warning 已实现，未改码）。
3. ~~**svd_subspace 默认 rank_start**（P1，3.4）~~ ✅ **已完成（2026-09-05 五轮，run#9）**：归因修正 + 默认 `rank_end` 20→2 全链同步（schema/注册表/wrapper/native fallback/fixture），auto_tune_candidates 收窄 [2,3,4,5,6,8]，benchmark 显式传参移除；score 0.0458→0.8695，fitness 62.80→65.12。详见 §3.4 run#9 注记。
4. ~~**删 ccbs 或暴露 reference_wave**（P2，§1/2）~~ ✅ **已完成（2026-09-05 十四轮）**：用户裁定"暴露 reference_wave"。ccbs 注册表（`core/method_registry_groups/background_denoise.py`）现暴露 `reference_wave`（float 数组，默认空=均值参考，GUI 可选自定义参考道）；ccbs 与 subtracting_average_2D 从此语义可区分，冗余判定解除。
5. **attitude/trajectory_smoothing 重新定位**（P2，3.2）——改名或并入"几何准备"分组。**提案已交付（2026-09-05 十四轮，见会话记录），待用户决策后实施。**
6. **Deconvolution 立项**（P2，§4）——UAV-GPR 处理链最大缺口。**立项 spec 已交付（2026-09-05 十四轮，见会话记录），待用户评审后进入实施。**
7. **wavelet_svd 处置**（P2，3.4 二轮）——与 wavelet_2d 输出相关性 0.9997（达到 §1 冗余判据同档），建议删除或重定义其 SVD 作用域（细节系数也做 SVD）以恢复独立价值。用户决策。（注：其 threshold 默认陷阱已由 run#10 修复，冗余性本身不受影响。）
8. ~~**speed/equidistant 合并入口**（P3）~~ ✅ **已完成（2026-09-05 十四轮）**：保留 `motion_compensation_speed` 唯一入口；`equidistant_trace_resample` 移除注册暴露并删除两文件（PythonModule + extended），8 处源码触点迁移，测试与 fixture（35 方法）同步，定向 102 passed 1 skipped。
9. ~~**剩余显式传参 vs schema 默认失配盘点**（P2，3.3）~~ ✅ **已完成（2026-09-05 八轮，run#12）**：18 处差异逐一判定——真失配仅 trace_qc `spike_zscore=0.0`（检测禁用陷阱，已修 6.0）；fk_filter 死参数名已从 benchmark 移除；hilbert normalize 无陷阱；其余 14 处为 fixture 地面真值契约不改。失配类问题至此穷尽。
10. ~~**kirchhoff_migration 默认 weight 0.5 过度平滑**（P1，3.1 run#13）~~ ✅ **已完成（2026-09-05 九轮，run#13）**：调查裁定 §3.1 原报的 tx**2 overflow 已由 canonical split-Bregman 重写消除（探针 weight 0–5.0 全 finite）；真实问题是 weight=0.5 抹平绕射尾巴（flatten=0.0）。默认 **0.5→0.05** 全链硬化 + benchmark 空参化，kirchhoff 0.5→0.9608，fitness 65.12→66.40。默认参数效率问题（dx=dz=c/(60f) 单次 >300s）仍留给用户决策（KIRCHHOFF_SENSIBLE 已用合理网格）。[2026-09-05 十四轮：默认参数耗时问题已由 note 14 实施 (a)+(b) 大幅缓解，网格 factor=60 仍留用户决策。]
11. ~~**stolt_migration 默认缺失教科书 obliquity 因子**（P1，3.5 run#14）~~ ✅ **已完成（2026-09-05 十轮，run#14）**：默认 `stolt_obliquity_power` 0.05→**1.0**、`stolt_jacobian_power` 0.05→**0.0** 全链五处同步（native global_spectral.py、PythonModule 包装器、legacy 注册表 imaging.py、native schema 曝光 knobs、descriptor_baseline.json fixture）。stolt 0.5386→**0.6716**，fitness 66.40→**66.77**。剩余 gap vs rtm 裁定为机制差异（f-k 映射 skirt 物理 + 指标奖励能量饥饿输出），详见 §3.5 注记。至此全部自动可迭代项穷尽，余项均为用户决策（#4/#5/#6/#7/#8 + kirchhoff 网格效率）。
12. ~~**set_zero_time 评分仪器倒置**（P1，benchmark 仪器）~~ ✅ **已完成（2026-09-05 十一轮，run#15/16）**：旧口径 after 侧在原零位测量，理想零化得 0.0415、不作为得 0.776（倒置）。修正为在**新零位 `ref_zero_idx − shift_samples`** 测量（clamp ≥0，shift 未知回退 0）；`set_zero_time` 0.0415→**0.9717**，fitness 66.77→**69.36**（确定性复现 ×2）。纯仪器修正，算法码零改动。

13. ~~**autotune `_score_zero_time` 倒置（与 run#15 同根因的应用侧实例）**（P1）~~ ✅ **已完成（2026-09-05 十三轮，应用侧）**：run#15 修正的是 benchmark 仪器；本项修正 autotune 引擎自身的同型倒置——`mygpr/application/autotune/scoring.py::_score_zero_time` after 侧 `pre_zero_energy_ratio`/`first_break_sharpness` 固定在原零位测量，set_zero_time 正确执行（波形上移 shift 行）后被测到 pre_zero≈0.8 → 重罚，不作为反而得分更高（scene_zero_time 探针：correct(18)=−4.51 < no-op(0)=+0.42）。修复：按 set_zero_time 执行契约（basic.py `result[:-shift]=arr[shift:]`）在 scorer 内推导 `shift=round(new_zero_time/step_ns)`、`after_zero_idx=max(0, zero_idx−shift)`，after 侧改在新零位测量；before 侧保持原零位配对；params 缺 `new_zero_time` 时回退 shift=0（安全降级）；metrics 新增 `after_zero_idx` 便于审计。修复后 scene_zero_time 全候选排序单调（no-op 0.42 < 27 头 0.68 < 9 尾 0.48 < 正确 18 → **8.03**）；端到端候选链（8 候选 × NativeProcessingExecutor 真执行）最优 = 61.98ns（idx 17，fixture 真值 18）——自动选参首次与地面真值一致。定向回归 autotune/quality 20 passed；全量 813 passed / 6 skipped / 5 failed（5 个均为 C 盘 TEMP 满的 InsufficientProcessingStorage 环境失败，基线同模式）。
14. **kirchhoff 默认参数 90.6s 阶段剖析（2026-09-05 测量 → 2026-09-05 同日实施 (a)+(b)+cancel）**：full 默认参数（freq=5e7, depth=40, length_m=None→traces−1, twn=header total_time_ns）端到端 **90.6s**（imaging grid 400×950, nt_matrix=4200, 950 shots）。分阶段：**TV 后处理 56.4s（62%）**（`_denoise_tv_bregman` max_iter=1000/eps=1e-6 在 400×950 上不收敛跑满），travel-time 表 10.5s（12%，纯 Python 双层循环 eikonal `_time2d` 逐格 11×11 matvec）、stack 10.7s（950 shots × 11ms fancy-index gather）、zoom 0.1s、其余 ≈0。合理参数（KIRCHHOFF_SENSIBLE: freq=2e7, depth=8）全链仅 0.16s（TV 30%、travel 19%）。
    **⚠ 实施前实测修正本条原论断**：travel-time 表**并非与 v 值无关**——常速模型 v=0.1 vs 0.2998 实测 max diff 1.29e-7（round 后索引数组不同）；`smooth2a` 对常数矩阵亦非 bitwise 恒等（4.2e-17）。缓存 key 必须含模型内容。
    **实施结果（2026-09-05 十四轮）**：(a) TV 预算默认 1000/1e-6 → **100/1e-4**（`_postprocess_kir_profile` 参数化，cpu/gpu 两调用点同步；400×800 实测 vs 旧预算 corr **0.992**、TV 阶段 36.9s→4.0s **9.1×**）；(b) `_compute_travel_time` 模块级内容哈希缓存（key=sha1(round(model.T,12)+dx)，LRU=4，返回 `.copy()` 防污染；实测同参二跑走时阶段≈0，warm/cold 1.3×@200×96 审计网格 37.0→28.2s，输出逐位一致）；(c) 三阶段 cancel 全通（走时表 `_time2d` 既有 + stack 既有 + TV 新增逐 32 迭代检查，pre-cancel 0.00s / stack 阶段 10.7s 触发）。benchmark migration 场景 digest 因 TV 预算变化而变（预期），score 0.9608→**0.9581**（≥0.95 通过）；golden bitwise 测试（weight=0 → TV 跳过）不受影响。默认网格 factor=60 未动（(d) 仍留用户决策）；`_time2d` Numba 化 (c) 未做。
