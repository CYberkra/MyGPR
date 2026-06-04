# MyGPR AutoTune 当前选参规则报告（V0.8.39）

## 1. 当前定位

AutoTune 当前定位为“目标倾向驱动的处理流程与参数推荐器”。它生成有限数量的 workflow recipe，并为每一步给出推荐参数。它不是全局最优 workflow 搜索器，也不是专家替代系统。

当前核心闭环：

```text
用户选择目标倾向和范围
→ 数据诊断
→ 背景抑制候选评分
→ workflow recipe 生成与排序
→ 推荐流程与参数
→ 一键运行 recipe
→ 处理链路逐步展开记录
```

## 2. 数据模式

AutoTune 区分两种数据模式：

| 模式 | 触发条件 | 评分边界 |
|---|---|---|
| 有参考响应 | 存在 `target_response` | 可加入 reference similarity / RMSE-like 指标 |
| 无参考标签 | 真实数据或未绑定 target_response | 只使用无先验 proxy 指标，不能解释为真实地下结构正确性 |

真实数据默认属于“无参考标签”。此时推荐结果只能表示当前处理指标下更合适，不能表示更接近真实地下结构。

## 3. 目标倾向与权重

V0.8.39 将用户目标倾向整理为 scoring v2 goal profile。每个 profile 都是一组高层指标权重。

### 3.1 均衡推荐

| 指标 | 权重 |
|---|---:|
| background_suppression | 0.20 |
| response_preservation | 0.20 |
| continuity | 0.16 |
| contrast | 0.16 |
| deep_balance | 0.10 |
| artifact_control | 0.12 |
| stability | 0.06 |

用途：默认一键推荐。兼顾背景抑制、响应保留和可辨识度。

### 3.2 连续界面保留

| 指标 | 权重 |
|---|---:|
| background_suppression | 0.15 |
| response_preservation | 0.22 |
| continuity | 0.28 |
| deep_balance | 0.12 |
| artifact_control | 0.13 |
| stability | 0.10 |

倾向：温和背景抑制、保守带通、宽窗轻度增益。避免把连续界面当背景扣掉。

### 3.3 滑坡基覆界面 / 潜在滑移面

| 指标 | 权重 |
|---|---:|
| background_suppression | 0.13 |
| response_preservation | 0.18 |
| continuity | 0.26 |
| deep_balance | 0.22 |
| artifact_control | 0.12 |
| stability | 0.09 |

倾向：保护深部、连续、弱反射界面；背景抑制采用温和方法；增益采用深部平衡策略。

### 3.4 局部异常增强

| 指标 | 权重 |
|---|---:|
| background_suppression | 0.22 |
| response_preservation | 0.17 |
| contrast | 0.26 |
| deep_balance | 0.09 |
| artifact_control | 0.14 |
| stability | 0.12 |

倾向：增强局部异常、双曲线或高响应区域，但仍保留伪影控制。

### 3.5 裂隙/破碎带保留

| 指标 | 权重 |
|---|---:|
| background_suppression | 0.17 |
| response_preservation | 0.16 |
| continuity | 0.20 |
| texture_preservation | 0.24 |
| artifact_control | 0.13 |
| stability | 0.10 |

倾向：保留断续、散射和纹理响应，避免过度平滑。

### 3.6 含水软弱带

| 指标 | 权重 |
|---|---:|
| background_suppression | 0.14 |
| response_preservation | 0.24 |
| continuity | 0.20 |
| deep_balance | 0.18 |
| artifact_control | 0.14 |
| stability | 0.10 |

倾向：保留衰减、弱反射和带状连续响应，不鼓励强高通锐化。

### 3.7 深部弱反射增强

| 指标 | 权重 |
|---|---:|
| background_suppression | 0.13 |
| response_preservation | 0.17 |
| deep_balance | 0.28 |
| gain_stability | 0.20 |
| artifact_control | 0.14 |
| stability | 0.08 |

倾向：宽窗稳定增益，避免把深部噪声放大成假异常。

## 4. 数据诊断规则

workflow planner 会先计算轻量诊断量：

| 诊断项 | 含义 | 用途 |
|---|---|---|
| drift_strength | 低频漂移强度 | 控制 Dewow 是否偏保守、窗口多大 |
| stripe_strength | 横向背景条纹强度 | 控制背景抑制必要性和候选优先级 |
| continuity | 道间连续性 | 控制界面/层状目标流程适配度 |
| deep_energy_ratio | 深部能量比例 | 控制深部增益和滑坡/深部目标倾向 |
| local_anomaly_density | 局部强响应密度 | 控制局部异常增强流程适配度 |
| spike_ratio | 尖峰比例 | 控制轻度去尖峰流程适配度 |
| target_response_available | 是否存在参考响应 | 控制是否加入参考相似度 |

这些诊断用于 recipe 排序，不直接代表地下结构判断。

## 5. 背景抑制候选规则

当前背景抑制候选包括：

```text
baseline / mean / median / sliding / svd rank sweep
```

但 V0.8.38 起执行以下边界：

```text
baseline 只作为对照，不进入最终推荐流程的背景抑制步骤。
```

也就是说，候选对比中可以显示“不处理基线”，但推荐 workflow 必须包含至少一个真实背景抑制方法：

```text
mean / median / sliding / svd
```

如果 baseline 得分最高，系统不跳过背景抑制，而是选择温和真实方法，并标注：

```text
背景抑制收益较弱，已采用温和背景抑制方法。
```

## 6. 背景候选评分 v2

每个背景候选先计算兼容旧 UI 的旧指标：

| 旧指标 | 含义 |
|---|---|
| roi_retention | ROI 能量保留 |
| residual | 背景残留降低 |
| cnr | 对比度 / CNR 变化 |
| shape | ROI 内形态相似度 |
| rmse | 有参考响应时的 RMSE-like 相似度 |

V0.8.39 新增高层 v2 指标：

| v2 指标 | 含义 |
|---|---|
| background_suppression | 背景区域能量降低程度 |
| response_preservation | ROI 响应能量是否接近原始响应，避免过扣或过放大 |
| continuity | ROI 内形态/连续性保持程度 |
| contrast | CNR 增益经 logistic 归一化后的对比度项 |
| deep_balance | 深部/浅部能量比例是否保持合理 |
| artifact_control | 过度削弱、过度放大、弱抑制等惩罚后的控制项 |
| stability | 按算法类型给出的稳定性先验 |
| reference_similarity | 有 target_response 时加入的参考响应相似度 |

最终背景候选分数：

```text
background_candidate_score = goal_profile_weighted_sum(v2_terms)
```

如果存在 target_response：

```text
final_background_score = 0.82 * goal_profile_score + 0.18 * reference_similarity
```

如果不存在 target_response：

```text
final_background_score = goal_profile_score
```

## 7. Workflow recipe 生成规则

当前采用 bounded recipe templates，不做自由组合搜索。

每个目标倾向有 2 个左右模板。例如：

| 目标 | 示例模板 |
|---|---|
| 均衡推荐 | Raw → Dewow → Bandpass → Background → Gain |
| 连续界面保留 | Raw → Conservative Dewow → Conservative Bandpass → Gentle Background → Mild Gain |
| 滑坡基覆界面 | Raw → Mild Dewow → Low-frequency-preserving Bandpass → Gentle Background → Depth-balanced Gain |
| 局部异常增强 | Raw → Dewow → Wide Bandpass → Stronger Background → Contrast Gain |
| 裂隙/破碎带 | Raw → Dewow → Texture-preserving Bandpass → Gentle Background → Optional Denoise → Mild Gain |
| 含水软弱带 | Raw → Conservative Dewow → Low-frequency-preserving Bandpass → Gentle Background → Stable Gain |
| 深部弱反射 | Raw → Conservative Dewow → Deep-preserving Bandpass → Gentle Background → Wide AGC |

每个模板再与背景抑制候选组合，生成有限数量 recipe。

## 8. Workflow recipe 评分 v2

recipe 分数由以下项组成：

| 项 | 权重 |
|---|---:|
| background_candidate_score | 0.44 |
| workflow_fit | 0.36 |
| compactness | 0.14 |
| target_response_available | 0.06（仅有参考响应时启用） |

其中：

- `background_candidate_score` 来自背景抑制候选 v2 分数。
- `workflow_fit` 来自数据诊断与目标模板的匹配度。
- `compactness` 奖励不过度堆叠处理步骤。
- `target_response_available` 只表示有参考响应时评分依据更充分，不代表真实数据准确性。

## 9. 参数推荐规则

### 9.1 Dewow

规则：

```text
样本数越大，窗口越大；界面/滑坡/含水/深部目标使用更保守窗口。
```

当前输出格式：

```text
window=<auto odd integer>
```

### 9.2 频带滤波

规则：

| 目标 | 推荐倾向 |
|---|---|
| 均衡推荐 | auto |
| 连续界面 | 保守带通 |
| 滑坡基覆界面 | 低频保留 |
| 局部异常 | 偏宽带通 |
| 裂隙/破碎带 | 偏宽/纹理保留 |
| 含水软弱带 | 低频保留 |
| 深部弱反射 | 低频/深部弱反射保留 |

### 9.3 背景抑制

规则：

```text
baseline 不作为最终推荐。
若 SVD 被安全阈值跳过或收益较弱，优先回退到 median / mean 等温和真实方法。
```

SVD 安全阈值：

```text
trace_count > max_svd_traces 时跳过 SVD 候选
```

### 9.4 增益

规则：

| 目标 | 推荐倾向 |
|---|---|
| 均衡 / 局部异常 / 裂隙 | 普通 AGC |
| 连续界面 | 宽窗轻度 AGC |
| 滑坡基覆界面 | 深部保留 / 宽窗 AGC |
| 含水软弱带 | 温和增益 |
| 深部弱反射 | 宽窗稳定 AGC |

### 9.5 轻度去尖峰

仅裂隙/破碎带等部分模板启用，当前映射为轻度平滑/去尖峰，不作为所有 workflow 的默认步骤。

## 10. 一键运行映射规则

AutoTune recipe 不直接执行自由脚本，而是映射到已有处理方法：

| Recipe step | 执行方法 |
|---|---|
| zero_time | 保持当前校正，不写入处理任务 |
| dewow | `dewow` |
| bandpass | `frequency_filter_1d` |
| background: svd | `svd_bg` |
| background: median | `median_background_2D` |
| background: mean/sliding | `subtracting_average_2D` |
| gain | `agcGain` |
| denoise | `running_average_2D` |
| display | display-only，不写入处理任务 |

执行后处理链路会展开每一步，而不是浓缩为一个 AutoTune 步骤。

## 11. 当前限制

1. Workflow 搜索仍是 bounded templates，不是全局自由搜索。
2. 频带滤波目前输出的是推荐倾向标签，不是完整物理频率反演。
3. Real no-prior 数据没有 ground truth，不能评价地下结构真实性。
4. Auto ROI 是启发式 scoring mask，不是目标标签。
5. 当前 scoring v2 已形成后端结构和 breakdown，但后续仍需把报告导出字段进一步闭合。

## 12. 后续建议

下一步建议：

```text
1. 在 UI 候选对比中显示 score breakdown 简表。
2. 在报告导出中写入 goal profile、workflow_score_weights、score_version。
3. 用 gprMax synthetic paired batch 校准各目标权重。
4. 给滑坡基覆界面目标单独做 synthetic benchmark 场景族。
```
