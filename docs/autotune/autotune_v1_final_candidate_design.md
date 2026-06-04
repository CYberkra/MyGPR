# MyGPR AutoTune V1.0 Final Candidate 设计定稿

**版本**：AutoTune V1.0 Final Candidate  
**日期**：2026-06-04  
**定位**：设计定稿，不等同于已完成生产版。生产版必须经过 synthetic paired batch 校准和回归审计后再冻结默认权重。

---

## 0. 结论

本方案可以作为 MyGPR AutoTune 的 V1.0 final candidate。它不是“全局最优自动流程”，而是：

> 在固定 workflow / recipe 边界内，基于目标倾向、有限候选参数、synthetic paired 评分和 real no-prior 风险提示，生成可复现、可审计、可导出的参数推荐。

需要冻结的不是某个算法结果，而是以下协议：

1. 目标倾向只保留 6 类。
2. 权重进入配置文件，所有默认值必须记录版本。
3. 参数候选采用“固定候选表 + 数据自适应补充”。
4. scoring mode 必须区分 `synthetic_paired` 与 `real_no_prior`。
5. AGC、强归一化、percentile stretch 默认属于 display-only，不能进入 full-reference 指标。
6. recipe 固定结构，AutoTune V1.0 只调参数和候选，不自动搜索任意 workflow。
7. 输出必须带 trial table、manifest、risk labels、claim boundary。

---

## 1. V1.0 范围与非范围

### 1.1 V1.0 范围

- 背景抑制候选比较。
- Dewow / bandpass / gain 的候选参数推荐。
- 轻量去噪候选，限 metric-safe 方法。
- 目标倾向驱动权重。
- synthetic paired full-reference scoring。
- real no-prior heuristic risk scoring。
- ROI-aware scoring。
- 每步处理完成后的 B-scan live preview。
- Evidence / report 记录。

### 1.2 V1.0 不做

- 不做全局 workflow 搜索。
- 不训练 CR-Net / YOLO / U-Net。
- 不把迁移作为默认 production scoring 步骤。
- 不在真实 no-prior 数据上宣称更接近真实地下结构。
- 不把 display-only 变化误认为算法处理结果。
- 不把少量 synthetic 结果写成真实泛化证明。

---

## 2. 目标倾向与权重

权重采用 0–1 工程重要性分值。运行时可以按总和归一化，也可以保留原始权重用于报告解释。权重值不是文献直接给出的自然常数，而是基于文献/软件实践和物理目标需求形成的 V1.0 default profile；后续必须用 synthetic paired batch 校准。

| Profile | 中文名 | Target preservation | Background suppression | Continuity | Contrast | False positive penalty | Ringing / artifact penalty | Depth / weak reflector |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `balanced` | 均衡处理 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 |
| `object_like_anomaly` | 局部异常增强 | 0.50 | 0.40 | 0.10 | 0.50 | 0.30 | 0.40 | 0.20 |
| `interface_layer_preservation` | 连续界面保留 | 0.40 | 0.20 | 0.55 | 0.25 | 0.40 | 0.30 | 0.50 |
| `landslide_bedrock_sliding_surface` | 滑坡基覆界面 / 潜在滑动面 | 0.35 | 0.20 | 0.65 | 0.25 | 0.50 | 0.30 | 0.65 |
| `wet_weak_zone` | 含水软弱带 | 0.40 | 0.25 | 0.45 | 0.35 | 0.40 | 0.35 | 0.70 |
| `deep_weak_reflector` | 深部弱反射增强 | 0.45 | 0.20 | 0.35 | 0.45 | 0.40 | 0.45 | 0.80 |


### 2.1 目标倾向解释

- **`balanced` / 均衡处理**：无明确解释目标时使用，所有指标保持中等权重，避免过度锐化或过度扣除。
- **`object_like_anomaly` / 局部异常增强**：适用于点状、双曲线或局部缺陷响应，重视目标对比和背景抑制，但必须惩罚 ringing 与假阳性。
- **`interface_layer_preservation` / 连续界面保留**：适用于层状/界面型目标，优先保留横向连续性和深部弱反射，限制强背景扣除和高阶 SVD。
- **`landslide_bedrock_sliding_surface` / 滑坡基覆界面 / 潜在滑动面**：面向滑坡基覆界面、潜在滑动面，属于 interface-like / zone-like 目标，连续性、深部保留和误报控制优先级最高。
- **`wet_weak_zone` / 含水软弱带**：面向含水软弱带、衰减相关异常，保留低频与弱反射，不使用强高通作为默认偏好。
- **`deep_weak_reflector` / 深部弱反射增强**：面向深部弱反射，重视深度补偿和弱信号保留，背景抑制保持保守。


### 2.2 权重进入实现时的约束

- 权重必须保存为配置文件，不写死在 UI 控件逻辑里。
- 每次 AutoTune 运行必须记录 `goal_profile_id`、`profile_version`、权重快照。
- 若用户手动改权重，必须记录 `profile_override=true`，并在 claim boundary 中声明人工干预。
- 对 `landslide_bedrock_sliding_surface`、`interface_layer_preservation`、`deep_weak_reflector`，默认降低强背景扣除、高阶 SVD、强高通和强平滑优先级。

---

## 3. 参数候选生成规范

V1.0 采用混合候选：

```text
fixed candidate table
+ data-adaptive candidate generator
+ profile-specific caps / penalties
+ coarse-to-fine refinement inside bounded range
```

### 3.1 背景抑制

| 方法 | V1.0 状态 | 参数 | 数据自适应 | 风险 |
|---|---|---|---|---|
| mean background | 默认候选 | none / global | 不需要 | 目标稀疏时可用；目标混入均值会扭曲异常 |
| median background | 默认候选 | none / global | 不需要 | 较稳健，但仍可能压制横向连续结构 |
| sliding mean | 默认候选 | `n_traces` | 由横向相关长度、trace spacing、目标横向尺度补充 | 窗太小估计不稳，太大跟踪不足 |
| sliding median | 默认候选 | `n_traces` | 同上 | 比 mean 稳健，但可能产生局部断裂 |
| SVD rank sweep | 默认候选 | `rank` | 固定 rank + elbow rank | 高 rank 会删除真实水平界面和深部弱反射 |
| RPCA / NMF | 后续高级候选 | lambda / rank | 需要单独 benchmark | 复杂度高，不进入 V1.0 默认 production |

默认固定候选：

```yaml
sliding_window_ntraces: [7, 11, 21, 31, 51, 81, 101]
svd_rank: [1, 2, 3, 5, 8]
```

Profile cap：

- `object_like_anomaly`：允许更强背景抑制，SVD rank 可到 elbow+1，但必须检查 ringing / false positive。
- `interface_layer_preservation`：默认 SVD rank cap = 1 或 conservative elbow lower bound。
- `landslide_bedrock_sliding_surface`：默认强背景扣除降权，优先 sliding weak background 或 rank 1。
- `wet_weak_zone` / `deep_weak_reflector`：背景抑制保守，避免压制衰减带和深部弱反射。

### 3.2 Dewow

- 固定候选：`[16, 32, 64, 128, 256]` samples。
- 自适应候选：若有中心频率和采样间隔，加入 1T / 2T 周期对应 samples。
- 风险：窗口过短相当于强高通，会损伤低频有效反射；对含水软弱带、深部弱反射、界面型目标应更保守。

### 3.3 Bandpass

- 必须依据 `dt / sampling_rate / Nyquist / antenna_center_frequency / dominant_frequency / spectrum energy percentile` 生成候选。
- 对 `wet_weak_zone`、`deep_weak_reflector`、`interface_layer_preservation`，强 high-pass 需要惩罚。
- 所有频带参数必须记录单位、来源和裁剪原因。

### 3.4 Gain

- V1.0 production scoring 只允许 SEC / exponential 等 metric-safe gain。
- AGC、强归一化、percentile stretch 默认 display-only。
- AGC 结果可以显示和导出预览，但不能进入 synthetic full-reference scoring。

### 3.5 Denoise

- V1.0 默认只放轻量候选：trace median light、Hampel spike removal、Savitzky-Golay light。
- wavelet / f-k filter 作为 experimental，不进入默认 recipe。
- 对界面型目标限制 smoothing window，避免抹掉破碎带、薄弱界面或局部中断。

### 3.6 Migration

- V1.0 不作为默认 AutoTune production scoring 步骤。
- 只在 object-like advanced recipe 中作为 experimental 可选。
- 必须有 velocity 来源或明确候选范围。
- UAV-GPR 必须先完成高度 / 姿态 / 地形相关补偿，再考虑 migration。

---

## 4. Scoring 设计

### 4.1 Synthetic paired scoring

输入必须来自同一 run/task：

```text
raw_<component>.npy
background_<component>.npy
target_response_<component>.npy = raw - background
```

禁止跨 run/task 配 raw/background。

核心指标：

- MAE / MSE / RMSE
- PSNR
- SSIM 或 SSIM-like structural metric
- correlation with target_response
- target ROI energy preservation
- background ROI energy suppression
- false positive energy outside target ROI
- ringing / artifact penalty

Synthetic score 形式：

```text
score_syn = weighted_sum(
  target_response_similarity,
  target_roi_preservation,
  background_roi_suppression,
  false_positive_penalty,
  continuity_or_contrast_by_profile,
  artifact_penalty
)
```

要求：

- 若 `target_response` 存在，full-reference 指标优先级高于 no-prior 启发式指标。
- 所有 candidate output 必须与 target_response shape 一致。
- ROI 生成参数必须记录。
- no_target / background-only 场景必须强化 false-positive risk。

Claim boundary：

> 本结果仅说明在当前 synthetic paired 数据、固定 workflow 和候选参数空间内，推荐参数在 full-reference / ROI-aware 指标上更接近 `target_response`。该结论不能直接推广到真实 no-prior 数据，也不能证明 AutoTune 全局最优。

### 4.2 Real no-prior scoring

禁止使用：

- MAE against unknown truth
- MSE / RMSE against unknown truth
- PSNR against unknown truth
- SSIM against unknown truth

允许使用：

- SCR / CNR proxy
- contrast
- entropy
- continuity / layer coherence
- texture stability
- background clutter proxy
- hot spot / ringing / new artifact penalty
- manual ROI energy change

Claim boundary：

> 本结果为无真值数据上的启发式参数推荐，只表示当前指标下的可视化改善、风险提示和人工复核依据，不能说明处理结果更接近真实地下结构。

---

## 5. ROI 策略

### 5.1 Synthetic labeled ROI

默认：

```text
A = abs(target_response)
target_seed = A >= P95(A)
target_roi = dilate(target_seed)
background_roi = A <= P60(A) and not target_roi
```

V1.0 增强：

- 对 interface-like 目标增加 band-like / layer-like ROI 选项。
- 若 `target_roi` 太小，输出 `roi_too_small` warning。
- 若 `background_roi` 太小，输出 `background_roi_too_small` warning。
- P95/P60 为默认启发式，不写成理论最优。

### 5.2 Real no-prior ROI

- 只能表示 target-likelihood / background-likelihood。
- 不能表示真目标。
- 默认 `manual_review_required=true`。

### 5.3 Manual ROI

必须记录：

- `roi_mode`
- `manual_roi_coordinates`
- `roi_used_for_scoring`
- `manual_review_required`
- `claim_boundary`

---

## 6. Workflow Recipe

| Recipe | 默认目标 | 步骤 | 风险 |
|---|---|---|---|
| `conservative_enhance` / 保守增强 | balanced, interface_layer_preservation, landslide_bedrock_sliding_surface | zero_time_correction → dewow_conservative → bandpass_conservative → weak_background_suppression_optional → metric_safe_gain → display_preview | 最稳但可能抑制不足；适合作为默认安全基线。 |
| `background_suppression_first` / 背景抑制优先 | object_like_anomaly | zero_time_correction → dewow → bandpass → background_candidates_mean_median_sliding_svd → artifact_check → metric_safe_gain | 可能删除真实水平界面或深部弱反射；不得用于层状目标默认流程。 |
| `interface_preservation` / 连续界面保留 | interface_layer_preservation, landslide_bedrock_sliding_surface | zero_time_correction → dewow_conservative → bandpass_low_cut_guarded → sliding_background_weak_or_svd_rank1_only → continuity_scoring → depth_compensation_safe | 可视上不一定最锐利；以保守处理和人工复核为边界。 |
| `deep_weak_reflector` / 深部弱反射增强 | deep_weak_reflector, wet_weak_zone, landslide_bedrock_sliding_surface | zero_time_correction → dewow_conservative → bandpass_low_cut_guarded → weak_background_suppression → sec_or_exponential_gain → artifact_check | 增益可能放大噪声；AGC 只能作为 display-only。 |
| `wet_weak_zone` / 含水软弱带倾向 | wet_weak_zone | zero_time_correction → dewow_conservative → bandpass_keep_low_frequency → weak_background_suppression → attenuation_continuity_scoring → risk_label | 不能把衰减带直接解释为含水层；必须人工复核。 |
| `object_focus_experimental` / 局部异常聚焦（高级） | object_like_anomaly | zero_time_correction → dewow → bandpass → background_suppression → migration_optional_experimental → hilbert_display_optional | 迁移依赖速度估计；V1.0 默认不纳入 production scoring，仅作为高级可选。 |


Recipe 原则：

- V1.0 固定 recipe，不自动任意改流程结构。
- AutoTune 在 recipe 内调参数和候选，不做全局 workflow search。
- 每个 recipe 必须带适用场景、风险提示和不能主张内容。
- 普通 UI 展示中文 recipe 名和工程解释；高级面板展示算法序列、候选空间和 scoring details。

---

## 7. UI 设计原则

AutoTune 页面最终应按工程用户理解组织：

```text
选择目标倾向
→ 设置重点区域 / ROI
→ 生成推荐
→ 查看推荐理由与风险
→ 应用推荐
→ 每步处理完成后实时刷新 B-scan
→ 导出处理记录 / 报告
```

普通界面避免堆叠以下术语：

- Evidence
- claim boundary
- manifest
- trial table
- synthetic supervised
- no-prior

替换为：

- 处理记录
- 报告导出
- 质量检查
- 风险提示
- 结论说明
- 参数对比明细
- 高级明细

---

## 8. Evidence / Report 字段

V1.0 每次 AutoTune 输出至少记录：

- `source_commit`
- `software_version`
- `input_identity`
- `input_shape`
- `component`
- `workflow_recipe`
- `goal_profile`
- `candidate_space_hash`
- `algorithm_sequence`
- `parameters`
- `scoring_mode`
- `metrics`
- `roi_mode`
- `roi_definition`
- `warnings`
- `risk_labels`
- `display_settings`
- `preview_paths`
- `trial_table_path`
- `claim_boundary`
- `manual_review_required`


---

## 9. 风险标签

基础 warning tags：

```yaml
- weak_target_response
- roi_too_small
- background_roi_too_small
- shape_mismatch
- nan_or_inf_detected
- autotune_failed
- display_only_transform_used
- agc_excluded_from_scoring
- svd_rank_may_remove_interface
- highpass_may_remove_deep_or_wet_response
- no_prior_manual_review_required
```

---

## 10. 实施阶段

### Phase 1：配置化定稿

- 新增 `configs/autotune_v1_profiles.yaml`
- 新增 `configs/autotune_v1_recipes.yaml`
- 新增 schema 校验
- 不改核心算法输出

### Phase 2：candidate generator

- 读取 metadata / data features
- 生成 fixed + adaptive candidates
- 输出 candidate space hash

### Phase 3：synthetic paired scoring

- 加入 MAE / MSE / RMSE / PSNR / SSIM-like
- 加入 target/background ROI metrics
- 加入 no_target false-positive scoring

### Phase 4：real no-prior risk scoring

- 启发式指标
- risk label
- manual review required

### Phase 5：UI 接入

- 目标倾向选择
- 推荐理由
- 实时 B-scan live preview
- 高级明细折叠

### Phase 6：batch calibration

- 用 gprMax synthetic paired batch 校准权重和阈值
- 审计 profile 是否稳定
- 再冻结 V1.0 production defaults

---

## 11. 论文与软著表达边界

可以主张：

- MyGPR 支持固定处理流程内的参数候选比较。
- MyGPR 支持目标倾向驱动的 AutoTune 参数推荐。
- MyGPR 支持 synthetic paired 数据上的 full-reference / ROI-aware scoring。
- MyGPR 支持真实 no-prior 数据上的启发式风险提示和人工复核依据。
- MyGPR 可导出 trial table、manifest、preview 和 claim boundary。

不能主张：

- AutoTune 是全局最优。
- AutoTune 替代专家。
- 真实 no-prior 结果更接近真实地下结构。
- 当前实现等价于 CR-Net / 深度学习 clutter removal。
- 少量 synthetic batch 证明真实泛化。

---

## 12. 验收门槛

进入 V1.0 production 前必须满足：

1. 所有 profile / recipe 可配置且可记录。
2. synthetic paired scoring 与 no-prior scoring 分离。
3. display-only transformation 不进入 full-reference scoring。
4. trial table 字段完整。
5. manifest 记录 input identity / workflow / 参数 / metrics / claim boundary。
6. 至少一批 synthetic paired 数据通过 batch audit。
7. 风险标签统计正常，无大量 `roi_too_small` 或 `weak_target_response` 未解释。
8. UI 能清楚展示推荐原因、风险、应用按钮和每步实时 B-scan。

---

## 13. 推荐开发任务名

- `AT-V1-CONFIG-001`: AutoTune profile / recipe 配置化
- `AT-V1-CANDIDATE-002`: 数据自适应候选生成器
- `AT-V1-SCORE-003`: Synthetic paired full-reference scoring
- `AT-V1-NOPRIOR-004`: Real no-prior risk scoring
- `AT-V1-REPORT-005`: Trial table / manifest / claim boundary export
- `AT-V1-UI-006`: 工程化 AutoTune UI 接入
- `AT-V1-CAL-007`: Synthetic paired batch calibration
