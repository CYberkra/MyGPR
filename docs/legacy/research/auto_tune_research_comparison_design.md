#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-tune research comparison design for MyGPR."""

# 自动选参科研对比页设计（人工 baseline vs 自动选参）

目标：在 MyGPR 中提供一个可复现实验入口，用同一份 UAV-GPR 数据、同一 ROI、同一显示尺度，对比“人工经验参数 baseline”和“自动选参参数”，输出 B-scan、参数表、评分和可导出的科研证据。

## 产品定位

当前“调参与实验”页已经能做单方法自动选参和同阶段比较，但它还不够直接回答组会问题：“自动选参为什么比人工更合适？”

新增能力应叫“人工/自动对比”或“科研对比”，放在现有 `AutoTunePage` 的结果查看/深度实验入口附近，不另造一套孤立逻辑。

## 对比对象

### 人工 baseline

人工 baseline 有两种来源，按优先级取值：

1. 日常处理页当前用户已调整的参数。
2. 若用户未调整，则使用项目预设“经验参数 baseline”。

这意味着日常处理页参数变化后，科研对比页里的“人工参数”应同步更新。同步对象不是 UI 文本，而是当前 method/pipeline 的参数字典。

建议定义：

- `manual_baseline_source`: `current_ui_params` 或 `experience_profile`
- `manual_baseline_params`: 每个方法的实际参数
- `experience_profile_key`: 例如 `uav_gpr_experience_baseline_v1`

### 自动选参方案

自动方案也保留两层：

1. 单方法自动选参：用于解释某个算法为什么选这个参数。
2. 标准流程自动选参：用于科研展示最终 B-scan 改善，默认推荐此模式。

标准流程的自动选参应按 stage 逐步运行，至少覆盖：

- `set_zero_time`
- `dewow`
- `motion_compensation_v2`（实现前可用 V1 motion stage）
- `background`
- `denoise`
- `gain/display`

## 页面结构

### 1. 实验设置

- 数据集：当前打开数据。
- ROI：当前裁剪区优先、自动 ROI、全图。
- 对比模式：单方法、当前阶段、完整标准流程。
- 人工 baseline 来源：跟随日常处理页、经验参数 baseline。
- 自动搜索强度：快速、标准、深入。
- 显示尺度锁定：必须默认开启，避免自动结果仅因 contrast 更强看起来更好。

### 2. 结果对比

主区域使用左右 B-scan：

- 左：人工 baseline 处理结果。
- 右：自动选参处理结果。
- 支持同步缩放、同 ROI 框、同色标、同 percentile/clip 设置。
- 支持滑块对比，复用现有 advanced compare/slider compare 能力。

### 3. 参数与评分

参数表列：

- stage
- method
- manual params
- auto params
- score delta
- 自动选择理由
- warnings

指标表列：

- 背景/水平条纹抑制
- first-break 稳定性
- 目标/局部显著性保持
- 深部能量可见性
- 过度平滑惩罚
- 饱和/裁剪比例
- 综合评分

页面文字必须避免绝对声称“自动总是更好”。更严谨的表述是：“在当前数据、当前 ROI 和这些指标下，自动选参优于人工 baseline。”

### 4. 导出

导出内容：

- `comparison_summary.json`
- `manual_bscan.png`
- `auto_bscan.png`
- `side_by_side.png`
- `params_table.csv`
- `metrics_table.csv`
- 可选 PDF/Markdown 报告

导出文件应记录：

- MyGPR 版本/commit（若可得）
- 输入路径和 hash
- header_info
- trace_metadata 质量摘要
- ROI
- manual baseline 来源
- 自动搜索模式与候选数量
- 每步参数、warnings、评分

## 后端数据契约

已新增核心服务 `core/auto_tune_comparison.py`，不要把科研对比逻辑写死在 GUI。GUI、CLI、导出报告都应复用这个后端。

```python
@dataclass
class ComparisonCandidate:
    name: str
    source: str
    pipeline: list[str]
    params_by_method: dict[str, dict[str, Any]]
    result: np.ndarray
    metadata: dict[str, Any]
    metrics: dict[str, float]
    warnings: list[str]


@dataclass
class AutoTuneComparisonRun:
    input_id: str
    roi_spec: dict[str, Any]
    display_spec: dict[str, Any]
    manual: ComparisonCandidate
    automatic: ComparisonCandidate
    metric_delta: dict[str, float]
    verdict: str
```

核心入口建议：

```python
run_auto_tune_comparison(
    data,
    header_info,
    trace_metadata,
    pipeline,
    manual_params_by_method,
    baseline_profile_key,
    roi_spec,
    search_mode,
)
```

当前已实现：

- `ComparisonCandidate`
- `AutoTuneComparisonRun`
- `run_auto_tune_comparison(...)`
- `to_summary_dict(...)`
- `uav_gpr_experience_baseline_v1` 经验参数 baseline profile

当前实现先覆盖后端运行、评分、JSON-safe 摘要，以及 GUI 内的人工/自动对比按钮、结果摘要和 B-scan 快照对比。

证据导出已下沉到 `core/auto_tune_comparison_export.py`，GUI 不再自己拼 PNG/CSV。统一导出内容包括：

- `comparison_summary.json`
- `manual_bscan.png`
- `auto_bscan.png`
- `side_by_side.png`
- `params_table.csv`
- `metrics_table.csv`
- `comparison_report.md`

所有图像默认锁定同一色标范围，避免自动结果仅因为显示对比度更强而显得更好。

## 流程级自动选参后端

2026-05-08 已新增 `core/auto_tune_pipeline.py`。它是后续科研论证和论文/专利证据链的首选后端，职责不是替代单方法自动选参，而是把“整条处理流程”按当前状态逐步运行并评分。

核心入口：

```python
run_auto_tune_pipeline(
    data,
    header_info=None,
    trace_metadata=None,
    pipeline=None,
    manual_params_by_method=None,
    baseline_profile_key=None,
    roi_spec=None,
    ground_truth=None,
    search_mode="standard",
    rollback_on_reject=True,
)
```

该后端与 `core.auto_tune_comparison.run_auto_tune_comparison(...)` 的分工：

- `run_auto_tune_comparison(...)`：适合简单人工/自动最终结果对比，输出最终 B-scan、参数表和指标。
- `run_auto_tune_pipeline(...)`：适合完整流程论证，保存每个算法步骤的 `manual_before`、`manual_after`、`auto_before`、`auto_after`，并记录每一步的参数、指标、risk flags、推荐结论和是否回退。

当前流程级契约：

- 每一步自动选参都基于自动分支“当前已处理状态”，而不是原始数据或孤立单步数据。
- 每一步同时运行人工 baseline 和自动候选，并计算 `pipeline_score` 与 `metric_delta`。
- 若提供 gprMax `ground_truth`，会合并 `truth_target_energy_preservation`、`truth_background_energy_reduction`、`truth_false_positive_ratio`、`truth_score` 等真值指标。
- 若自动候选损伤真值目标、制造无目标假异常、综合分低于人工、置信度过低、多个候选近似最优、参数被强制约束或疑似过曝，会写入风险标记。
- 当 `rollback_on_reject=True` 且某一步结论为 `keep_manual`，自动流程后续状态会回退到该步人工结果，但 step record 仍保留被拒绝的自动候选图像和参数，便于报告解释。
- `to_summary_dict(...)` 输出 JSON-safe 摘要，不携带原始 B-scan 数组；HTML/报告若需要逐步图像，应直接读取 `AutoTunePipelineRun.steps` 中的数组。

后续 GUI/报告接入建议：

1. 科研 HTML 报告应改为优先消费 `AutoTunePipelineRun.steps`，这样每一步都能展示运行前、人工后、自动后的 B-scan。
2. GUI 的“自动选参比人工更好”结论应显示为 `overall_recommendation`，并同时列出 `risk_flags`，避免绝对化表达。
3. gprMax 多场景报告应把 `ground_truth` 传入新后端，使正演真值直接参与流程级评分和回退。
4. 旧比较后端继续保留，作为轻量导出和兼容入口。

## 和现有代码的连接点

- `core.auto_tune.auto_tune_method`：继续作为单方法选参入口。
- `core.auto_tune.auto_select_method_group`：可作为 stage 比较入口。
- `core.preset_profiles.RECOMMENDED_RUN_PROFILES`：补充 `uav_gpr_standard` 与 `uav_gpr_experience_baseline_v1`。
- `ui.gui_auto_tune_page.AutoTunePage`：新增“人工/自动对比”动作和结果面板。
- `app_qt.py`：提供当前日常处理页参数快照，作为 manual baseline。
- `core.auto_tune_comparison_export`：统一导出 JSON、PNG、CSV、Markdown 报告。
- `core.auto_tune_pipeline`：完整流程逐步选参、truth-aware 评分、风险提示和回退建议。

## 实现顺序

1. 新增 `core/auto_tune_comparison.py`，先实现单方法和固定 pipeline 对比，不改 UI。
2. 增加 tests，验证人工参数来源、自动参数来源、同 ROI/同显示尺度、导出 JSON schema。
3. 在 CLI 或脚本中加一个最小 smoke，用 sample_data 生成对比报告。
4. 在 `AutoTunePage` 增加按钮和结果表。（已完成）
5. 接入左右 B-scan/滑块对比。（已通过现有 compare snapshots 完成）
6. 导出完整科研证据包。（已完成）
7. 用 GPRMAX 正演构建带 ground truth 的 benchmark 数据集，验证自动选参是否真正保留目标双曲线、裂缝、层状界面等结构。
8. 再扩展完整标准流程逐 stage 自动选参。

## GPRMAX 正演验证决策

组会已决策：真实外业数据很难可靠判断哪些双曲线、裂缝或结构必须被保留，因此需要使用 GPRMAX 正演数据来验证和改进自动选参。GPRMAX 数据不是替代真实外业数据，而是提供可控 ground truth：

- 场景中有哪些目标、边界、双曲线 apex、层状界面是已知的。
- 自动选参是否保留这些目标，而不是仅让图像更干净。
- 哪些候选参数会过度平滑、削弱目标能量或误删弱反射。
- 不同噪声、含水率、埋深、介电常数和飞行高度扰动下，评分函数是否稳定。

优先级：GPRMAX benchmark 是当前导出闭环完成后的下一条主线，早于继续细调自动选参权重。原因是没有 ground truth 时，继续调权重容易只优化视觉观感。

本机已有 `E:\gprMax\gprMax-v.3.1.7`，其中包含 `user_models/cylinder_Bscan_2D.in`、`crack_model_generator.py`、`landslide_model_generator.py`、`gprmax_test/landslide_model.in` 等可参考材料。但该目录混有原版 gprMax、venv、GUI 试验、硬编码输出路径和缺少 MyGPR 可读 ground-truth manifest 的旧实验文件。结论是：参考旧模型思路，不把旧目录直接作为 MyGPR benchmark 依赖；应在 MyGPR 中重新定义干净的 scenario schema、输出目录、ground-truth manifest 和导入测试。

## 评价指标原则

科研对比不能只靠视觉截图。推荐使用“指标 + 图像”双证据：

- 自动结果必须有更高综合评分。
- 自动结果不能触发更严重的 data_sanitized、clipping、over_smoothing、metadata_missing warning。
- 若目标/ROI 明显存在，自动结果应提升目标显著性或聚焦；若无明显目标，自动结果至少应提升背景稳定和深部可见性。
- 若自动结果指标更好但图像明显失真，应在报告里给出“自动不推荐”或“低置信度”。

## 组会表述建议

“我们不是简单把图像调得更亮，而是在同一输入、同一 ROI、同一显示尺度下，对人工经验参数和自动候选搜索进行量化比较。自动选参的优势来自可复现搜索、指标约束和过处理惩罚，因此能减少人工调参的主观性。”
