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

当前实现先覆盖后端运行、评分、JSON-safe 摘要，以及 GUI 内的人工/自动对比按钮、结果摘要和 B-scan 快照对比；PNG/PDF/CSV 证据导出留给下一阶段接入。

## 和现有代码的连接点

- `core.auto_tune.auto_tune_method`：继续作为单方法选参入口。
- `core.auto_tune.auto_select_method_group`：可作为 stage 比较入口。
- `core.preset_profiles.RECOMMENDED_RUN_PROFILES`：补充 `uav_gpr_standard` 与 `uav_gpr_experience_baseline_v1`。
- `ui.gui_auto_tune_page.AutoTunePage`：新增“人工/自动对比”动作和结果面板。
- `app_qt.py`：提供当前日常处理页参数快照，作为 manual baseline。
- `core.evidence_export` 或新 `core.comparison_export`：统一导出 JSON、PNG、CSV。

## 实现顺序

1. 新增 `core/auto_tune_comparison.py`，先实现单方法和固定 pipeline 对比，不改 UI。
2. 增加 tests，验证人工参数来源、自动参数来源、同 ROI/同显示尺度、导出 JSON schema。
3. 在 CLI 或脚本中加一个最小 smoke，用 sample_data 生成对比报告。
4. 在 `AutoTunePage` 增加按钮和结果表。（已完成）
5. 接入左右 B-scan/滑块对比。（已通过现有 compare snapshots 完成）
6. 再扩展完整标准流程逐 stage 自动选参。

## 评价指标原则

科研对比不能只靠视觉截图。推荐使用“指标 + 图像”双证据：

- 自动结果必须有更高综合评分。
- 自动结果不能触发更严重的 data_sanitized、clipping、over_smoothing、metadata_missing warning。
- 若目标/ROI 明显存在，自动结果应提升目标显著性或聚焦；若无明显目标，自动结果至少应提升背景稳定和深部可见性。
- 若自动结果指标更好但图像明显失真，应在报告里给出“自动不推荐”或“低置信度”。

## 组会表述建议

“我们不是简单把图像调得更亮，而是在同一输入、同一 ROI、同一显示尺度下，对人工经验参数和自动候选搜索进行量化比较。自动选参的优势来自可复现搜索、指标约束和过处理惩罚，因此能减少人工调参的主观性。”
