# MyGPR 三页重构 — 交接文档

> 本文档是给接手改造的 AI/开发者看的快速入口。详细设计规范见 `docs/superpowers/specs/2026-07-03-three-page-redesign-design.md`

## 项目概况

MyGPR 是面向 GPR / UAV-GPR 勘探项目的桌面软件。
- Python 3.11+，PyQt6 + qfluentwidgets
- 5 个主页面：项目管理 → 测线处理 → 目标定位 → 空间成果 → 成果报告
- 本次改造范围：**项目管理、测线处理、空间成果 三页**（排除目标定位页）

## 改造已完成的工作

最后一次 commit `60cc89a`（"refactor: 三页重构型审美升级 — Part 1 视觉总原则落地"）

6 个文件已修改：

| 文件 | 改了什么 |
|-------------------------|---------|
| `ui/field_panels/widgets.py` | Card 内边距、MetricCard 高度、PlotCard 高度弹性化 |
| `ui/field_panels/layout_metrics.py` | 三页布局比例调整（主区增大、辅区缩小） |
| `ui/field_panels/processing_page.py` | 按钮层级重排（主次分明）、操作区固定高度移除 |
| `ui/field_panels/spatial_page.py` | 地图权重提升、工具栏精简 |
| `ui/field_panels/project_page.py` | 操作面板紧凑化、工具栏精简 |
| `ui/field_workbench_window.py` | QSS 全面调优（圆角/字号/边距/行距） |

**已推送**到 GitHub `v1.0-refactor` 分支。

## 待完成工作

### 1. Part 4 — 界面评审框架（未写）
按原计划，在 Part 3 之后应当交付一个实用评审清单，包括：
- 判断"空白过多"的标准（某区域空置率 > 30% 等）
- 判断"挤但不密"的检查项
- 判断"主次关系失控"的参照标准
- 判断"按钮过载"的规则
- 判断"图表被压扁"的阈值
- 三页通用评审清单 + 单页专项清单

### 2. 运行验证改后效果
这次改完后**未实际运行过**，需要：
- 在 Windows 1080P/125% 环境运行验证
- 用 `layout_diagnostics_rules.py` 跑一遍布局诊断
- 目视检查三页有无新的空白/重叠问题
- 确认按钮层级变更后功能正常

### 3. 布局诊断规则同步更新
`ui/field_panels/layout_diagnostics_rules.py`
需要同步更新的阈值：
- `processing_bscan_min_height` 规则中的 360→380（因 B-scan 高度比例上调）
- `processing_params_width_max` 规则中的 300→310（因参数栏宽度比例调整）
- `bottom_region_height_max` 规则中的 22%→20%（因底部区比例下调）
- `processing_continuous_card_min_height` 规则中的 150→140（因操作区重构）
- 空间侧栏宽度规则可能需要放宽

### 4. 视觉舒适度规则同步更新
`ui/field_panels/visual_comfort_rules.py`
- `spatial_aux_not_heavy` 规则中的阈值可能需要调整（侧栏已收窄）
- 新增 B-scan 舒适高度检查（当 B-scan 高度 < 380 时报警）

### 5. 新增 widget 对象名对齐
`layout_diagnostics_rules.py` 中定义了 `CANVAS_CARD_PAIRS`，如果新增了 PlotCard 或修改了 object name，需要确保这个字典同步更新。

## 关键文件速查

| 文件 | 行数 | 用途 |
|------|------|------|
| `ui/field_workbench_window.py` | ~1380 | 主窗口 + QSS 全局样式 |
| `ui/field_panels/widgets.py` | ~290 | Card / MetricCard / PlotCard / CollapsibleSidePanel |
| `ui/field_panels/layout_metrics.py` | ~170 | 所有页面的自适应尺寸 |
| `ui/field_panels/processing_page.py` | ~860 | 测线处理页 |
| `ui/field_panels/spatial_page.py` | ~470 | 空间成果页 |
| `ui/field_panels/project_page.py` | ~990 | 项目管理页 |
| `ui/field_panels/layout_diagnostics_rules.py` | ~310 | 布局诊断规则 |
| `ui/field_panels/visual_comfort_rules.py` | ~? | 视觉舒适度规则 |
| `ui/field_panels/field_ui_styles.py` | ~36 | 颜色常量 |

## 设计方向摘要

**风格**：80% 工程仪表台 + 20% 轻量现代
**色调**：冷蓝灰系（不是暖白/奶白/纯蓝）
**核心口号**："主区顶满，辅区收紧，操作降噪，信息归位。"
**三页各自主角**：
- 项目管理 → 项目状态与任务入口
- 测线处理 → B-scan / 处理结果
- 空间成果 → 平面图 / 空间关系

## Git 信息

```
分支: v1.0-refactor
最新 commit: 60cc89a
远程: https://github.com/CYberkra/MyGPR.git
```
