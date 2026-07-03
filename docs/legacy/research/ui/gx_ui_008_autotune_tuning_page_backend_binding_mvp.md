# GX-UI-008 AutoTune 调参页状态绑定 MVP

## 任务目标

本次修改在 `GX-UI-007` 新增的 `AutoTuneTuningPage` 基础上，加入轻量级 UI 状态绑定，使页面不再只是静态占位。该任务只绑定页面控件与本地状态，不启用 AutoTune 后端执行，不修改生产评分逻辑。

## 修改范围

新增/修改：

- `ui/autotune_tuning_page.py`
- `docs/ui/gx_ui_008_autotune_tuning_page_backend_binding_mvp.md`
- `scripts/quick_start_mygpr.bat`

未修改：

- `ui/gui_auto_tune_page.py`
- `ui/research_console_page.py`
- GX-008 / GX-009 模型文件
- gprMax campaign
- Evidence 仓库或 Evidence 产物
- AutoTune 生产评分逻辑

## 页面状态模型

新增 UI 本地状态模型：

`AutoTuneTuningState`

字段包括：

- `workflow_step`
- `candidate_methods`
- `svd_rank_min`
- `svd_rank_max`
- `roi_trace_start`
- `roi_trace_end`
- `roi_sample_start`
- `roi_sample_end`
- `scoring_metrics`
- `no_prior_warning_enabled`
- `display_only_warning_enabled`
- `manual_review_required`
- `claim_boundary_required`
- `data_label`
- `evidence_ready`

该状态目前只服务 UI 预览，不写入生产配置，不触发后端任务。

## 已绑定控件

### Workflow Step

支持选择：

- Background Suppression
- Gain
- Dewow
- Bandpass
- Display Enhancement

切换后会更新：

- 顶部 session chip
- 右侧推荐预览
- Trial Table
- Manifest 占位文本

### Candidate Space

支持勾选：

- no suppression
- mean background
- median background
- SVD
- sliding window

支持配置：

- SVD rank min
- SVD rank max

变化后会更新：

- 候选数量
- Trial Table 行
- 推荐参数占位
- 风险提示

### ROI

支持配置：

- trace start
- trace end
- sample start
- sample end

变化后会更新：

- ROI 状态：未设置 / 已设置
- ROI Overlay Preview 占位文本
- Metrics 占位文本
- Evidence readiness 状态

### Scoring

支持勾选：

- RMSE
- ROI energy retention
- outside ROI residual
- CNR/SNR
- apex stability

变化后会更新：

- scoring metric count
- Metrics 面板
- Candidate Score 预览
- Warnings

### Safety

支持勾选：

- no-prior warning
- display-only flag
- manual review required
- claim boundary required

变化后会更新：

- Risk Warnings
- Claim Boundary
- Evidence readiness

## 仍为占位的部分

以下功能仍未启用：

- 载入真实数据
- 运行 AutoTune
- 导出 Evidence
- 真实 B-scan / ROI overlay
- 真实 metrics 计算
- 真实 trial table 结果
- 生产 scoring 修改

顶部按钮当前只给出 MVP 提示，不执行后端任务。

## Legacy 兼容

`AutoTuneTuningPage` 仍保留隐藏的 legacy `AutoTunePage` 实例，用于兼容 `app_qt.py` 中已有的信号和状态调用。

保留 legacy 页面：

- `ui/gui_auto_tune_page.py`
- `ui/research_console_page.py`

本任务不删除旧页面。

## 依赖边界

本任务未引入：

- PyVista
- PyVistaQt
- QtInteractor
- gprMax runtime dependency
- Evidence dependency

MyGPR 启动路径不应因为本任务引入 PyVista/PyVistaQt。

## Claim Boundary

本任务是 UI 状态绑定，不是算法更新。

不能主张：

- AutoTune 生产逻辑已改进
- AutoTune 已可自动运行
- AutoTune 优于人工
- 已完成 field validation
- 已完成 Evidence export 自动化

可以主张：

- 新 AutoTune 调参页已具备 ROI / Candidate / Scoring / Safety 的本地交互状态
- Trial Table / Metrics / Inspector / Manifest / Claim Boundary 占位区可随控件变化同步更新
- 生产 AutoTune 逻辑保持不变
