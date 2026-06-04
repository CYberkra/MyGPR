# GX-UI-024~030 Consolidated UI Final Pass

本轮合并完成以下方向的 UI 收口：

- GX-UI-024 STYLE-CONSOLIDATION
- GX-UI-025 BSCAN-WORKSPACE-DENSITY-POLISH
- GX-UI-026 DAILY-PROCESSING-PAGE-POLISH
- GX-UI-027 AUTOTUNE-RECOMMENDATION-UX-POLISH
- GX-UI-028 LEGACY-UI-REFERENCE-AUDIT
- GX-UI-029 UI-TERMINOLOGY-PASS
- GX-UI-030 EVIDENCE-CHECKLIST-PAGE（视觉入口级收口）

## 主要改动

### 样式体系收口

- 将 legacy AutoTune、quality log、research console 中的部分页面内 `setStyleSheet(...)` 改为 objectName / dynamic property。
- 在 `ui/theme.py` 中增加统一样式：
  - `TruthPreviewBox`
  - `TruthStatusLabel`
  - `QualityToolButton`
  - `ResearchNavCard`
  - `MetricCard`
  - `InfoBanner`
  - `WarningBanner`
  - `RuntimeSummaryChip`
  - `NextStepHint`

### 主 B-scan 工作区

- 减少 drawer 高度，降低运行日志对主图的挤压。
- 增加 `RuntimeSummaryChip`，折叠状态下保留一句运行状态摘要。
- `_apply_main_workspace_direct_theme` 改为属性刷新，不再直接塞局部 QSS，降低深浅主题冲突风险。

### AutoTune 推荐 UX

- 顶部增加“下一步”提示条，随数据、ROI、候选与推荐状态变化。
- 候选排名第一项显示为“推荐”。
- Trial Table 第一候选统一显示“推荐候选”。
- 保持 UI-local 推荐，不调用生产 AutoTune。

### 术语统一

- `shape` 统一为 `尺寸`。
- `Raw` 视觉层面统一为 `原始数据`。
- `baseline` 视觉层面统一为 `基线`。
- `rank sweep` 视觉层面统一为 `秩扫描`。

## 旧页面审计结论

- `ui/gui_workbench.py` 仍作为 retired shim 保留。
- `ui/gui_auto_tune_page.py` 仍被 `AutoTuneTuningPage` 作为兼容层引用，因此本轮不删除。
- `ui/gui_method_browser.py` / `ui/gui_param_editor.py` 仍属于历史工作台组件，当前主 UI 不主动实例化。
- `ui/research_console_page.py` 为只读研究控制台，当前不从主 UI 主路径进入，但保留作为文档/实验入口候选。

## 未改动

- 不修改处理算法。
- 不修改 AutoTune 生产评分逻辑。
- 不运行 gprMax。
- 不修改 GX-008 / GX-009 模型。
- 不修改 MyGPR-Evidence。
- 不引入 PyVista / PyVistaQt。
