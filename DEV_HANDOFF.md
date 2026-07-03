# MyGPR 开发交接

当前版本：v0.9.24 beta

## v0.9.24 当前完成项

本版取消 v0.9.21 那种“模板式处理链”方向，改为更轻量的连续手动处理。处理页不再要求用户维护模板；而是围绕当前测线，在原始数据基础上逐步叠加处理步骤。

新增：

- `core/manual_processing_chain.py`：维护单条测线的临时处理链、撤回一步、重置到原始和保存载荷构造。
- 测线处理页右侧操作重组为：执行当前步骤、撤回一步、重置到原始、前后对比、保存当前结果、参数推荐。
- 底部消息区新增“处理历史”页签，直接展示当前已叠加的步骤。
- 保存处理结果时，manifest 会附带 `processing_mode=manual_step_chain`、`chain_step_count` 和 `chain_steps`。

本版新增模块联动闭环：`core/project_events.py`、`core/project_dependency_rules.py`、`core/project_state_tracker.py` 与 `ui/field_linkage_controller.py`。后续页面操作应优先发出 ProjectEvent，不要继续让页面之间互相直接刷新。

主流程为：

```text
项目管理 -> 测线处理 -> 目标定位 -> 空间成果 -> 成果报告
```

已完成：

- 正式项目创建、打开、最近项目、项目设置。
- 单条 CSV / TXT / NPY / NPZ / H5 / HDF5 导入和后台批量导入。
- 导入预检、坏文件诊断、数据质检和 B-scan 方向修正。
- 测线清单 CSV 导出和项目 ZIP 备份。
- 空间成果工具栏真实回调：刷新、坐标导出、平面图、三维视图、图层控制。
- 三维成果窗口：三维轨迹、目标点、平面图、数据汇总、PNG 和点云 CSV 导出。
- 成果报告包：HTML、JSON、CSV 和 PDF。
- 15.6 寸 1080P compact mode：读取 `availableGeometry()`，按 Windows 任务栏可用区域选择初始窗口尺寸。
- Windows 真机截图诊断脚本：`scripts/capture_field_workbench_windows_diagnostics.py`。
- 已回收一次 Windows 真机 `windows_fit_check` 截图并修复 PlotCard canvas 垂直居中导致的预览图下沉问题。
- 已读取用户回传的 `windows_fit_check_after_patch`，确认核心布局已适配 `1536×816` 可用区域，并对表格/参数面板文字对比度做最终收口。
- v0.9.24 新增 `ui/field_panels/layout_metrics.py`，四个核心页面的主图高度、侧栏宽度、底部区高度应继续从该模块取值，不要再在页面里散落固定尺寸。
- v0.9.24 新增右侧栏折叠与主图放大查看，通用组件位于 `ui/field_panels/widgets.py`。

## 关键文件

- `app_qt.py`：正常入口，创建 `FieldWorkbenchWindow`。
- `ui/field_workbench_window.py`：现场工作台 shell、屏幕适配、页面装配。
- `ui/field_panels/project_page.py`：项目管理与导入操作。
- `ui/field_panels/processing_page.py`：测线处理页面。
- `ui/field_panels/interpretation_page.py`：目标定位页面。
- `ui/field_panels/spatial_page.py`：空间成果工具栏和空间页面。
- `ui/field_panels/spatial_3d_dialog.py`：三维成果窗口。
- `ui/field_panels/delivery_page.py`：报告页和 PDF 入口。
- `ui/field_panels/layout_metrics.py`：统一自适应布局参数，负责主图、侧栏、底部区尺寸计算。
- `ui/field_panels/widgets.py`：包含 `CollapsibleSidePanel`、`PlotCard`、`PlotViewerDialog` 等通用 UI 组件。
- `core/field_report_export.py`：报告包 / PDF 生成。
- `core/field_project_status.py`：项目真实状态聚合。
- `scripts/capture_field_workbench.py`：通用截图脚本。
- `scripts/capture_field_workbench_windows_diagnostics.py`：Windows 可用屏幕 / DPI / 任务栏诊断截图脚本。

## 快速回归命令

```bash
python -m compileall . -q
python scripts/check_version_consistency.py --expected 0.9.24
python -m pytest tests/test_workbench_1080p_fit.py tests/test_capture_summary.py tests/test_field_project_store.py tests/test_field_project_operations.py tests/test_report_export_v098.py tests/test_version_consistency.py tests/test_field_workbench_boundaries.py tests/test_field_processing_bridge.py tests/test_project_status_metrics.py -q
python scripts/capture_field_workbench.py --output /mnt/data/mygpr_v0.9.24_screenshots --width 1450 --height 790
```

Windows 真机适配验证：

```bat
python scripts\capture_field_workbench_windows_diagnostics.py --output windows_fit_check
```

## 开发边界

- 不把项目统计逻辑写进 UI 回调，继续放在 `core/field_project_status.py`。
- 不把新建 / 打开 / 导入校验 / 项目设置逻辑写进大型 UI 回调，继续放在 `core/field_project_operations.py`、`core/field_import_preview.py` 或 store 层。
- 不把启发式目标检测宣传为深度学习模型。
- 不把只识别扩展名的厂商格式宣传为完整原生解码。
- 新增 UI 必须优先保证 15.6 寸 1080P、Windows 125% 缩放、任务栏可见时关键按钮可操作。
- 新增页面或调整布局时，优先扩展 `layout_metrics.py`，避免重新回到分散的 `setFixedWidth` / `PlotCard(height=...)` 魔法数字。

## 下一轮建议

1. 让用户用最终包在 Windows 真机做一次完整 beta 操作压测：新建/打开项目、导入、处理、目标、空间、PDF 报告和备份。
2. 做 Excel 报告闭环。
3. 做目标标注编辑 / 删除 / 复核状态流转。
4. 做厂商格式专项，每种格式必须有样例文件和读取测试。

## v0.9.24 交接补充

- 关键 PlotCard 已设置最小高度，避免 canvas 高于所属卡片。
- 布局诊断规则位于 `ui/field_panels/layout_diagnostics_rules.py`，CLI 位于 `scripts/check_layout_diagnostics.py`。
- 右侧栏折叠使用 `CollapsibleSidePanel`，不要在页面中重复写折叠逻辑。
- 主图放大使用 `PlotCard(expand_callback=...)` 或报告页独立预览对话框，后续新增主图应继续沿用该接口。
- Windows 真机截图目录应包含 `layout_diagnostics.json` 与 `layout_check_report.json`；先看 pass/fail，再看截图。

## v0.9.24 交接补充

- 项目和测线删除必须继续走 `core.field_project_operations`，不得在 UI 回调里直接 `rmtree`。
- 测线删除当前为直接删除项目目录内关联文件；项目删除当前为直接删除项目文件夹，均不得触碰项目目录外的原始导入来源文件。
- 删除后必须刷新项目树、测线清单、首页统计、空间成果和报告状态。

## v0.9.24 交接补充

- `metadata/project_state.json` 保存 selected_line_id、dirty、stale_reasons 和 last_events。
- 报告失效不要只改 UI，应通过 `ProjectLinkageController.emit(ProjectEventType.REPORT_MARKED_STALE, ...)` 或具体事件触发。
- 目标、空间、报告联动由依赖规则维护；后续批量处理必须复用该机制。
