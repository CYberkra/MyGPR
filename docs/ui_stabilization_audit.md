# UI Stabilization Audit（UI-STAB-001）

## 范围与约束核对
- 仅修改 MyGPR 源码与文档。
- 未删除 Workbench。
- 未修改 `processing_engine` 算法语义。
- 未修改 AutoTune scoring。
- 未修改 motion compensation 算法语义。
- 未修改 MyGPR-Evidence 仓库。

## 问题逐项自审（1-9）
1. Tab 内容溢出与尺寸问题：**fixed/partial**
- 已修复：`质量与导出` 导出按钮从单行 4 按钮拆为 2+2 行，窄宽度可用性提升。
- 已修复：`显示与对比` 多处固定宽度输入改为 min/max 范围，降低截断。
- 剩余：极端窄窗口下 Qt 原生控件仍可能触发局部横向滚动。

2. 质量/导出页布局溢出：**fixed**
- 导出操作按钮重排，避免右侧超出。
- 图表区与日志区保持可访问。

3. 右侧扩展后大面积空白：**partial**
- 主绘图区/空状态卡片/plot host 显式设置 `Expanding`。
- 剩余：图像内容本身受数据比例影响，仍可能出现“有意留白”，但不再是固定宽子控件导致的假空白。

4. B-scan 鼠标操作卡顿/回滚：**fixed/partial**
- 拖动平移绘制改为节流 `draw_idle`（60Hz 上限）。
- 坐标标签更新节流（40Hz 上限）。
- Shift 拖框 ROI 改为复用 patch（`set_bounds`），避免每帧删除重建。
- 保持：视图交互不触发处理算法，不写入处理历史。

5. 3D 地质预览看不到航迹/越界：**fixed**
- 3D 轴限不再只取最后一个 payload，改为汇总所有可见层 + 可选 B-scan curtain 的统一 bounds。
- 无数据提示改为“暂无三维地理参考数据 / 暂无航迹数据”。

6. B-scan 标题/工具栏缺少处理链路：**fixed**
- 新增主图工具栏链路标签 `处理链路: ...`。
- 新增链路 tooltip（数据源 + 步骤链 + 视图交互不入链说明）。
- 单图标题附加链路文本（如 `Raw -> dewow -> ...`）。

7. Workbench 独占控制与高级参数：**fixed/partial**
- 已修复：`日常处理` 新增“显示高级参数”，可直接编辑完整参数集。
- `motion_compensation_v2` 高级参数（APC offsets / 安全阈值）可在主界面展开后编辑。
- 剩余：Workbench 仍保留部分深度实验交互，作为 fallback。

8. Workbench 退役准备：**fixed**
- 主界面入口按钮改为“进入旧工作台（Legacy）”并加说明 tooltip。
- Workbench 顶部标题标记 `Legacy Fallback`。
- 新增迁移状态文档：`docs/workbench_retirement_migration_status.md`。

9. 最终自审与交付：**in_progress（本文件）**
- 本文档记录已修复项、剩余项、风险与后续建议。

## 变更文件（本任务）
- `app_qt.py`
- `ui/gui_basic_flow.py`
- `ui/gui_advanced_settings.py`
- `ui/gui_quality_log.py`
- `ui/gui_workbench.py`
- `tests/test_gui_presets.py`
- `docs/workbench_retirement_migration_status.md`
- `docs/ui_stabilization_audit.md`

## 风险等级
- 中等：本次是 UI 行为整合与交互节流，涉及主窗口高频路径，但未触碰算法语义层。

## 已知剩余风险
- 极端窄窗口下仍可能有个别组合控件显示拥挤（Qt 原生控件尺寸限制）。
- B-scan 超大矩阵下仍受 Matplotlib 本身性能上限影响；本次仅降低交互抖动与无效重绘。
- Workbench 仍存在 legacy 专属交互，尚未完全退役。

## 建议后续任务
1. UI-STAB-002：对主图进行“交互中低分辨率预览 + 停止后全分辨率恢复”的 LOD 策略。
2. UI-STAB-003：进一步收敛 Workbench 独占交互，并在主界面补齐对应入口后进入删除评审。
3. UI-SMOKE-001：建立固定窗口尺寸集（窄/中/宽）的截图式 UI 回归基线。

