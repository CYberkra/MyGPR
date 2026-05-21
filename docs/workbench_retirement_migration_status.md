# Workbench 退役迁移状态（UI-STAB-001）

## 当前定位
- 主路径：`app_qt.py` 主界面（`日常处理` / `调参与实验` / `显示与对比` / `质量与导出`）。
- `ui/gui_workbench.py`：**Legacy Fallback**，保留回退，不删除。

## 已迁移到主界面
- 显示与对比页已可直接访问：
  - 摆动图（`view_style=wiggle`）。
  - 深浅主题切换按钮（主界面入口，不再仅 Workbench 可切换）。
- 日常处理页已可切换“显示高级参数”，不再仅 Workbench 才能编辑高级算法参数。
- no-prior guard 主链路仍以主界面策略为源，Workbench 通过回调复用。

## 仍为 Workbench 兼容能力
- Workbench 的组合式深度实验交互（模板编排、预览提交等）仍保留。
- 部分 Workbench 专属可视化微交互仍存在，不影响主路径使用。

## 风险评估（若立即删除 Workbench）
- 风险：中高。
- 原因：仍有用户依赖 Workbench 的历史实验链路和专属交互，不适合在本阶段直接移除。

## 建议退役阶段
1. 标识阶段：入口明确标注 Legacy Fallback（已完成）。
2. 迁移阶段：把仍需保留的独占能力逐项迁移到主界面。
3. 对齐阶段：做主界面/Workbench 能力与行为一致性回归。
4. 删除阶段：仅在单独评审通过后移除 Workbench。
