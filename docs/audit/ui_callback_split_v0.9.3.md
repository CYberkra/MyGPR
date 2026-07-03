# MyGPR v0.9.3 UI 回调拆分记录

## 目标

解决 `field_workbench_window.py` 持续膨胀问题，优先拆出项目管理页和目标定位回调。

## 拆分结果

- `ui/field_panels/project_page.py`：项目管理页、项目操作、导入、质检、备份、测线清单导出。
- `ui/field_panels/target_actions.py`：目标标注来源、标注新增/删除/保存、自动识别辅助、目标 B-scan 点击回调。
- `ui/field_workbench_window.py`：保留主窗口壳、全局状态协调、导航、测线处理页和公共绘图/表格工具。

## 约束

后续新增项目操作不得继续写入 `field_workbench_window.py`；应优先进入 `ProjectPageMixin` 或 core/service 层。
后续新增目标标注行为不得继续写入 `interpretation_page.py`；应优先进入 `TargetActionsMixin`。
