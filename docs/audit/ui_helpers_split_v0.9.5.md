# MyGPR v0.9.5 UI 工具拆分审计

## 范围

本轮只拆分主窗口内的首页构建、通用表格工具和预览绘图辅助，不改变核心数据导入、坐标投影、算法处理、质检或成果保存协议。

## 拆分结果

- `ui/field_panels/home_page.py`：首页项目总览。
- `ui/field_panels/table_utils.py`：通用表格创建与填充。
- `ui/field_panels/preview_helpers.py`：B-scan 和轨迹预览绘图辅助。
- `ui/field_workbench_window.py`：继续保留主窗口状态、导航、项目状态同步和通用窗口样式。

## 行数变化

- `field_workbench_window.py`：1212 行 -> 907 行。

## 风险

- 本轮为 UI 结构拆分，主要风险是 mixin 方法解析顺序和导入缺失。已通过 compileall、回归测试和 GUI 截图验证。
