# MyGPR v0.8.80 项目管理页真实指标替换

## 目标

本轮将首页与项目管理页中的固定 demo 指标替换为从当前项目目录计算的真实状态。UI 不再自行拼装测线数量、导入数据量、辅助定位文件、目标数量、空间成果、报告状态和最近活动。

## 新增模块

- `core/field_project_status.py`
  - 读取 `project.json`、`raw/`、`processed/`、`targets/`、`spatial/`、`reports/`、`logs/`。
  - 输出 `ProjectStatusSnapshot`。
  - 为首页、项目管理页、任务/检查/交付/日志表提供统一数据源。

## UI 边界

- `ui/field_workbench_window.py` 只消费 `ProjectStatusSnapshot`。
- 项目树从真实测线、处理结果、目标标注、空间成果文件生成。
- 数据质检入口根据 snapshot 给出提示，不再使用固定 L03 demo 文案。

## 未完成事项

- 正式项目配置向导仍需增强。
- 报告页真实导出闭环尚未完成。
- 厂商格式原生解析仍未接入。
