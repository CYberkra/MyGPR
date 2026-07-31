# MyGPR v0.8.80 代码健康记录

## 已处理风险

- 首页和项目管理页的核心指标开始从项目文件计算，减少 demo 数据残留。
- 新增 `core/field_project_status.py`，避免 UI 继续直接统计 raw / processed / targets / reports 文件。
- 项目树和任务表开始读取真实项目状态。

## 剩余风险

- `ui/field_workbench_window.py` 仍包含较多页面协调逻辑，后续新增功能应继续放在 `core/` 或 `ui/field_panels/`。
- 报告页仍未接入真实导出闭环。
- 厂商格式导入仍处于识别和提示阶段。
