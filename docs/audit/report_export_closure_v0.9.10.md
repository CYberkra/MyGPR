# MyGPR v0.9.13 成果报告导出闭环

## 当前输出

`core/field_report_export.py` 可生成项目报告包，包含：

- HTML 报告
- JSON 摘要
- CSV 测线清单
- CSV 质量统计
- CSV 目标点统计
- CSV 处理成果清单
- CSV 空间成果清单
- PDF 报告

## UI 入口

成果报告页包含：

- `生成报告包`
- `打开报告目录`
- `生成/打开 PDF`

PDF 按钮不再是占位提示；会调用报告包生成流程并打开生成的 PDF。

## 后续计划

下一轮补充 Excel 报告：`project_report.xlsx`，建议包含项目信息、测线清单、质量统计、目标点、空间成果和处理记录。
