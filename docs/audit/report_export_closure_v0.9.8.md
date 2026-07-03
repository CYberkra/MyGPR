# MyGPR v0.9.8 成果报告导出闭环

## 范围

本轮接入稳定可审计的报告包导出，输出 CSV / JSON / HTML。PDF 暂不接入，避免 beta 阶段引入额外版式和字体依赖。

## 输出目录

```text
reports/report_<timestamp>/
├─ report_manifest.json
├─ json/project_report_summary.json
├─ tables/line_manifest.csv
├─ tables/quality_summary.csv
├─ tables/targets_summary.csv
├─ tables/processing_artifacts.csv
├─ tables/spatial_exports.csv
└─ html/project_report.html
```

同时写入：

```text
reports/latest_report_manifest.json
project.json -> reports.status = 已生成
```

## 数据来源

- 项目元数据：project.json
- 测线清单：FieldProjectStore.list_lines()
- 数据质检：raw/<line_id>/<line_id>_quality_report.json
- 处理结果：processed/<line_id>/*_processing_manifest_*.json / *_params_*.json / *.npy
- 目标标注：targets/<line_id>_targets.csv
- 空间成果：spatial/<line_id>_targets_xy.csv

## 风险控制

- 不生成虚假目标或坐标。
- 无质检报告的测线标记为“未质检”。
- 无空间 CSV 的测线在空间成果汇总中保留空路径。
- 所有导出均为项目目录下相对路径。
