# MyGPR v0.9.9 按钮回调审计

## 现场工作台主页面

主页面包括：项目管理、测线处理、目标定位、空间成果、成果报告。

## 空间成果工具栏

| 按钮 | 当前状态 |
|---|---|
| 刷新空间成果 | 已接 `_action_refresh_spatial_results` |
| 导出坐标成果 | 已接 `_action_export_spatial_coordinates` |
| 打开三维视图 | 已接 `_action_open_3d_view` |
| 生成平面图 | 已接 `_action_generate_plan_map` |
| 图层控制 | 已接 `QMenu` 和 `_set_spatial_layer` |

## 成果报告页

| 按钮 | 当前状态 |
|---|---|
| 生成报告包 | 已接 `generate_project_report_package` |
| 打开报告目录 | 已接 `QDesktopServices.openUrl` |
| 生成/打开 PDF | 已接 `_action_generate_or_open_pdf_report` |

## 仍需后续审计

- 目标定位页的编辑 / 复核状态流转。
- 厂商格式导入对话框与真实 reader 的一致性。
- Excel 报告按钮接入后需新增审计记录。
