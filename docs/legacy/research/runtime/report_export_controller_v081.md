# ReportExportController v0.8.1

`ui/report_export_controller.py` 是 MyGPR V0.8.x 架构整理的第一个 controller 拆分模块。它负责把报告导出和 Evidence package 生成逻辑从 `app_qt.py` 中移出。

## 职责

`ReportExportController` 负责：

- 导出当前 B-scan 600 DPI 图像。
- 生成 `report.md` 与 `report.html`。
- 写入 Evidence sidecar：
  - `manifest.json`
  - `evidence_index.json`
  - `workflow.json`
  - `processing_chain.json`
  - `params.json`
  - `display_settings.json`
  - `input_identity.json`
  - `software_version.json`
  - `method_registry_version.json`
  - `environment_summary.txt`
  - `runtime_log.txt`
  - `runtime_events.json`
  - `warnings.json`
  - `roi.json`
  - `figure_manifest.json`
  - `claim_boundary.txt`
  - `audit_note.md`

## 当前实现策略

为了降低 V0.8.1 的回归风险，controller 暂时采用 host-forwarding 模式：

```python
controller = ReportExportController(main_window)
```

controller 内部报告相关方法已经迁移；其他通用 UI / 数据方法通过 `host` 访问，例如：

- `_default_output_dir()`
- `_apply_preprocess()`
- `_build_time_axis()`
- `_get_crop_bounds()`
- `_log()`
- `_json_safe()`

这保证了报告逻辑独立，同时避免一次性重写主窗口状态接口。

## 兼容性

`GPRGuiQt` 继续保留以下兼容 wrapper：

- `generate_report()`
- `_build_processing_chain_export()`
- `_build_report_input_identity()`
- `_build_report_display_params()`
- `_write_report_sidecars()`
- `_write_branded_report_html()`

旧测试和 UI signal 连接无需修改。

## 后续方向

当 `ProcessingLineageController`、`BscanInteractionController` 等拆分完成后，可以逐步减少 host-forwarding，改成显式传入稳定的数据上下文对象。
