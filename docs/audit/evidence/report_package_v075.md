# MyGPR V0.7.5 Evidence Report Package

`生成报告包` 会在当前输出目录下创建一个规范化包目录：

```text
MyGPR_Evidence_Report_YYYYMMDD_HHMMSS_xxxxxx/
├─ report.md
├─ report.html
├─ bscan_current_600dpi.png
├─ manifest.json
├─ evidence_index.json
├─ workflow.json
├─ processing_chain.json
├─ params.json
├─ display_settings.json
├─ input_identity.json
├─ software_version.json
├─ method_registry_version.json
├─ environment_summary.txt
├─ runtime_log.txt
├─ warnings.json
├─ roi.json
├─ figure_manifest.json
├─ claim_boundary.txt
└─ audit_note.md
```

## 文件角色

- `manifest.json`: 包级 manifest，schema 为 `mygpr.report_manifest.v3`。
- `evidence_index.json`: 包内 artifact 索引，方便外部工具快速读取。
- `workflow.json`: 最近一次运行/当前方法选择的 workflow 记录。
- `processing_chain.json`: B-scan 当前处理链路和历史 stepper 摘要。
- `params.json`: 真实处理参数，不包含 display-only 设置。
- `display_settings.json`: 色图、拉伸、网格、坐标轴等 display-only 设置。
- `roi.json`: 手动 ROI / AutoTune ROI 状态和 ROI claim。
- `figure_manifest.json`: 图件角色、DPI、display-only 属性和显示参数摘要。
- `input_identity.json`: 输入路径、文件名、大小、mtime 和小文件 sha256。
- `software_version.json`: MyGPR 版本、VERSION 文件、Python 入口。
- `method_registry_version.json`: 方法注册表 key 列表和 sha256 fingerprint。
- `environment_summary.txt`: Python、Qt、numpy、pandas、matplotlib、PyWavelets 等环境摘要。
- `claim_boundary.txt`: no-prior / 当前报告 claim boundary。
- `audit_note.md`: 给人工审阅的简要 audit note。

## 约束

该包记录当前处理状态与可见图件。除非输入来自已验证的 synthetic paired contract 或其他 ground truth contract，否则报告包本身不证明地下目标检测正确性。
