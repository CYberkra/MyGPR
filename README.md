# GPR GUI

PyQt6 GUI (`app_qt.py`) to load CSV/文件夹数据, display B-scan, and apply processing methods from `PythonModule/` (SVD background, F-K filter, Hankel SVD, sliding average, etc.).

## Features
- 导入 CSV 矩阵文件 或 A-scan 文件夹（自动选择格式）
- Display B-scan (matplotlib)
- Method list (original + researched)
- Per-method parameter inputs (window width, time, rank, etc.)
- 多页面（Tab）结构：**日常处理 / 调参与实验 / 显示与对比 / 质量与导出**
- 一键推荐处理链：**快速预览 / 稳健成像 / 高聚焦**
- 运行后质量看板：`focus_ratio`, `hot_pixels`, `spikiness`, `time_ms`
- 一键导出质量快照（CSV+JSON）

## Requirements
- Python 3.8+
- numpy, pandas, matplotlib, scipy
- PyQt6
- PyQt6-Fluent-Widgets[full]

Install deps:
```bash
pip install -r requirements-dev.txt
```

## Run
```bash
python app_qt.py
```

Or double-click `启动GPR.bat` on Windows.

## Sample data
- Example B-scan CSV: `sample_data/sample_bscan.csv`
2) Click **Import CSV** and select `sample_data/sample_bscan.csv`
3) Select a method and set parameters
4) Click **Apply Selected Method** to see output

## CLI Batch MVP（Phase-1）
已提供最小可运行 CLI 批处理主链路（不含报告引擎），入口：`cli_batch.py`。

### 1) 配置校验
```bash
python cli_batch.py validate --config config/cli_batch_mvp_example.json
```

### 2) 运行批处理
```bash
python cli_batch.py run --config config/cli_batch_mvp_example.json
```

### 3) resume 接口（占位）
```bash
python cli_batch.py resume --summary output/cli_batch_mvp/summary_xxx.json
```

说明：
- 支持 `run / validate` 主命令；`resume` 先保留稳定接口（phase-2 再增强）
- 优先复用既有处理链语义：core 方法调用参数与 GUI 保持一致
- 输出目录由配置 `output_dir` 控制，默认示例写入 `output/cli_batch_mvp/`
- 每次 `run` 会生成 `summary_*.json` 记录每个 job/step 的状态与产物路径

示例配置：`config/cli_batch_mvp_example.json`

## Template Report Engine v1（Phase-2 最小闭环）
已提供基于模板的报告引擎 v1：
- 报告数据模型与字段约定：`docs/report-engine-v1.md`
- 单模板渲染器：`report_engine_v1.py`（HTML）
- 渲染入口：`scripts/render_report_v1.py`
- 标准化输入样例：`config/report_input_v1_example.json`
- 最小单测：`tests/test_report_engine_v1.py`
- 冒烟脚本：`scripts/smoke_report_v1.sh`

从 CLI summary 生成报告：
```bash
python scripts/render_report_v1.py \
  --summary output/cli_batch_mvp/summary_20260315_132054.json \
  --output output/reports/report_v1_from_summary.html
```

## Repo layout
- `app_qt.py` — main GUI (Qt, default entry)
- `app.py` — legacy compatibility entry (deprecated for new usage)
- `app_enhanced.py` — legacy prototype entry (deprecated)
- `archive/legacy_snapshots/` — archived historical GUI snapshots (`gpr_gui_kir_fixed_v5/v6/v8.py`)
- `read_file_data.py` — minimal CSV IO helpers
- `output/` — generated results
- `docs/release-rollout.md` — 发布计划与灰度/回滚流程（RC → Canary → Stable）

## Docs
- 发布流程文档：`docs/release-rollout.md`
- 结构审视改造清单：`docs/structure-review-checklist-2026-04-10.md`

## Archive Automation
阶段性稳定版本可以用脚本自动归档到 Obsidian：

```bash
python scripts/archive_checkpoint.py \
  --summary "当前 GUI 版本已验证可用" \
  --changes "完成 auto-tune round2 升级" \
  --changes "调整应用方法菜单与调参与实验标签页" \
  --risks "调参与实验页文案还可继续收敛" \
  --next-steps "如确认稳定，建议 commit + tag + push"
```

默认会写入：
- `D:\ClawX-Data\Obsidian\uav_gpr\01-项目\版本归档\`
- 并自动更新 `01-项目/版本归档索引.md`

## Preflight（发布前自动检查）
在发布前可运行一键 preflight，覆盖版本号校验、依赖检查、关键路径冒烟（启动/导入/处理/导出）以及质量门禁快速版。

```bash
python scripts/preflight_check.py
```

常用参数：
- `--allow-dev-version`：允许未写入 RELEASE_VERSION/VERSION 时以 `dev-*` 继续（默认会判失败）
- `--strict-semver`：将非 semver 版本号判为失败（默认仅告警）
- `--sample <path>`：指定冒烟测试样例 CSV
- `--method-key <name>`：指定处理方法（默认 `dewow`）

输出报告：
- `output/preflight_report.json`
- `output/preflight_report.md`
