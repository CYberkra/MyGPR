# MyGPR v0.9.28 — UAV-GPR 基覆界面勘探工作台（旧版 CLAUDE.md，2026-07-30 归档）

> 此文件为被重写前的 CLAUDE.md 原文存档。其中引用的 `ui/field_panels/`、
> `ui/job_manager.py`、`scripts/run_quality_gate.py`、`gpr_gui.spec` 均已不存在，
> 五页布局描述也已过时，仅作历史参考。现行说明见根目录 CLAUDE.md。

## 项目概述

MyGPR 是面向 GPR / UAV-GPR 野外勘探项目的桌面软件，基于 PyQt6 + qfluentwidgets。
正式流程：项目建档 → GPR 导入与质检 → 雷达/RTK/IMU 同步 → 测线处理 → 连续基覆界面标注 → GIS 空间成果 → 正式报告。

默认主界面五个页面：`项目管理 -> 测线处理 -> 界面标注 -> 空间成果 -> 成果报告`

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| GUI | PyQt6 6.6+ + PyQt6-Fluent-Widgets 1.8+ |
| 科学计算 | numpy 1.24+, scipy 1.10+, pandas 2.0+, matplotlib 3.7+ |
| 数据处理 | h5py 3.10+, PyWavelets 1.5+, pyproj 3.6+ |
| GIS | rasterio 1.3+, fiona 1.9+ |
| 报告 | openpyxl 3.1+, matplotlib PDF/HTML |
| 测试 | pytest 8.0+ |
| 打包 | PyInstaller (`gpr_gui.spec`) |

## 强制工程约束

- 所有重计算、大文件 I/O、项目遍历和正式报告生成必须通过统一 Job Manager。
- 核心长任务必须支持协作式取消、阶段进度和事务提交；取消后不能覆盖最后有效成果。
- 多 GB 数据使用 mmap/分块 I/O，不得为了预检、显示或质检整体复制到 RAM。
- 雷达道时间戳是 RTK/IMU 同步主轴；超范围、超残差和无固定解区段必须显式标识。
- GIS 成果必须使用真实 CRS、GeoTIFF/DEM 或矢量图层；禁止装饰性伪底图。
- 正式标注对象是逐道连续基覆界面，原始 `trace/sample` 坐标不可被显示变换覆盖。
- 正式报告包必须包含审批信息、Excel/PDF/HTML、审计清单和 SHA-256 校验。

## 关键模块

```text
core/job_manager.py                # 无 Qt 的任务状态/取消合同
ui/job_manager.py                  # Qt 线程池与任务中心
core/field_import_preview.py       # 轻量导入预检
core/chunked_gpr_io.py             # 分块、可取消 I/O
core/sensor_sync.py                # 多传感器时间同步核心
core/sensor_sync_service.py        # 项目级同步事务
core/gis_layers.py                 # 离线 GIS 图层注册与读取
core/field_interface_store.py      # 连续界面成果与训练标签
core/field_report_export.py        # 正式报告包 v3
ui/field_panels/interpretation_page.py
ui/field_panels/spatial_page.py
ui/field_panels/delivery_page.py
```

## 数值与线程约定

- GPR 存储通常为 float32；算法可按需要使用 float64，但不得无界复制。
- 工作线程不得直接更新 Qt 控件；通过信号回到主线程。
- 算法内应按块检查取消标志，不能只在开始/结束检查。
- 正式文件先写隐藏临时文件/目录，通过校验后原子替换。

## 测试与发布

```bash
python scripts/run_quality_gate.py affected --plan
python scripts/run_quality_gate.py affected
python scripts/run_quality_gate.py merge --base origin/main
```

GUI/DPI、真实多 GB 数据和真实传感器文件必须在 Windows 目标机验收。
