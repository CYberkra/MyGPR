# MyGPR v0.9.0 beta 功能边界

## 已纳入 beta 的能力

- 正式项目创建、打开、最近项目切换和项目设置。
- 营山 / 旧 MyGPR sidecar CSV 识别、预检、导入和标准化。
- 多文件批量导入、后台线程、取消后续文件、逐文件诊断表。
- 经纬度到 CGCS2000 / 3-degree GK 工程坐标投影。
- 项目树、测线清单、快速预览、空间成果、报告页的真实项目联动。
- 导入后数据质检与 B-scan 方向风险检测。
- 手动 B-scan 转置修正、修正前备份和 manifest 记录。
- 测线清单 CSV 导出。
- 项目 ZIP 备份。

## 不纳入 beta 的能力

- DZT/RD3/DT1 等厂商格式完整原生解码。
- 自动智能识别模型、Mask-RCNN 或 PGDA-CSNet 模型接入。
- 正式 PDF / Excel 交付报告生成闭环。
- 三维成果增强。
- 单文件解析过程的强制中断。

## 架构边界

- 项目文件操作继续放在 `core/field_project_operations.py` 和 store mixin，不写入按钮回调。
- CSV/GPR 数据解析继续放在 `core/gpr_data_model.py`。
- 质检继续放在 `core/field_data_quality.py`。
- 坐标投影继续放在 `core/coordinate_projection.py`。
- GUI 只负责触发服务层和刷新项目状态。

## beta 验收重点

1. Windows 本地启动稳定。
2. 新建项目、打开项目、导入单条 CSV、批量导入多条 CSV 不崩溃。
3. 导入后项目树、测线清单、快速预览和空间成果同步真实项目。
4. 数据质检报告生成正常。
5. B-scan 方向修正可追溯、可回退到备份文件。
6. 项目备份 ZIP 可生成并包含 `project.json`、`raw/`、`reports/` 等关键内容。
