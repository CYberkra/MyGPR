# MyGPR 第十二阶段：第二轮定向瘦身与后端发行包报告

## 结论

本轮完成了预定的最后一轮后端定向拆分。没有修改 Backend API v1、项目存储格式、算法 ID 或数值公式，也没有删除旧前端回退入口。

后端不需要继续进行全仓库式重构。当前最合理的动作是冻结本阶段基线，等待新前端合并；后续只针对缺陷、性能瓶颈和实机验收结果做小范围修改。

## 本轮拆分

### 1. 工程报告导出

`core/field_report_export.py` 从 1079 行降至 472 行，仅保留报告包事务编排。新增：

- `core/report_export_models.py`：报告结果契约；
- `core/report_export_rows.py`：规范化证据表和哈希工具；
- `core/report_export_renderers.py`：HTML、PDF、XLSX、图件与校验清单渲染。

历史公开入口 `REPORT_PACKAGE_SCHEMA`、`ReportPackageResult` 和 `generate_project_report_package` 保持不变。

### 2. 传感器同步

`core/sensor_sync.py` 从 671 行降为 20 行兼容门面。新增：

- `core/sensor_sync_models.py`：时钟、标定、诊断和同步结果模型；
- `core/sensor_sync_engine.py`：RTK、IMU、测高与雷达时间同步数值内核；
- `core/sensor_sync_io.py`：轨迹、manifest 和逐道 metadata 持久化。

旧导入路径和公开对象保持不变。

### 3. AutoTune 流程级编排

`core/auto_tune_pipeline.py` 从 1006 行降至 525 行。新增：

- `core/auto_tune_pipeline_models.py`；
- `core/auto_tune_pipeline_geometry.py`；
- `core/auto_tune_pipeline_evaluation.py`；
- `core/auto_tune_pipeline_summary.py`。

保留了原模块中的 `auto_tune_method` 可替换点，因此既有 monkeypatch 测试和旧调用行为未改变。重复的 ROI 边界函数已删除并改为单一实现。

### 4. AutoTune 对比证据导出

`core/auto_tune_comparison_export.py` 从 1041 行降至 214 行。图像、表格、证据清单、Markdown 和通用序列化分别拆入独立模块。公开函数和 `_locked_display_spec` 兼容入口保持不变。

## 债务指标

- 超过 1000 行的生产模块：20 → 17；
- 发布债务基线已下调至 17，后续版本不得回升；
- 长期目标更新为 15；
- Backend API v1、项目 Schema 和算法目录无破坏性变化。

当前仍超过 1000 行的生产模块主要集中在旧 Qt 前端和兼容层。它们应在新前端通过端到端验收后整体退出，而不是现在继续拆分。

## 发行包瘦身

本轮同时生成独立的 backend-only wheel：

- 不包含 `ui/`；
- 不包含 `compatibility/`；
- 不包含 `app_qt.py`；
- 保留 `core/`、`mygpr/`、`PythonModule/`、批处理入口和 Backend API。

与第十一阶段完整桌面 wheel 相比：

| 指标 | 第十一阶段完整 wheel | 第十二阶段 backend wheel | 变化 |
|---|---:|---:|---:|
| 文件大小 | 1,269,299 B | 776,231 B | -38.85% |
| Python 行数 | 114,976 | 69,162 | -39.85% |
| 包含旧 UI | 是 | 否 | 已移除 |
| 包含兼容前端 | 是 | 否 | 已移除 |

完整桌面 wheel 因把四个超大文件拆成多个职责模块，大小增加约 0.82%。这是模块元数据与导入头增加造成的正常变化，不属于运行逻辑膨胀。后端部署应使用 backend-only wheel；桌面版本继续保留旧前端作为临时回退。

## 验证结果

### 工业自动化套件

- 49 项通过；
- 关键覆盖率门禁通过；
- 22 项需求、14 项风险追踪通过；
- 1517 个测试重复审计通过；
- 3/3 变异被捕获；
- 营山 3、6、7、9、L1、X1 六个完整实测文件验证通过。

### 本轮重构定向回归

- 报告、同步、AutoTune pipeline、对比证据及 GPRMAX 合同：44 项通过；
- 原生处理、迁移、运动补偿、CLI、API 和架构回归：86 项通过；
- 2 项 PyWavelets CLI profile 测试因当前环境未安装 PyWavelets 明确取消选择，未伪报通过。

### 门禁

- Python 编译：652 个文件通过；
- 架构、循环依赖、Schema、Backend API、项目格式、依赖合同、复杂度、技术债、测试分类、追踪、源码清单和发布清洁度全部通过；
- 完整 wheel 与 backend-only wheel 安装后 Backend smoke 均通过。

## 限制

本轮环境仍不能替代以下目标环境证据：

- PyWavelets 完整数值回归；
- Windows GUI、高 DPI、安装器和破坏性故障测试；
- NVIDIA CUDA 实机一致性与显存测试；
- RTK、IMU、雷达硬件在环测试。

## 停止条件

本轮已达到预定停止条件：

- 生产超大模块减少 3 个；
- 公开 API 和存储 Schema 不变；
- 关键回归和营山黄金基线通过；
- 后端独立发行包缩减超过 30%；
- 旧前端仍可回退。

因此，不建议继续为了降低行数而拆分后端。下一步应合并新前端并执行 Windows、CUDA、硬件在环和端到端 Release Candidate 验收。
