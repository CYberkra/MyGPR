# 前端契约对接状态

| 前端能力 | 后端状态 | 对接入口 |
|---|---|---|
| 方法目录 | 已实现 | `backend.processing.list_methods()` |
| 数组处理 | 已实现 | `ProcessingRequest` / `PipelineDefinition` |
| 自动选参 | 已实现 | `backend.autotune.tune_method()` |
| 异步任务 | 已实现 | `backend.jobs` |
| 进度与取消 | 已实现 | `ExecutionContext` / `CancellationToken` |
| 项目创建/打开/关闭 | 已实现 | `backend.projects` |
| 测线清单和元数据 | 已实现 | `list_lines()` / `get_dataset_info()` |
| B-scan 视窗读取 | 已实现 | `read_window()` |
| 项目处理与结果回写 | 已实现 | `backend.submit_project_pipeline()` |
| 成果列表与谱系 | 已实现 | `backend.projects.list_artifacts()` |
| 项目完整性审计 | 已实现 | `audit_project()` |
| 备份与恢复 | 已实现 | `backup_project()` / `restore_project()` |
| 报告导出 | 已实现 | `backend.reporting.generate_package()` |
| 原始厂商格式统一导入 | 后续阶段 | acquisition ports 尚待迁移 |
| 测线删除/重命名等全量编辑 | 后续阶段 | 需要补充明确的事务用例 |
| 处理结果按块流式执行 | 后续阶段 | 当前项目流水线明确物化完整矩阵 |

## 前端边界

新前端不得直接导入：

- `core.field_project_store`；
- `core.project_catalog`；
- `core.hdf5_line_container`；
- `core.processing_engine`；
- `h5py.Dataset` 或 SQLite connection。

报告和成果路径均为项目内相对引用。前端应使用当前项目的 `root_path` 解析，不自行推断存储目录结构。
