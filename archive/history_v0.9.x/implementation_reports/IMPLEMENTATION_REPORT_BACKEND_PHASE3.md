# MyGPR 后端拆分第三阶段实施报告

## 1. 本阶段范围

本阶段继续冻结旧 Qt 前端，仅处理项目级后端闭环。目标是让未来 Qt/QML/Web/CLI 前端通过稳定 Backend API 完成：

1. 项目创建、打开、关闭和会话管理；
2. 测线数据写入、元数据读取和按视窗读取；
3. 项目测线处理流水线执行与成果回写；
4. HDF5/SQLite 成果谱系查询；
5. 项目完整性审计、保守修复和 staging 清理；
6. 校验型备份与安全恢复；
7. 工程报告包生成；
8. 全流程无 Qt 冒烟验证。

本阶段未修改旧前端页面、布局或交互架构。

## 2. 新增后端分层

### 2.1 Project Domain

新增：

- `mygpr/domain/project/models.py`
- `mygpr/domain/project/__init__.py`

建立 UI 和存储无关的公开模型：

- `ProjectSummary`
- `ProjectLine`
- `LineDatasetInfo`
- `ProjectLineData`
- `ProjectArtifact`
- `IntegrityIssue` / `IntegrityReport`
- `ProjectBackup` / `ProjectRestore`

公开模型不包含 QWidget、SQLite connection、h5py Dataset 或具体算法对象。

### 2.2 Project Application

新增：

- `mygpr/application/project/ports.py`
- `mygpr/application/project/service.py`
- `mygpr/application/project/processing_service.py`

其中：

- `ProjectRepositoryPort` 和 `ProjectSessionPort` 定义项目存储边界；
- `ProjectService` 管理多个打开项目，不向前端泄漏具体 Store；
- `ProjectProcessingService` 完成“读取测线 → 执行流水线 → 提交成果 → 发布 artifact 事件”。

### 2.3 Persistence Adapter

新增：

- `mygpr/infrastructure/persistence/field_project_adapter.py`

该 adapter 暂时包裹已经经过验证的：

- `FieldProjectStore`
- SQLite `ProjectCatalog`
- 分测线 HDF5 容器
- `ProjectIntegrityAuditor`
- 项目备份/恢复
- 报告导出器

application/domain 不再直接导入这些具体实现。该 adapter 已登记为受控迁移例外，后续可逐步替换为原生 persistence 模块。

### 2.4 Reporting

新增：

- `mygpr/domain/reporting/models.py`
- `mygpr/application/reporting/service.py`

`backend.reporting.generate_package()` 返回稳定 `ReportPackage`，报告路径保持为项目内相对引用。

## 3. Backend API v1 扩展

`MyGPRBackend` 新增：

- `backend.projects`
- `backend.project_processing`
- `backend.reporting`
- `submit_project_pipeline(...)`
- `submit_project_report(...)`
- `submit_project_backup(...)`

Backend API 版本继续保持 `1.0`，本次为兼容性新增能力，没有删除或重命名第一、二阶段接口。

项目级典型流程：

```python
backend = MyGPRBackend.create_default()
project = backend.projects.create_project(path, name="项目")
backend.projects.save_line_dataset(project.project_id, "L01", bscan)
job_id = backend.submit_project_pipeline(project.project_id, "L01", pipeline)
artifact = backend.jobs.wait(job_id).result
```

## 4. 数据安全改进

### 4.1 反序列化边界校验

修复了项目清单中的测线编号只在部分调用点校验的问题：

- `FieldLineRecord.from_dict()` 现在强制执行 `validate_line_id()`；
- `FieldProjectManifest.from_dict()` 校验 `lines` 类型并逐条规范化；
- `HybridProjectStorageBackend.line_container_path()` 再次校验 line id，并确认解析路径仍位于项目根目录内。

因此，被人工修改或损坏的 `project.json` 不能通过 `../`、嵌套路径或 Windows 保留设备名逃逸到受管目录之外。

### 4.2 处理成果取消与回滚

`FieldArtifactStoreMixin.save_processed_line()` 新增可选：

- `cancel_requested`
- `progress_callback`

HDF5 分块写入使用既有 staging/commit 机制；取消或 Catalog 提交失败时，未完成成果不会留在正式 artifact 路径。项目级任务通过统一 `ExecutionContext` 传递取消和进度。

### 4.3 前端隔离

项目创建、HDF5 读取、处理成果回写、审计、备份、恢复和报告生成均可在未安装/未导入 PyQt 的进程中执行。

## 5. 无 GUI 冒烟入口

新增：

- `backend_project_smoke.py`
- `mygpr/interfaces/cli/project_smoke.py`

流程包括：

1. 创建混合存储项目；
2. 写入合成 B-scan；
3. 提交 Dewow 项目处理任务；
4. 将结果保存为 HDF5 artifact 并登记 SQLite 谱系；
5. 执行项目完整性审计；
6. 验证未加载 Qt。

## 6. 验证结果

### 6.1 编译与门禁

- Python 编译检查：`592` 个文件通过；
- Architecture Policy：PASS；
- Schema Catalog：PASS，110 个已归属 Schema、106 个引用；
- Project Format Compatibility：PASS；
- Debt Budget：PASS；
- Complexity Budget：PASS；
- Source Package Manifest：PASS；
- Test Policy：PASS，266 个测试模块、26 个测试组、18 条规则。

### 6.2 自动测试

- 新增项目 Backend API 测试：`5` 项通过；
- 后端、AutoTune、混合存储、完整性、备份和报告选定回归集：`172` 项通过；
- 额外项目相关非 GUI 回归集：`96` 项通过；
- `backend_smoke.py --skip-autotune`：通过；
- `backend_project_smoke.py`：通过，成果引用为 `h5://...::/processing/artifacts/.../bscan`，完整性健康，Qt 未加载。

### 6.3 环境限制

当前环境未安装：

- PyQt6：未执行 GUI 测试；
- ruff：未执行 ruff critical lint；
- mypy：未执行完整静态类型检查。

本阶段没有修改旧 Qt 前端，且新后端测试明确验证项目级流程不需要 Qt。

## 7. 当前边界与下一阶段

项目级 Backend API 已闭环，但仍存在以下受控迁移边界：

1. `LegacyFieldProjectRepository` 仍包裹旧 `core` 存储与报告实现；
2. 项目处理目前明确物化完整 B-scan，尚未实现所有算法的分块/外存执行；
3. 厂商原始格式、GNSS/RTK/IMU 导入尚未迁入 acquisition ports；
4. 测线删除、重命名、批量导入等编辑用例尚未全部进入新 API；
5. 报告生成器内部仍是历史大函数，已通过 application port 隔离，但尚未物理拆分。

下一阶段建议优先：

- Acquisition domain/ports 与统一原始数据导入；
- Project 编辑事务（删除、重命名、批量导入）；
- 分块处理与资源预算执行器；
- 将 SQLite/HDF5 adapter 从单一 legacy wrapper 拆成原生 repository 实现；
- 为新前端补充稳定 DTO 序列化和 API 契约测试。
