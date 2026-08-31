# MyGPR Backend API v1

## 入口

```python
from mygpr.interfaces.backend import MyGPRBackend

backend = MyGPRBackend.create_default()
```

稳定服务：

- `backend.processing`：数组级单方法和流水线处理；
- `backend.autotune`：自动选参；
- `backend.projects`：项目创建、打开、测线数据、完整性、备份和恢复；
- `backend.project_processing`：读取项目测线、执行流水线并提交处理成果；
- `backend.reporting`：工程报告包生成；
- `backend.jobs`：异步执行、进度、取消、警告、成果事件和状态轮询。

公开接口不返回 QWidget、QThread、SQLite connection、h5py Dataset 或算法内部 callable。

## 项目创建与测线数据

```python
summary = backend.projects.create_project(
    "D:/projects/demo",
    name="营山航空探地雷达项目",
)

line = backend.projects.save_line_dataset(
    summary.project_id,
    "L01",
    bscan,
    name="1 号测线",
    length_m=120.0,
    time_window_ns=700.0,
)

info = backend.projects.get_dataset_info(summary.project_id, "L01")
window, sample_indices, trace_indices = backend.projects.read_window(
    summary.project_id,
    "L01",
    sample_start=0,
    sample_end=400,
    trace_start=100,
    trace_end=600,
)
```

`read_window` 在 HDF5 数据源上先切片、后降采样，不向前端暴露 HDF5 句柄。`read_dataset` 明确表示完整矩阵物化，仅用于需要全数据的处理任务。

## 数组级处理

```python
from mygpr.domain.processing.models import ProcessingRequest

result = backend.processing.execute_method(
    ProcessingRequest(
        data=bscan,
        method_id="dewow",
        params={"window": 23},
    )
)
```

## 流水线

```python
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep

pipeline = PipelineDefinition(
    name="常规预处理",
    steps=(
        PipelineStep(method_id="dewow", params={"window": 23}),
        PipelineStep(method_id="agcGain", params={"window": 11}),
    ),
)
```

数组级同步执行：

```python
result = backend.processing.execute_pipeline(bscan, pipeline)
```

项目级异步执行并回写 HDF5/SQLite：

```python
job_id = backend.submit_project_pipeline(
    summary.project_id,
    "L01",
    pipeline,
    result_name="常规预处理结果",
)
snapshot = backend.jobs.wait(job_id, timeout=120)
artifact = snapshot.result
```

处理成果包含稳定 `artifact_id`、项目内数据引用、分支和父成果标识，并通过任务事件发布 `artifact_created`。

## 完整性、备份与恢复

```python
report = backend.projects.audit_project(
    summary.project_id,
    repair_context=True,
    clean_staging=True,
)

backup = backend.projects.backup_project(
    summary.project_id,
    "E:/MyGPR_Backups",
)

restored = backend.projects.restore_project(
    backup.archive_path,
    "D:/restored-projects",
)
```

备份在生成后执行 ZIP 成员与 SHA-256 校验。恢复拒绝路径穿越、符号链接、重复成员和清单不一致。

## 报告

```python
package = backend.reporting.generate_package(
    summary.project_id,
    package_name="final_delivery",
)
```

报告路径为项目根目录下的相对路径，前端通过 `summary.root_path / package.pdf_path` 解析。

## 异步任务

```python
job_id = backend.submit_project_report(summary.project_id)
snapshot = backend.jobs.wait(job_id, timeout=300)
new_events = backend.jobs.events(job_id, after_sequence=last_sequence)
```

任务状态：`queued`、`running`、`completed`、`failed`、`cancelled`。事件 sequence 单调递增。

## 兼容范围

Backend API 版本保持 `1.0`。本阶段是向 v1 增加项目能力，不删除或重命名既有字段和方法。

- 原 `core.auto_tune` 历史入口继续可用；
- 旧 Qt 前端无需同步重构；
- 原项目格式、HDF5 数据集路径和 SQLite Catalog 保持兼容；
- 新前端应只消费本 API，不直接访问 `core`、SQLite 或 HDF5。
