# MyGPR v0.8.76 项目存储边界拆分记录

## 本轮目标

v0.8.75 已经收敛主窗口膨胀风险，下一阶段主要风险转移到 `core/field_project_store.py` 职责偏多。本轮不新增业务功能，只拆分项目存储边界，避免项目管理、测线、处理结果、目标、空间成果和演示数据继续堆在同一个文件中。

## 拆分结果

`core/field_project_store.py` 现在只保留项目 manifest 协调职责：

- 打开 / 创建项目；
- 保证项目目录结构；
- 保存 `project.json`；
- 作为兼容入口组合各 store mixin。

新增模块：

```text
core/field_project_models.py     项目 schema、FieldLineRecord、FieldProjectManifest、原子写入工具
core/field_line_store.py         测线记录、raw 导入、GPR 数据集、RTK/IMU 轨迹
core/field_target_store.py       targets CSV 读写与目标字段归一化
core/field_spatial_store.py      spatial/ 目标 XY 导出
core/field_artifact_store.py     processed/ 处理结果、参数和 manifest 保存
core/field_demo_store.py         demo 项目补齐、存储统计、日志
```

## 兼容性原则

原有外部入口保持不变：

```python
from core.field_project_store import FieldProjectStore, FieldLineRecord, FIELD_PROJECT_SCHEMA
```

这保证 UI、测试和后续功能不用立即改 import 路径。

## 当前边界

- `FieldProjectStore`：项目 manifest 与目录结构；
- `FieldLineStoreMixin`：测线、raw、GPRDataSet、Trajectory；
- `FieldTargetStoreMixin`：目标标注 CSV；
- `FieldSpatialStoreMixin`：空间成果 CSV；
- `FieldArtifactStoreMixin`：处理结果 artifact；
- `FieldDemoStoreMixin`：示例项目和日志辅助。

## 未改变内容

- 不改变项目目录协议；
- 不改变 UI 布局；
- 不改变算法入口；
- 不接入预设流程处理；
- 不移除 `time_to_depth`；
- 不改变目标 CSV 和 spatial CSV 字段。

## 验证重点

- `FieldProjectStore.create()` 仍能创建项目；
- 示例 GPR 数据、轨迹、目标、空间成果仍能生成；
- 处理结果仍能保存到 `processed/`；
- 现有测试仍通过；
- GUI 能继续启动并截图。
