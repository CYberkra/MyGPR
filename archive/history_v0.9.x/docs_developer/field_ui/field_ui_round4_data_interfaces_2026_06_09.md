# MyGPR Field UI Round4：真实数据接口层接入

## 本轮目标

Round4 在 Round3 的项目持久化基础上，新增真实数据工作流接口层，使 UI 不再只依赖静态/假图，而是通过统一数据模型读取、处理、检测和空间插值。

## 新增核心模块

- `core/gpr_data_model.py`
  - `GPRDataSet`：统一 B-scan 数据结构。
  - 支持 `CSV / NPY / NPZ / H5 / HDF5` 的矩阵读取入口。
  - 统一字段：`matrix`、`distance_axis_m`、`time_axis_ns`、`depth_axis_m`、`dielectric_constant`、`metadata`。
  - 提供 `GPRDataSet.synthetic()` 作为 demo 和无真实矩阵时的兜底。

- `core/gpr_processing_pipeline.py`
  - `ProcessingParams`：处理参数对象。
  - `process_gpr_dataset()`：基础处理链。
  - 当前实现包括去直流、背景去除、简易带通、SEC 增益。

- `core/target_detection.py`
  - `TargetCandidate`：目标候选统一结构。
  - `detect_targets()`：可替换的目标检测接口。
  - 当前为确定性能量峰候选算法，后续 PGDA-CSNet 模型可替换该入口。

- `core/trajectory_model.py`
  - `TrajectoryModel`：RTK/IMU 轨迹模型。
  - 支持轨迹 CSV 读取、demo 轨迹生成、按里程插值坐标。

## 项目结构新增文件

在 `runtime_projects/field_demo_project/` 下新增或更新：

```text
raw/L03/L03_gpr_dataset.npz
raw/L03/L03_gpr_meta.json
raw/L03/L03_trajectory.csv
processed/L03/L03_processed_*.npy
processed/L03/L03_params.json
targets/L03_targets.csv
spatial/L03_targets_xy.csv
```

## GUI 接入点

- 项目管理页显示项目已接入 GPR 矩阵和 RTK/IMU 轨迹。
- 测线处理页从 `GPRDataSet` 渲染 B-scan，参数变化触发 `process_gpr_dataset()`。
- 保存处理结果写入 `processed/<line_id>/` 并回写 `project.json`。
- 目标定位页的“自动识别辅助”调用 `detect_targets()`，输出候选目标。
- 新增/检测目标根据 `TrajectoryModel.interpolate()` 写入空间坐标。
- 空间成果 CSV 使用轨迹插值结果，而不是固定线性假坐标。

## 验证

```text
python -m py_compile core/gpr_data_model.py core/gpr_processing_pipeline.py core/target_detection.py core/trajectory_model.py core/field_project_store.py ui/field_workbench_window.py app_qt.py
python -m pytest tests/test_field_project_store.py tests/test_round4_data_interfaces.py -q
```

当前测试：`6 passed`。

## 边界说明

- DZT / RD3 / DT1 等厂商格式本轮只预留适配层，尚未实现完整解析。
- 目标检测当前是启发式接口实现，不代表最终 PGDA-CSNet 模型效果。
- 若导入的 CSV 不是足够大的二维雷达矩阵，会作为原始侧车/项目证据保留，并自动使用 demo B-scan 矩阵兜底，避免小型配置 CSV 被误渲染为雷达图。
