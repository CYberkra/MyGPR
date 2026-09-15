# MyGPR Domain Context

## Core Concepts

### Project
 fieldwork task. A project groups all survey lines captured during a single fieldwork assignment. It owns a root directory, a manifest (`project.json`), and subdirectories for raw data, processed results, targets, and spatial exports.

### Line
The data for a single survey line. Each line has a stable identifier (e.g. `L01`), a GPR dataset file, metadata, and may have zero or more processed artifacts.

### Dataset
The raw GPR matrix and its metadata for one line. Loaded via `load_gpr_dataset(line_id)` and accessed in bounded windows via `read_window()`.

### Artifact
An immutable processing result produced by running a processing pipeline on a line. Stored under `processed/<line_id>/` with a descriptor, manifest, and params JSON.

### Processing Pipeline
A sequence of processing steps submitted as a job. Submitted via `submit_project_pipeline()` and watched through the job bridge.

### Target
An annotation object (e.g. suspected pipe, cavity) placed on a line. Stored as CSV under `targets/<line_id>_targets.csv`. Currently only storage is implemented; the annotation UI page is not yet developed.

### Interface
A basal/overburden interface annotation for a line. Stored separately from targets. The interpretation UI controller exists but is not yet fully wired.

### Spatial Result
A derived coordinate export (trajectories, targets, interfaces) aggregated at the project level. Distinct from per-line processing artifacts.

### Job
An asynchronous backend task (pipeline run, import, quality check, etc.). Managed by `JobBridge`, which polls and emits Qt signals.

### Preview Bundle
A bounded, downsampled view of a dataset or artifact for display in the UI. Never loads the full matrix into memory.

### 处理阶段（Processing Stage）
按处理目的划分的一类操作，例如低频漂移校正、去背景、去噪、增益和滤波。阶段与具体算法不同，同一阶段可以有多种算法。

### 处理步骤（Processing Step）
处理链中的一次具体算法操作，包含所选算法及其参数。处理链同时包含步骤的先后顺序。

### 自动处理（Automatic Processing）
根据 B-scan 数据的情况选择处理阶段、具体算法、执行顺序和参数，并执行所选处理链。它不同于仅为用户预先选定的一个算法调整参数。

### 目标信号（Target Signal）
来自当前项目所关注地下结构或异常的雷达回波。目标信号是数据中的物理响应，不等同于 Target 所指的人工标注对象；调查对象随项目目的而变化。

### 地表与直达干扰（Surface and Direct-wave Interference）
本项目希望抑制的地表反射、天线直达及空中耦合等非地下目标成分。它们不等同于所有水平相干回波；地下层状界面的反射也可能呈水平形态。

## Unresolved / Pending Clarification

- **Artifact vs Result**: code uses both terms; the business distinction is not yet confirmed.
- **Processing as verb**: whether "processing" refers only to running algorithms, or also to configuring parameters (drafts).