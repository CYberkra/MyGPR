# MyGPR 后端拆分第六阶段实施报告

## 1. 阶段目标

本阶段聚焦原始测线导入、GNSS/RTK/IMU/测高数据、雷达道时间同步以及无人机运动补偿工作流的后端化。旧 Qt 前端未做结构性修改；新增能力全部通过 `MyGPRBackend` 暴露，可由 Qt、CLI 或后续新前端共同调用。

## 2. 已完成内容

### 2.1 Acquisition 垂直业务域

新增以下后端分层：

- `mygpr/domain/acquisition/`
- `mygpr/application/acquisition/`
- `mygpr/infrastructure/acquisition/`

Domain 层新增测线导入预检、标准化数据集、传感器流、同步参数、同步结果和项目同步成果等稳定模型。Application 层新增 `AcquisitionService`，Infrastructure 层通过受控适配器复用已验证的分块导入、sidecar 解析和传感器同步实现。

### 2.2 稳定 Backend API

`MyGPRBackend` 新增：

- `backend.acquisition.preflight(...)`
- `backend.acquisition.load_dataset(...)`
- `backend.acquisition.import_line(...)`
- `backend.acquisition.parse_sidecar(...)`
- `backend.acquisition.synchronize_streams(...)`
- `backend.acquisition.synchronize_project_line(...)`
- `backend.acquisition.motion_pipeline(...)`
- `backend.submit_line_import(...)`
- `backend.submit_sensor_sync(...)`

这些接口不暴露 Qt、SQLite connection、h5py Dataset、旧 `GPRDataSet` 或旧传感器对象。

### 2.3 测线导入闭环

项目测线导入已接入新后端服务：

1. bounded preflight，不完整扫描超大 CSV；
2. 后台分块复制源文件；
3. CSV / NPY / NPZ / HDF5 标准化；
4. 项目 HDF5 原始矩阵写入；
5. 项目清单、原始文件、导入清单和数据质检更新；
6. 进度和取消通过统一 `ExecutionContext` 传递。

适配器兼容历史导入链路中两种进度回调签名，避免分块导入与 HDF5 写入阶段的参数顺序冲突。

### 2.4 RTK / IMU / 测高 sidecar 标准化

新增 UI 无关的 `SensorKind`、`SensorStream` 和 `SensorSyncSettings`：

- RTK：时间、经纬度、本地坐标、高程、飞高、定位解、卫星数和 DOP；
- IMU：时间、滚转、俯仰、偏航和角速度；
- 测高：时间、AGL 高度、来源、SNR、目标数和有效性；
- 输入字段统一为一维、等长、显式类型的 NumPy 数组；
- sidecar 文件解析后不再向上层泄漏旧字典契约。

### 2.5 雷达—传感器同步与持久化

后端支持：

- 雷达逐道时间戳；
- RTK / IMU / 测高独立时钟偏置；
- constant / affine / piecewise 时钟模型；
- 最近时间残差上限；
- 禁止静默外推；
- 杆臂和触发延迟校正；
- RTK 覆盖率、固定解比例、间断和跳变诊断。

项目同步成果保存为：

- 轨迹 CSV；
- 同步 manifest JSON；
- 逐道 metadata NPZ；
- 原始 sidecar 副本。

### 2.6 同步元数据进入处理链

此前项目处理读取测线时未加载 `trace_metadata_path`，导致同步成果无法进入运动补偿算法。本阶段已修复：

- 使用受管理路径校验读取同步 NPZ；
- 禁止 pickle；
- 校验每个逐道字段长度与 trace 数一致；
- `ProjectLineData.trace_metadata` 正式返回同步数据；
- 新增 `backend.projects.read_trace_metadata(...)`，无需为了读取传感器数据而物化完整 B-scan；
- 项目级 `motion_compensation_v2` 已能消费持久化的高度、轨迹、姿态和对齐状态。

### 2.7 运动补偿工作流契约

新增 `MotionCompensationProfile` 和 `mygpr.motion_pipeline.v1`：

- 默认 integrated V2：`motion_compensation_v2`；
- atomic 模式：速度/道间距、姿态、高度；
- 可选振动与旋翼干扰抑制；
- 前端只选择模式和参数，不复制算法编排规则。

运动补偿算法本体当前仍由受控 legacy/global 处理适配器执行，本阶段完成的是业务契约、传感器数据入口和项目处理闭环。

### 2.8 无 GUI 冒烟入口

新增：

```text
python -m mygpr.interfaces.cli.acquisition_smoke
```

冒烟流程完整执行：

```text
创建项目 → 导入 NPY 测线 → 同步 RTK/IMU → 加载逐道 metadata
→ 运行 motion_compensation_v2 → 保存项目处理成果
```

执行过程中未加载 Qt。

## 3. 验证结果

### 3.1 自动测试

合并后的非 GUI 回归集共 **142 项通过**，覆盖：

- 新 Acquisition Backend API；
- 分块导入和预检；
- MyGPR 航空 CSV；
- 营山 CSV 导入；
- sidecar 解析；
- 传感器标定和时间同步；
- 高度、速度、姿态、振动和 V2 运动补偿；
- 运动补偿端到端证据导出；
- 项目 API、任务、处理和 AutoTune；
- HDF5/SQLite 存储恢复、备份、完整性和报告；
- 文件后端分块处理及处理成果索引。

测试仅出现当前 Linux 环境缺少 CJK 字体导致的 Matplotlib glyph warning，不影响结果。

### 3.2 门禁与编译

- Python 编译检查：**597 个文件通过**；
- 架构层级、循环依赖、迁移所有权和新代码规模：通过；
- Schema Catalog：通过，新增 `mygpr.motion_pipeline.v1` 所有权；
- 项目格式兼容性：通过；
- 复杂度预算：通过；
- 技术债 release ratchet：通过；
- 源码包清单：通过；
- 测试策略：通过；
- Acquisition 无 GUI 冒烟：通过，Qt 未加载。

完整 release collection 在当前环境仍因未安装 PyQt6 无法收集 GUI 测试；该限制不是本阶段代码回归。当前环境也未提供 ruff、mypy 和完整 Windows 打包工具链。

## 4. 当前限制

1. `mygpr.infrastructure.acquisition.legacy_adapter` 仍包装 `core.field_import_preview`、`core.gpr_data_model`、`core.sidecar_parsers` 和 `core.sensor_sync`，已登记到 1.1.0 前移除的迁移例外。
2. 运动补偿算法当前仍属于 legacy/global 执行路径，尚未原生化或文件后端化。
3. 复杂厂商格式仍遵循原项目的“原生支持 / 识别后提示转换”矩阵。
4. 目标 Windows 机器上的大文件导入中断、USB 断连、磁盘写满和设备实采联调仍需专项验收。
5. Qt 前端未进行结构性修改，后续新前端应只调用 Backend API。

## 5. 阶段结论

第六阶段已经打通工业后端所需的主链路：

```text
源文件 → bounded preflight → 可取消分块导入 → 项目 HDF5
→ RTK/IMU/测高解析 → 时间同步与诊断 → 持久化逐道 metadata
→ 项目运动补偿 → 处理成果与谱系保存
```

后端仍未最终完成。下一阶段应集中处理 SVD、F-K、Stolt、Kirchhoff 和 RTM 等全局重型算法的原生化、资源上限和外存/GPU 执行策略。
