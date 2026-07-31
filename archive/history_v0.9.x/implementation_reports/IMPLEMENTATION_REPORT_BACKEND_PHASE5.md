# MyGPR 后端拆分第五阶段实施报告

## 1. 阶段目标

本阶段聚焦“处理引擎原生化与大数据文件后端执行”，不对旧 Qt 前端做结构性调整。目标是减少新后端对 `core.processing_engine` 的依赖，为项目级大矩阵处理建立可取消、可预估资源、可追溯的文件后端执行路径。

## 2. 已完成内容

### 2.1 原生处理算法与组合路由

新增 `mygpr.infrastructure.processing.algorithms` 和原生处理适配器。以下 11 个常用方法已由新后端原生执行，不再经过旧处理引擎：

1. `compensatingGain`
2. `dewow`
3. `set_zero_time`
4. `agcGain`
5. `sec_gain`
6. `subtracting_average_2D`
7. `running_average_2D`
8. `sliding_avg`
9. `frequency_filter_1d`
10. `trace_median_filter`
11. `trace_savgol_filter`

建立 `NativeProcessingCatalog`、`NativeProcessingExecutor`、`CompositeProcessingCatalog` 和 `CompositeProcessingExecutor`。已迁移方法优先走原生实现，尚未迁移的重型或全局算法继续走受控 legacy fallback，因此 Backend API v1 未发生破坏性变化。

AutoTune 的历史函数式入口也已切换到同一套“原生优先、旧实现兜底”的组合执行器。

### 2.2 文件后端分块流水线

新增 `FileBackedBlockPipelineExecutor`：

- 项目数据通过 `iter_dataset_blocks()` 分块读取；
- 中间结果使用轮转 float32 NumPy memmap；
- 按算法特征选择 sample-row 或 trace-column 分块；
- 每个块执行取消检查和进度上报；
- 每一步生成算法版本、参数、输出形状、输出类型和 SHA-256；
- 最终结果直接交给现有 HDF5 成果写入链路；
- 任务取消或异常时临时工作目录自动清理；
- memmap 按块重新映射和释放，避免长期保持整个文件的驻留页。

分块执行只在数学等价条件成立时启用。下列特殊参数组合会自动退回完整矩阵路径：

- AGC 的 `_low_energy_guard=true`；
- 中值滤波的 `preserve_mean=true`；
- `sliding_avg` 使用非 trace 轴；
- 背景均值抑制使用自定义时间范围或边缘 taper。

### 2.3 项目级处理闭环

`ProjectProcessingService` 已支持：

- 项目测线资源预估；
- 可分块流水线不调用完整 `read_dataset()`；
- 文件后端执行后直接保存处理成果；
- 在成果参数和谱系中记录 `execution_mode=file_backed_blocks`；
- 保存输入哈希、最终输出哈希以及逐步骤哈希；
- 对不支持分块的全局算法，在完整数据读取前执行内存预检。

### 2.4 资源保护

新增 `LocalProcessingResourcePolicy`：

- 默认限制任务估算内存不超过主机当前可用内存的 75%；
- 可通过 `MYGPR_PROCESSING_MEMORY_FRACTION` 调整；
- 单任务可通过 `ExecutionContext.metadata["max_memory_bytes"]` 设置更严格上限；
- 资源不足时在完整数据读取和算法执行前失败；
- 文件后端执行前检查临时磁盘容量并保留安全余量。

### 2.5 元数据安全

项目持久化适配器的 JSON 安全转换已改为递归处理，嵌套的谱系、参数和运行元数据不再被整体字符串化。

## 3. 验证结果

### 3.1 数值一致性

11 个原生方法逐一与重构前的 verified legacy engine 对照：

- 大多数方法逐元素完全一致；
- SEC、频率滤波和 Savitzky-Golay 的最大允许绝对误差为 `1e-6`～`3e-6`；
- 六步骤文件后端流水线与内存流水线结果在 `4e-6` 绝对误差内一致。

### 3.2 自动测试

- 第五阶段新增测试：16 项通过；
- 含上述新增测试在内的后端、项目、存储、备份、报告、AutoTune 和处理综合回归集：70 项通过；
- 其中 1 项小波测试因当前环境缺少 PyWavelets 自动跳过；
- 项目级测试确认分块路径不会调用完整测线读取接口；
- 取消测试确认临时工作区无残留；
- 资源保护测试确认全局算法在完整数据读取前被拒绝。

### 3.3 1 GiB 文件后端压力测试

使用形状 `(4096, 65536)`、float32、总大小恰好 1 GiB 的合成矩阵执行 `compensatingGain`：

- 处理完成；
- 输出 SHA-256：`a18dc386868ef4d0ed0b9c443eadc9b5a5164566e2dc96dcd8d38f103ac3acda`；
- 临时工作区执行后为空；
- 128 MiB 目标块配置下耗时约 26.9 秒；
- 最大常驻内存约 774 MiB（792600 KiB），低于输入矩阵总大小；
- 未出现 swap。

该测试证明当前文件后端可在不将完整 1 GiB 矩阵一次性物化为普通 ndarray 的情况下完成处理。不同操作系统、文件系统和算法的峰值内存仍需在目标 Windows 构建机上重复测量。

### 3.4 门禁与冒烟

- Python 编译检查：596 个文件通过；
- Schema Catalog：通过；
- 架构依赖、循环依赖、迁移所有权和增长限制：通过；
- 复杂度预算：通过；
- 技术债 release ratchet：通过；
- 源码包清单：通过；
- 测试策略：通过；
- 项目格式兼容性：通过；
- 无 GUI Backend 冒烟：通过；
- 无 GUI 项目创建、处理、成果写回及完整性审计冒烟：通过；
- 冒烟过程中未加载 Qt。

## 4. 当前限制

以下算法仍属于 legacy/global 路径，尚未完成文件后端原生化：

- SVD / RPCA；
- F-K；
- Stolt；
- Kirchhoff；
- RTM；
- 全波形反演及其他高内存全局算法。

这些算法当前具备执行前资源预估和内存保护，但仍可能完整载入数据。后续应针对算法数学特征分别设计外存、重叠块、频域切片、GPU 显存预算或明确的数据规模上限，不能机械套用本阶段的局部分块方式。

当前验证环境未安装 PyQt6、PyWavelets、ruff 和 mypy，因此未执行 GUI、小波、ruff 和完整 mypy 验证。旧 Qt 前端未进行结构性修改。

## 5. 阶段结论

第五阶段已完成常用处理方法的原生后端迁移，并打通“项目 HDF5 分块读取 → 文件后端处理 → 谱系与哈希记录 → HDF5 成果提交”的闭环。后端尚未全部完成；下一阶段建议处理原始数据导入、GNSS/RTK/IMU 同步与运动补偿域，随后单独攻克 Kirchhoff、F-K、SVD 等重型算法。
