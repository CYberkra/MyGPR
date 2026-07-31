# MyGPR 后端拆分第八阶段实施报告

## 1. 阶段目标

本阶段完成处理后端的原生化收口，重点解决 Kirchhoff 迁移、运动补偿、剩余公开算法、GPU 执行策略和 RTM 能力缺口。旧 Qt 前端未做结构性修改。

本阶段的核心发布契约是：

> `core.methods_registry` 中所有公开处理方法，均必须由 `NativeProcessingExecutor` 支持；默认 Backend API 不得因公开方法缺少原生实现而静默回退到历史处理引擎。

## 2. 已完成内容

### 2.1 公开处理目录完成 34/34 原生覆盖

当前处理目录统计：

- 历史注册方法总数：35；
- `visibility=public` 的公开方法：34；
- 原生算法注册数：36；
- 公开方法原生覆盖：**34/34**；
- 未覆盖公开方法：**0**。

新增 `tests/test_native_public_catalog_closure.py` 作为发布契约。后续只要新增公开算法却未提供原生后端，测试将直接失败。

`LegacyProcessingExecutor` 仍保留在 composition root 中，用于旧入口兼容、非公开历史能力及迁移期兜底；但当前公开处理目录的正常 Backend API 执行均由原生执行器优先接管。

### 2.2 Kirchhoff 迁移原生化

新增：

```text
mygpr/infrastructure/processing/algorithms/kirchhoff/
├── cpu.py
├── gpu.py
├── shared.py
└── __init__.py
```

原生 Kirchhoff 实现具备：

- UI 无关的 CPU 数值内核；
- 旅行时计算、叠加成像、网格设置和后处理拆分；
- 飞高时间校正、深度校正和地形校正；
- 输入网格与输出网格标定；
- 进度和取消检查；
- 资源估算；
- 算法版本 `native-kirchhoff-2.0`；
- 输出参数、网格、修正策略、后端选择和运行警告进入处理谱系。

`PythonModule/kirchhoff_migration.py` 已缩减为兼容门面，不再承载正式算法实现。

固定 CPU 回归样例的底层数值哈希为：

```text
b9a5013a9fa455cefa0333acc5a87725f6a126e0236d1fdeab09ec4bfbf6af88
```

项目级 Kirchhoff 冒烟成果哈希为：

```text
74b70f0a14f623acad1f7b965bc0cf3c93e659d95b500796de2a2f1c463effe7
```

### 2.3 GPU 选择、显存预算和回退合同

新增 `gpu_policy.py`，GPU 能力按需探测，不在后端导入时强制加载 CuPy。

策略包括：

- `backend=cpu`：严格使用 CPU；
- `backend=gpu`：显式 GPU 请求，设备或显存不可用时直接失败，不静默降级；
- `backend=auto`：仅在设备可用且满足显存预算时选择 GPU，否则记录原因并回退 CPU；
- 查询设备空闲/总显存；
- 预留安全显存；
- 执行前实际小额分配探测；
- GPU 能力缓存及测试清理接口；
- GPU OOM/设备不可用错误标准化。

当前环境没有 CuPy/CUDA，因此只验证了 CPU 路径、能力探测、严格请求和自动回退合同；尚未完成真实 GPU 数值一致性验收。

### 2.4 实验性 RTM 后端基线

新增方法：

```text
rtm_migration
implementation_version = experimental-rtm-1.0
```

其明确建模合同为：

```text
zero_offset_exploding_reflector_scalar_2d
```

当前实现是二维标量、零偏移、爆炸反射面假设下的反时延拓基线，包含：

- 二阶有限差分反向传播；
- 按 CFL 条件自动增加时间子步；
- sponge 吸收边界；
- 逐时刻数据注入；
- 网格元素和 cell-update 硬预算；
- 进度、取消和资源预估；
- 深度轴及成像头信息输出；
- 明确的 `experimental_rtm_baseline` 运行警告。

它**不是**完整的电磁波方程 shot-gather RTM，也不应作为正式工程 RTM 结论依据。项目级冒烟成果哈希为：

```text
0064ce210d8c88fe395a4bd8cc48fb0a30e3dc326b4268ef36cca5a8960f5357
```

### 2.5 六类运动补偿算法原生化

新增原生运动算法包并注册 `native-motion-2.0`：

- `trajectory_smoothing`；
- `motion_compensation_speed`；
- `motion_compensation_height`；
- `motion_compensation_attitude`；
- `motion_compensation_vibration`；
- `motion_compensation_v2`。

能力包括：

- 轨迹平滑及 metadata 更新；
- 按累计距离等距重采样；
- 飞高幅值归一化和走时平移；
- 姿态/APC/足迹修正；
- 周期振动及旋翼干扰抑制；
- V2 集成流程；
- 逐道 metadata 长度、有限性和必需字段校验；
- 进度、取消、结果谱系和形状变化记录。

历史 `PythonModule` 运动补偿文件已改为兼容门面，保留旧导入和测试 seam。

### 2.6 剩余十一类处理方法原生化

新增 `algorithms/extended/`，迁入：

- `time_cut`；
- `trace_qc`；
- `equidistant_trace_resample`；
- `energy_decay_gain`；
- `amplitude_scale`；
- `median_background_2D`；
- `wavelet_2d`；
- `wavelet_svd`；
- `hilbert_envelope`；
- `ccbs`；
- `time_to_depth`。

这些方法统一使用 `native-extended-1.0`，并纳入参数校验、资源估算、元数据更新和 Backend API。

其中：

- 等距重采样在后端仍可显式调用，但继续从旧现场工作台方法列表中隐藏，避免旧 UI 无意改变道数；
- 小波算法延迟导入 PyWavelets，后端在未安装 PyWavelets 时仍可正常导入和执行其他算法；
- CCBS 重构为输入校验、归一化互相关和加权背景扣除三个小型内核，保持原运算次序和回归结果；
- `time_to_depth` 输出统一为 float32，并更新深度轴头信息。

### 2.7 项目级谱系增强

对 loaded/global 算法，项目处理谱系新增或强化：

- `implementation_version`；
- 实际执行后端；
- 求解器/有效秩/网格等摘要；
- 输入参数；
- 输入和输出形状、dtype；
- 输出 SHA-256；
- 大型数组只保存结构摘要，避免把矩阵写入 manifest。

### 2.8 无 GUI 迁移成像冒烟

新增：

```bash
python -m mygpr.interfaces.cli.migration_imaging_smoke
```

流程为：

```text
创建项目 → 保存合成测线 → 原生 Kirchhoff → 实验性 RTM
→ HDF5 成果提交 → 谱系与 SHA-256 校验
```

执行确认 `qt_loaded=false`。

## 3. 验证结果

### 3.1 综合非 GUI 回归

本阶段组合回归集结果：

- **428 项通过**；
- **3 项取消收集**，均为依赖 PyWavelets 的 AutoTune 小波候选测试；
- 运行时间约 14 秒；
- 仅出现当前 Linux 环境缺少 CJK 字体导致的 Matplotlib glyph warning，不影响数值和产物。

覆盖范围包括：

- 原生基础、扩展、全局、迁移和运动算法；
- GPU 后端选择合同；
- AutoTune；
- 项目级分块处理；
- SQLite/HDF5 存储恢复；
- 备份恢复安全；
- 导入、Acquisition 和项目状态；
- 处理谱系、兼容门面和公开目录闭环。

小波算法的实际数值测试未在本环境执行，原因是未安装 PyWavelets，而不是测试通过后跳过失败。

### 3.2 编译与治理门禁

最终检查结果：

- Python 源码编译：**603 个文件通过**；
- 架构层级、循环依赖、迁移所有权和新代码规模：通过；
- Schema Catalog：通过，113 个登记 schema、108 个引用；
- 项目格式兼容性：通过；
- 复杂度预算：通过；
- 技术债 release ratchet：通过；
- 源码包清单：通过；
- 测试策略：通过，276 个测试模块；
- 版本一致性：通过，`0.9.28`；
- Release hygiene：通过，无 `__pycache__`、`.pyc` 和 pytest 缓存残留。

技术债现状相对发布基线继续下降：

- 超过 1000 行的模块：22 → 21；
- 超过 100 行的函数：124 → 123；
- 宽泛和静默异常数量未恶化。

### 3.3 五个无 Qt 冒烟流程

全部通过：

```text
backend_smoke
project_smoke
global_processing_smoke
acquisition_smoke
migration_imaging_smoke
```

覆盖：

```text
Backend API → AutoTune → 项目存储 → 文件后端全局处理
→ 数据导入与传感器同步 → 运动补偿 → Kirchhoff → RTM
```

所有流程均确认未加载 Qt。

## 4. 当前限制与未完成验收

1. RTM 是明确标记的实验性标量基线，不是完整电磁 RTM，不能作为正式工程能力宣称。
2. 当前 Linux 环境没有 PyWavelets，因此小波与小波-SVD的实际数值路径未执行。
3. 当前环境没有 CuPy/CUDA，未完成 Kirchhoff CPU/GPU 数值容差、显存 OOM 和多型号 GPU 验证。
4. 当前环境没有 PyQt6，未执行旧 GUI 回归；本阶段未修改旧前端结构。
5. 尚未在目标 Windows 机器完成 10 GB/50 GB 项目、长时间运行、磁盘写满、强杀进程、USB 断连、杀毒软件占用、备份/恢复中断和安装升级测试。
6. Acquisition、Persistence 和 Reporting 仍存在已登记的 legacy adapter；它们已被 Backend API 隔离，但还不是全部原生实现。
7. `LegacyProcessingExecutor` 仍保留为兼容/兜底基础设施。虽然 34 个公开处理方法均已原生覆盖，旧脚本或未来新增的未迁移私有方法仍可能使用该 seam。
8. ruff、mypy、Windows 打包、代码签名、SBOM 和第三方许可证闭环未在本环境完成。

## 5. 阶段结论

第八阶段完成了**公开处理后端的原生化闭环**。对新前端而言，处理、AutoTune、项目、导入、同步、运动补偿和迁移成像已经可以通过 UI 无关的 Backend API 调用；公开处理目录不再依赖历史引擎才能运行。

当前后端可定位为：

> **发布候选架构 / 工程试点增强版**。

不能仅凭本阶段 Linux 结果宣称已经完成商业工业级认证。正式商业发布仍需在目标 Windows/CUDA 环境完成破坏性故障、超大数据、GPU、安装升级、供应链、许可证和签名验收。

后续建议按以下顺序执行：

1. 目标 Windows/CUDA 发布候选验收；
2. 将前端 Agent 成果接入 `MyGPRBackend API v1`；
3. 处理 API/DTO 差异和端到端项目回归；
4. 逐步移除 Acquisition、Persistence、Reporting 的 legacy adapter；
5. RTM 若要对外正式宣称，单独立项实现并验证电磁 shot-gather RTM。
