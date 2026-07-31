# MyGPR 后端拆分第二阶段实施报告

## 1. 本阶段目标

本阶段冻结旧 Qt 前端的结构性修改，优先建立可由 Qt、Web、CLI 或未来新前端共同调用的后端基础。范围限定为：

1. AutoTune 依赖反转闭环；
2. Processing Engine application/domain 接口；
3. 无 Qt 的任务、进度和取消机制；
4. 稳定 Backend Facade；
5. CLI 后端冒烟流程；
6. 架构、Schema、复杂度和技术债门禁闭环。

未修改旧前端布局、交互和页面架构。

## 2. 主要实现

### 2.1 Processing 后端

新增：

- `mygpr/domain/processing/models.py`
- `mygpr/application/processing/ports.py`
- `mygpr/application/processing/service.py`
- `mygpr/infrastructure/processing/legacy_adapter.py`

建立了统一的：

- 处理方法描述；
- 单方法请求和结果；
- 资源估算；
- 流水线步骤和执行结果；
- Catalog Port；
- Executor Port。

现有 `core.methods_registry` 和 `core.processing_engine` 被包裹在 infrastructure adapter 中，application 层不再直接依赖具体处理实现。

### 2.2 AutoTune 后端闭环

将以下共用能力迁入 domain：

- 标量规范化；
- 结构化错误；
- 运行时警告；
- 质量指标；
- 数据上下文；
- 参数约束策略。

新增：

- `AutoTuneConstraintPort`
- `AutoTuneDependencies`
- `AutoTuneService`
- `auto_tune_method_with_dependencies`
- `auto_select_method_group_with_dependencies`

候选生成、粗筛、细化、评分和试验执行均通过注入的 catalog、executor 和 constraint policy 工作。`core/auto_tune.py` 等旧模块保留为兼容门面。

### 2.3 UI 无关任务系统

新增 `mygpr/application/jobs`：

- `CancellationToken` / `CancellationTokenSource`；
- `ExecutionContext`；
- `JobStatus`、`JobEvent`、`JobSnapshot`；
- `InMemoryJobRunner`。

支持：

- 线程池执行；
- 协作式取消；
- 分阶段进度映射；
- 警告和成果事件；
- 失败状态标准化；
- 有序事件订阅；
- 事件历史轮询；
- 状态快照。

后端不依赖 `QThread`、`pyqtSignal`、`QProgressDialog` 或 Qt 事件循环。

### 2.4 Backend Facade

新增：

```python
from mygpr.interfaces.backend import MyGPRBackend
backend = MyGPRBackend.create_default()
```

公开：

- `backend.processing`
- `backend.autotune`
- `backend.jobs`

并提供单方法、流水线和自动选参异步提交入口。

### 2.5 CLI 冒烟流程

新增根目录 `backend_smoke.py` 和 `mygpr/interfaces/cli/backend_smoke.py`。冒烟流程在不导入 PyQt 的情况下完成：

1. 合成二维 B-scan；
2. Dewow；
3. Dewow + AGC 流水线；
4. Dewow 自动选参；
5. 输出形状、有限值和推荐参数摘要。

## 3. 门禁修复

- Schema Catalog：补齐原有 7 个未归属 Schema，并登记处理流水线和任务事件 Schema；
- 架构策略：覆盖 `mygpr/interfaces`，限制 application/domain 对旧 core 和具体 infrastructure 的依赖；
- 技术债：保持宽泛异常、静默异常和 `sys.path` 修改数量不增长；
- 复杂度：未增加超过既有预算的新高复杂度函数；
- 源码包清单和测试策略检查通过。

## 4. 验证结果

### 4.1 门禁

- Python 编译：591 个文件通过；
- Schema Catalog：PASS，110 个已归属 Schema、106 个引用；
- Architecture Policy：PASS；
- Debt Budget：PASS；
- Complexity Budget：PASS；
- Source Package Manifest：PASS；
- Test Policy：PASS。

### 4.2 自动测试

- 新增后端测试：6 项通过；
- AutoTune 常规非 GUI 测试：198 项通过，3 项因当前环境缺少 PyWavelets 对应能力而排除；
- 第一阶段既有核心回归测试及错误/观测性测试已通过。

### 4.3 数值回归

对第一阶段源码与本阶段源码使用相同确定性输入，比较：

- `set_zero_time`
- `dewow`
- `subtracting_average_2D`
- `agcGain`
- `sec_gain`

结果中最佳参数、推荐参数、试验数量一致，最佳分数差值为 0。

## 5. 当前边界

本阶段还没有迁移：

- 项目创建/打开业务；
- SQLite Catalog 与 HDF5 Line Store 的 application ports；
- 项目事务、启动恢复和损坏修复；
- 项目绑定的测线预览与处理结果回写；
- 报告 application model 与渲染 adapter。

因此当前 Backend API 已完成“数组级处理、自动选参和任务执行”闭环，尚未完成“项目级端到端”闭环。

## 6. 下一阶段

下一阶段应按以下顺序推进：

1. Project domain models 和 identifiers；
2. ProjectRepository / LineDataRepository ports；
3. SQLite、HDF5 和 Hybrid Store adapters；
4. 项目恢复、审计和事务补偿；
5. 项目绑定的 Processing/AutoTune service；
6. Reporting ports 与 renderer；
7. 与新前端契约的最终适配层。
