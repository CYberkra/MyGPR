# MyGPR 工业化架构治理与 AutoTune 垂直切片迁移（第一阶段）

日期：2026-07-22  
基线版本：MyGPR 0.9.28 / Hybrid Project Store V2.0.0 Phase 1

## 1. 本阶段目标

本阶段不修改算法公式、处理参数含义、项目存储格式或 GUI 操作流程，重点完成：

1. 将 `core/auto_tune.py` 从超大业务模块改造为兼容门面；
2. 建立 `domain -> application` 的 AutoTune 垂直切片；
3. 强化架构门禁，检查层间依赖、循环依赖、迁移例外、冻结模块和新代码规模；
4. 将技术债目标从“已自动达成”改为真实可量化的下降目标；
5. 保持历史导入路径和数值行为兼容。

## 2. 代码结构变化

### 2.1 历史入口

`core/auto_tune.py` 从约 3414 行缩减为 16 行兼容门面。历史代码仍可继续使用：

```python
from core.auto_tune import auto_tune_method
```

新代码应改用：

```python
from mygpr.application.autotune.use_case import auto_tune_method
```

生产调用点已迁移，包括：

- `core/auto_tune_pipeline.py`
- `core/auto_tune_comparison.py`
- `core/processing_session.py`
- `ui/worker_threads.py`
- `ui/field_panels/processing_page.py`
- `compatibility/legacy_app_qt.py`

### 2.2 新增领域层

```text
mygpr/domain/autotune/
├── models.py       # 试验评分、上下文、推荐档位等领域模型
└── selection.py    # Pareto 前沿与三档推荐规则
```

领域层不依赖 Qt、UI、文件系统、数据库或 `core`。

### 2.3 新增应用层

```text
mygpr/application/autotune/
├── use_case.py              # 稳定应用入口
├── ports.py                 # 表现层调用边界类型
├── legacy_engine.py         # 当前编排器，已由 3414 行降至约 370 行
├── candidate_planner.py     # 粗筛候选规划
├── candidate_generators.py  # 各方法族候选生成
├── context.py               # ROI 与数据特征上下文
├── evaluation.py            # 试验执行、评分和种子选择
├── refinement.py            # 细化搜索
├── scoring.py               # 方法族评分器
├── diagnostics.py           # 失败记录、稳定性和参数域诊断
└── utils.py                 # 公共纯函数
```

所有新模块均低于 1000 行，最大模块约 721 行。

## 3. 架构门禁增强

`config/architecture_policy.toml` 升级为 `mygpr.architecture_policy.v2`，`scripts/check_architecture.py` 现在检查：

- `domain / application / infrastructure / presentation` 层间允许依赖；
- 新架构包内循环依赖；
- AutoTune 迁移例外的负责人、删除版本和原因；
- 新代码模块、类、函数规模；
- 新代码静默异常处理；
- 历史大模块净增长；
- 兼容门面的生命周期登记；
- `sys.path` 修改和入口重型导入。

AutoTune 仍直接依赖部分 `core` 处理和质量服务。该依赖被登记为显式迁移例外：

- Owner：`autotune`
- Remove after：`1.0.0`
- 下一阶段：定义 processing/metrics ports，并由 infrastructure adapters 实现。

以下模块已冻结，禁止继续净增长：

- `compatibility/legacy_app_qt.py`
- `ui/autotune_tuning_page.py`
- `ui/field_panels/processing_page.py`

## 4. 技术债变化

| 指标 | 原发布基线 | 本阶段基线 | 变化 |
|---|---:|---:|---:|
| 超过 1000 行模块 | 23 | 22 | -1 |
| 宽泛异常处理 | 524 | 519 | -5 |
| 静默异常处理 | 123 | 119 | -4 |
| 超过 1000 行类 | 12 | 12 | 0 |
| 超过 100 行函数 | 124 | 124 | 0 |

新的主动下降目标为：

- 超过 1000 行模块：20
- 超过 1000 行类：10
- 超过 100 行函数：110
- 宽泛异常处理：500
- 静默异常处理：100
- `sys.path` 修改：0

发布基线继续执行“只降不升”规则；下降目标不再通过放宽目标值伪装为完成。

## 5. 兼容性和验证

### 5.1 数值等价验证

使用固定随机种子构造 128 × 64 B-scan，对原始包和改造包分别执行：

- 零时校正自动选参；
- 去低频漂移自动选参；
- 背景抑制自动选参；
- SEC 增益自动选参；
- F-K 滤波自动选参。

以下输出逐字段一致：

- `best_params`
- `best_score`
- 推荐档位
- 候选数量
- Pareto 数量
- 风险等级
- 采用/复核建议

### 5.2 自动测试

已通过：

- Python 编译检查：588 个 Python 文件；
- 架构门禁 V2；
- 技术债发布基线门禁；
- 源码包清单检查；
- AutoTune、候选约束、pipeline、comparison、维护收口和架构策略相关测试：65 项通过。

当前 Linux 审查环境未安装 PyWavelets，因此 3 个小波方法测试被排除。原始未改造包在同一环境下也以相同原因失败，判定不是本次重构回归。

## 6. 下一阶段

优先顺序：

1. 为 AutoTune 定义 `ProcessingRuntimePort`、`QualityMetricsPort`；
2. 在 `mygpr/infrastructure/processing/` 实现对现有 `core` 服务的适配器；
3. 删除 `mygpr/application/autotune` 对 `core` 的迁移例外；
4. 拆分 `ui/autotune_tuning_page.py` 为 Page、Controller、ViewModel、Presenter；
5. 将页面调用统一切换到 `AutoTuneUseCase`；
6. 清理 AutoTune 范围内的宽泛异常和超长函数。

## 7. 发布判断

本阶段属于内部架构增量，不改变用户数据格式和算法结果。可合入内部 Pilot 分支，但在完成 Windows GUI 回归和完整依赖环境测试前，不应单独标记为商业正式发布版本。
