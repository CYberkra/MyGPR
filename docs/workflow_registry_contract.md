# Workflow Registry Contract

本文档记录当前 MyGPR Workflow Studio 的工作流注册契约。它用于避免 UI、CLI、自动调参、报告脚本各自维护一套流程真相。

## 当前状态

当前实现采用“非侵入式 registry facade”：

- 核心文件：`core/workflow_registry.py`
- 旧数据仍保留在 `core/workflow_data.py` 与 `core/preset_profiles.py`
- 新 registry 负责提供类型化读取、输出端口推荐、端口标签和一致性校验

这样做的原因是当前分支已经有大量 UI 和测试依赖旧常量。直接搬迁会制造导入循环和大范围回归风险，因此本阶段先把契约稳定下来，再逐步把旧常量迁入单一源。

## 固定契约

### MyGPR 标准流程

`mygpr_standard` 必须保持为原 MyGPR 五步链：

```text
set_zero_time
→ dewow
→ subtracting_average_2D
→ sec_gain
→ svd_subspace
```

这套流程服务于“用户熟悉的 MyGPR 经典处理链”，不是 UAV-GPR 全量科研链。它必须在 quick preset 与 recommended profile 中保持同一顺序。

### 高质量 UAV-GPR 流程

`high_quality_uav_gpr` 是面向项目真实 UAV-GPR SFCW CSV 的默认完整链路。它可以包含 DC 去偏、频带控制、运动补偿、速度模型和几何-深度上下文，但必须遵守：

- 缺 RTK/IMU/AGL 时不伪造传感器输入。
- 运动补偿节点应输出 warning 或跳过风险，而不是静默执行假补偿。
- 实测 SFCW CSV 默认频带来自 `core.data_context`，不能在 UI 或脚本中各自写死。

### 输出端口推荐

输出端口拖到空白处时，推荐菜单从 `core.workflow_registry` 读取：

- `bscan`: 查看此步 B-scan
- `compare`: 前后对比
- `qc`: QC 指标
- `spectrum`: 频谱 / 能量分布
- `evidence`: 导出此步结果

算法下一步推荐同样从 registry 读取。画布可以展示这些推荐，但不应把推荐逻辑硬编码在 UI 私有方法里。

### 执行语义

当前执行模式固定为：

```text
WorkflowMethod.order 决定实际运行顺序
canvas_links 用于可视化、Preview、Evidence 和一致性检查
```

也就是说，画布连线暂时不是 DAG executor。Validate 必须持续提示这个事实，避免用户误以为连接图已经完全控制执行顺序。

## 测试

注册契约由 `tests/test_workflow_registry_contract.py` 覆盖，至少检查：

- registry 没有 error 级别一致性问题。
- `mygpr_standard` 保持五步顺序。
- 增益阶段候选算法包含 SEC、AGC、energy decay、compensating gain。
- 输出效果 metadata 完整。
- velocity / preview / evidence 等端口标签有基础区分。

后续如果迁移 registry 的内部数据源，应先保证这些测试不变。
