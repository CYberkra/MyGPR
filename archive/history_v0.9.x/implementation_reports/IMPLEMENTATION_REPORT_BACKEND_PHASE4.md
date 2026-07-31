# MyGPR 后端拆分第四阶段实施报告

## 1. 本阶段范围

本阶段继续冻结旧 Qt 前端，集中强化混合项目存储的崩溃一致性、深度校验和大数据读取边界。核心目标是将第三阶段的“可审计、可补偿”进一步提升为“可在下次可写打开时自动恢复”。

本阶段未修改旧前端页面、布局或交互架构，Backend API 版本继续保持 `1.0`，新增能力均为向后兼容扩展。

## 2. 已完成内容

### 2.1 文件事务启动恢复

`core/storage_primitives.py` 新增 `recover_file_transactions()`，并由 `ProjectRepository.open_session()` 在首次可写打开时自动执行：

- `active` 事务根据 `before/` 快照自动回滚；
- `committed` / `rolled_back` 残留事务执行清理；
- 路径必须位于项目根目录，拒绝绝对路径和 `..` 逃逸；
- 恢复失败时拒绝继续以可写模式打开，保留日志和备份供诊断；
- 同进程重入会话不会恢复正在执行的活事务，避免误回滚并发中的合法写入。

### 2.2 SQLite + HDF5 跨存储恢复日志

新增：

- `core/hybrid_transaction_journal.py`
- Schema：`mygpr.hybrid_artifact_transaction.v1`

处理成果保存现在使用持久化写前日志记录：

1. 写入提交意图；
2. HDF5 staging 写入并移动到正式 artifact 路径；
3. 记录 HDF5 已提交状态；
4. 提交 SQLite Catalog；
5. 标记完成并删除日志。

若进程在任一步骤终止，下次首次可写打开会自动判断：

- HDF5 存在、SQLite 缺失：从 HDF5 manifest 重建 Catalog，执行 roll-forward；
- SQLite 存在、HDF5 缺失：删除无效 Catalog 记录并恢复分支头，执行 rollback；
- 两者均存在：确认提交并清理日志；
- 两者均不存在：清理未提交事务。

所有恢复动作写入 Catalog `audit_log`。

### 2.3 HDF5 落盘语义强化

`core/hdf5_line_container.py` 完成：

- 新建容器和原始矩阵替换统一使用临时文件、文件 `fsync`、原子替换和父目录 `fsync`；
- 处理 artifact 提交、删除后对 HDF5 文件执行 `fsync`；
- 保留原有 chunk、gzip、shuffle、Fletcher32 与 staging group 机制；
- 原始数据替换失败时保留上一份有效容器。

### 2.4 项目锁 V2

项目锁升级为 `mygpr.project_lock.v2`：

- 保留 PID、主机名和 token；
- 新增 Linux boot ID；
- 新增进程启动标记，降低 PID 复用造成的陈旧锁误判；
- 兼容旧 v1 锁文件；
- 对同进程重入会话显式标记，避免触发启动恢复。

说明：Windows/POSIX 跨平台仍采用项目锁文件；尚未宣称支持 NAS 多主机并发写入。

### 2.5 深度哈希审计

`ProjectIntegrityAuditor.audit()` 新增 `deep_hash` 选项，Backend API 可通过：

```python
backend.projects.audit_project(project_id, deep_hash=True)
```

深度审计会按 HDF5 chunk/行块重新读取矩阵并计算规范 SHA-256，不需要整矩阵物化，覆盖：

- `/raw/bscan` 原始矩阵；
- SQLite Catalog 登记的所有 processing artifact；
- 缺失哈希、哈希不匹配和不可读取状态。

默认快速审计保持原行为，不承担全量数据读取成本。

### 2.6 分块读取 Backend API

新增：

```python
backend.projects.iter_dataset_blocks(...)
```

能力包括：

- 按指定行块读取；
- 支持样点和道范围裁剪；
- HDF5 读取过程中不调用 `HDF5ArrayProxy.__array__()`；
- 为下一阶段分块处理和外存执行器提供稳定数据通道。

该能力目前是“分块读取基础设施”，不代表所有历史算法已经完成分块执行。

### 2.7 质量治理

- Schema Catalog 新增并归属：
  - `mygpr.hybrid_artifact_transaction.v1`
  - `mygpr.project_lock.v2`
- `config/test_impact.toml` 将 HDF5、混合事务、完整性模块纳入项目存储影响规则；
- 新增 `tests/test_storage_recovery_phase4.py`，覆盖 7 个崩溃/一致性/分块场景。

## 3. 自动测试与门禁

### 3.1 新增故障测试

新增测试覆盖：

1. 中断文件事务在首次可写打开时自动回滚；
2. 同进程重入不会恢复活事务；
3. PID 复用标记可识别陈旧锁；
4. HDF5 已提交但 SQLite 缺失时自动 roll-forward；
5. SQLite 已登记但 HDF5 缺失时自动 rollback；
6. 篡改原始 HDF5 数据后深度审计报告 SHA-256 不匹配；
7. Backend 分块读取不触发全矩阵物化。

结果：`7 passed`。

### 3.2 选定后端与存储回归

后端、AutoTune、项目、混合存储、完整性、备份、报告和 sidecar 选定回归集：

- `51 passed`

### 3.3 编译和质量门禁

- Python 编译：`594` 个文件通过；
- Architecture Policy：PASS；
- Schema Catalog：PASS（112 owned schemas / 107 referenced）；
- Project Format Compatibility：PASS；
- Complexity Budget：PASS；
- Debt Budget：PASS，未增加宽泛异常、静默异常、超大模块、超大类或超长函数基线；
- Source Package Manifest：PASS；
- Test Policy：PASS（267 个测试模块、26 个组、18 条规则）；
- `backend_smoke.py --skip-autotune`：PASS；
- `backend_project_smoke.py`：PASS，未加载 Qt。

## 4. 当前工业化判断

本阶段完成后，混合存储已从“事后发现并人工处理”提升到“关键事务可自动重放或回滚”。对于单机、本地磁盘、有技术人员值守的生产作业，可靠性明显提升。

但仍不能宣称后端和存储已经全部工业化完成，主要剩余边界：

1. 大部分历史处理算法仍会整矩阵物化；
2. HDF5 + SQLite 的事务日志尚未覆盖后续 sidecar、项目清单和报告等全部多文件提交；
3. `LegacyFieldProjectRepository` 仍包裹旧 `core` 实现；
4. 尚未完成 Windows 真机断电、磁盘满、外置盘断连和 10GB/50GB 项目破坏性测试；
5. 项目锁尚未提供 NAS 多主机租约、心跳和操作系统级协同锁；
6. 当前环境缺少 PyQt6 和 PyWavelets，未执行 GUI 与 wavelet 测试。

当前存储定位可更新为：**单机本地磁盘工程生产级基础已基本形成，跨存储崩溃恢复进入可用状态；大数据处理与真实故障验证仍未闭环。**

## 5. 下一阶段建议

优先顺序：

1. 分块处理执行器、资源预算和整矩阵物化保护；
2. 原生 Persistence Repository 拆分，降低 `LegacyFieldProjectRepository` 范围；
3. 测线删除、重命名、批量导入的统一项目事务；
4. Acquisition domain/ports，统一雷达、GNSS、RTK、IMU 输入；
5. Windows 故障注入与大项目稳定性测试；
6. sidecar/manifest 与 HDF5/Catalog 的更完整事务闭环。
