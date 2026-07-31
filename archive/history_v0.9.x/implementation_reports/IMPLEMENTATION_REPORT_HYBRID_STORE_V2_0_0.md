# MyGPR V2.0.0 Hybrid Project Store Phase 1 实施报告

> 产品版本仍为 **v0.9.28 beta**。V2.0.0 是本轮项目存储架构迭代号，不代表产品正式版已达到 2.0。

## 1. 本轮目标

将原有以 `.npy`、`.npz`、JSON 和目录扫描为主的项目数据管理，升级为适合工业级长期项目的混合存储架构：

- `project.json`：项目身份、格式版本和少量配置；
- `catalog.sqlite`：测线、处理分支、处理产物、导出、审计和迁移记录；
- `data/lines/<line_id>.h5`：每条测线独立的大数组容器；
- `exports/`：PDF、ZIP、CSV、GeoTIFF、VTK、PNG 等外部交付成果；
- `cache/`：允许删除并重建的缓存数据。

本轮没有采用单一巨型 `project.h5`，避免单文件损坏、增量备份成本、多任务写锁冲突和单测线归档困难。

## 2. 最终目录结构

```text
<ProjectRoot>/
├─ project.json
├─ catalog.sqlite
├─ data/
│  └─ lines/
│     ├─ L01.h5
│     ├─ L03.h5
│     └─ L09.h5
├─ attachments/
├─ exports/
├─ cache/
├─ logs/
├─ backups/
└─ project_trash/
```

新建项目默认格式为：

```text
mygpr.field_project.v3
```

## 3. 已实施能力

### 3.1 存储抽象层

新增统一 Hybrid Store 后端，正式业务代码不再需要直接推断 HDF5 内部路径。新后端支持：

- 判断项目使用旧目录存储还是 Hybrid Store；
- 生成和解析 `h5://...::/dataset/path` URI；
- 按测线打开 HDF5 容器；
- 创建、查询和删除处理产物；
- 管理处理分支和分支头；
- 登记空间成果、报告和交付包；
- 执行完整性检查和回滚。

### 3.2 单测线 HDF5

每条测线使用独立 HDF5 文件。当前规范组包括：

```text
/raw
/navigation
/processing
/interpretation
/qc
/provenance
/_staging
```

本阶段正式写入 HDF5 的核心内容包括：

- 标准化原始 B-scan；
- 时间轴、距离轴和深度轴；
- 处理结果矩阵；
- 处理参数、算法标识、父版本、分支和哈希属性。

数组使用分块、压缩、shuffle 和 Fletcher32 校验。显示和普通切片通过 HDF5 代理按需读取，打开项目或切换测线不会默认加载完整 B-scan。

### 3.3 SQLite 项目目录

`catalog.sqlite` 管理：

- 测线目录；
- 处理分支；
- 处理产物与父子谱系；
- 分支头；
- 外部导出成果；
- 审计日志；
- 存储迁移日志；
- HDF5 URI、shape、dtype 和数据哈希。

数据库启用 WAL，并在备份前执行 checkpoint。项目采用单写者策略，不宣称支持多个写任务同时修改同一项目。

### 3.4 处理分支与谱系

处理结果不再仅依赖时间戳目录。每个处理版本具有：

- 唯一 artifact ID；
- 所属测线；
- 所属分支；
- 父 artifact ID；
- 方法和参数；
- 数据集 URI；
- shape、dtype 和 hash；
- 创建时间和状态。

测线处理页已接入真实分支选择器，并支持从当前分支头创建新分支。保存处理结果时，分支关系同时写入 HDF5 和 SQLite。

### 3.5 事务与故障恢复

处理产物采用“暂存—写入—登记—提交”的可恢复流程：

- HDF5 写入失败时，不登记 SQLite；
- SQLite 最终登记失败时，删除对应 HDF5 组；
- 分支头更新失败时恢复旧分支头；
- 完整性审计可以识别 HDF5 中未登记的处理组；
- 完整性审计可以识别 SQLite 指向但 HDF5 已缺失的产物。

需要说明：HDF5 和 SQLite 无法组成真正的跨文件 ACID 事务。本实现通过 staging、提交顺序、补偿回滚和完整性审计降低半提交风险。

### 3.6 备份、删除与导出

- 项目备份前执行 SQLite WAL checkpoint；
- 备份中排除临时 WAL/SHM 文件；
- 删除测线时，HDF5 移入项目回收站，SQLite 目录执行级联清理；
- 空间成果、报告 PDF、报告交付 ZIP 和校验文件登记到统一导出目录；
- 外部成果继续保持可直接交付的文件形态，不强行塞入 HDF5。

### 3.7 旧项目迁移

提供两个迁移入口：

- 项目管理界面的“迁移项目数据结构”；
- CLI：

```bash
python -m scripts.migrate_project_storage_v3 /path/to/project
```

迁移行为：

1. 识别旧项目；
2. 备份旧 `project.json`；
3. 逐测线转换到独立 HDF5；
4. 建立 SQLite 目录和处理谱系；
5. 执行 shape、dtype、数据集和目录一致性验证；
6. 验证完成后提交 v3 manifest；
7. 迁移失败时恢复原 manifest，并移除本轮已提交的新存储文件；
8. 旧数据文件默认保留，不做静默删除。

## 4. 数据不可变性边界

本轮明确区分两种数据：

- **原始来源文件**：作为工程证据保持不可变；
- **HDF5 标准化原始矩阵**：作为软件内部规范化表示，在存在备份的方向修正、转置修正等流程中允许受控替换。

处理产物按版本保存，不覆盖已提交版本。下一阶段将进一步把标准化原始矩阵改为显式版本，而不是受控替换。

## 5. 内存边界

已完成：

- HDF5 分块存储；
- B-scan 按切片读取；
- UI 打开和视区预览不默认展开整幅矩阵；
- 旧代码通过 NumPy 兼容代理继续运行。

尚未完成：

- 所有处理算法真正块执行；
- 全局统计的两遍计算；
- SVD、RPCA、FK、Stolt 等全局算法的分区或外存实现；
- halo 重叠区融合；
- 中间数据磁盘预算和自动回收。

因此，本阶段显著降低了“打开、浏览和普通切片”造成的内存峰值，但旧算法执行全数组 NumPy 运算时，兼容层仍可能物化完整矩阵。本阶段不宣称已经彻底消除全部处理过程的内存溢出风险。

## 6. 主要代码变更

新增核心模块：

- `core/storage_uri.py`
- `core/hdf5_array_proxy.py`
- `core/hdf5_line_container.py`
- `core/project_catalog.py`
- `core/project_storage_backend.py`
- `core/project_storage_migration.py`

新增迁移工具：

- `scripts/migrate_project_storage_v3.py`

新增测试：

- `tests/test_hybrid_project_storage_v3.py`
- `tests/test_hybrid_storage_ui_v200.py`

新增开发文档：

- `docs/developer/HYBRID_PROJECT_STORE_V1.md`
- `RELEASE_NOTES_hybrid_project_store_v2_0_0.md`

同时修改项目模型、测线存储、处理产物、完整性检查、备份、删除、空间成果、报告交付和处理页分支交互。

## 7. 验证结果

本轮按正式桌面业务范围分组执行，累计 **189 项相关测试通过**：

- 核心存储、项目、处理及新 UI：75 项；
- 空间成果与报告：21 项；
- 工作台、正式 UI 和旧兼容桥：38 项；
- 静态、发布、Schema 与环境门禁：55 项。

其他验证：

- 全量 Python 语法编译通过；
- 技术债预算门禁通过，未通过提高基线规避问题；
- 项目格式兼容性门禁通过；
- 严格环境检查通过；
- 正式 Qt 工作台离屏启动烟测通过；
- ZIP 打包后将再次运行 Hybrid Store 核心测试和格式检查。

上述数字不是仓库全部科研 Runner 的全量测试声明。耗时较长的离线科研验证任务不属于本轮桌面存储发布门禁。

## 8. Phase 2 建议范围

下一阶段建议按以下顺序继续：

1. 将 RTK、IMU、测高、同步质量数组规范化写入 `/navigation`；
2. 将正式界面标注、关键点、置信度、no-interface 区段写入 `/interpretation`；
3. 将 QC 曲线和指标数组写入 `/qc`；
4. 建立统一块处理执行器，支持 block、halo、两遍统计和取消；
5. 优先改造背景抑制、增益、滤波和逐道算法；
6. 为 SVD、RPCA、FK、Stolt 设计外存或分区算法；
7. 建立多分辨率 B-scan 预览金字塔；
8. 增加项目磁盘配额、中间产物保留策略和 LRU 回收；
9. 将标准化原始矩阵改为显式版本；
10. 增加崩溃恢复向导和孤儿数据修复工具。

## 9. 阶段结论

Phase 1 已完成可运行的 Hybrid Project Store 最小闭环：

> **小型项目清单 + SQLite 项目目录 + 每测线独立 HDF5 + 外部交付成果 + 显式处理谱系 + 旧项目迁移。**

该结构已经解决了原有目录扫描、处理版本关系不清、单个处理结果旁文件过多和大数组浏览默认物化的问题，并为后续真正的块处理、缓存回收和完整数据版本化提供了稳定边界。
