# Hybrid Project Store v1

## 1. 设计目标

Hybrid Store 将大型数值数组、项目关系和外部交付成果分开管理：

- HDF5：大型雷达与处理数组；
- SQLite：测线目录、分支、谱系、导出和审计；
- JSON：项目身份、兼容配置及少量可读旁文件；
- 外部文件：PDF、CSV、GeoTIFF、VTK、PNG、ZIP 等交付成果。

整个项目不使用单一巨型 HDF5。每条测线一个容器，以降低锁冲突、备份放大和局部损坏影响。

## 2. 项目目录

```text
<project>/
├─ project.json
├─ catalog.sqlite
├─ data/lines/
│  ├─ L01.h5
│  └─ L09.h5
├─ raw/                  # 来源文件、导入清单、轨迹兼容旁文件
├─ processed/            # JSON 清单与 .artifact 指针
├─ targets/              # 当前标注兼容层
├─ spatial/              # 正式空间版本
├─ reports/              # 正式报告版本
├─ exports/              # 通用外部导出
├─ attachments/
├─ cache/staging/
├─ cache/previews/
├─ metadata/
├─ logs/
└─ backups/
```

## 3. 单测线 HDF5

```text
/raw/bscan
/raw/distance_m
/raw/time_ns
/raw/depth_m
/navigation/                 # Phase 2 规范化目标
/processing/artifacts/<id>/bscan
/processing/recipes/
/processing/branches/
/interpretation/             # Phase 2 规范化目标
/qc/                         # Phase 2 规范化目标
/provenance/
/_staging/
```

矩阵默认使用 `float32`、约 1 MiB 的二维 chunk、gzip level 2、shuffle 和 Fletcher32。写入按行块计算 SHA-256，并支持取消和进度回调。

## 4. SQLite 目录

- `catalog_meta`
- `lines`
- `processing_branches`
- `artifacts`
- `exports`
- `audit_log`
- `migration_journal`

连接为短生命周期；启用外键、30 秒 busy timeout、WAL 和 `synchronous=FULL`。项目仍坚持单写者合同。

## 5. 提交与恢复

### 原始标准化矩阵

通过临时 HDF5 文件完整写入并校验后原子替换。既有 processing、interpretation、qc 与 provenance 组会复制到新容器。来源文件继续保留，方向修正前另建备份。

### 处理产物

1. 写入 `/_staging/<uuid>`；
2. HDF5 flush；
3. 移动到 `/processing/artifacts/<artifact_id>`；
4. SQLite 登记分支和产物；
5. 写入 JSON 清单及 `.artifact` 指针。

SQLite 登记失败时删除 HDF5 组。最终清单登记失败时同时删除目录记录、恢复分支头并删除 HDF5 组。

### 完整性审计

核对：

- 项目清单测线与 SQLite `lines`；
- 每条测线 HDF5 是否存在、schema/project_id/line_id 是否匹配；
- SQLite 处理产物与 HDF5 处理组是否双向一致；
- HDF5 URI 指向的数据集是否存在；
- 标注、空间成果、报告及工作区指针是否仍可解析。

## 6. 旧项目迁移

UI：项目管理 → 存储与备份 → **迁移到 HDF5 + SQLite**。

CLI：

```bash
python -m scripts.migrate_project_storage_v3 /path/to/project
```

迁移采用 staging，完成校验后再提交；旧 `raw/processed` 文件不会删除。迁移失败时恢复原 `project.json` 并移除已经提交的新文件。

## 7. 内存语义

`HDF5ArrayProxy` 对切片读取按需打开文件，只返回请求区域。视窗预览保持有界内存。

为兼容旧算法，`np.asarray(proxy)`、`proxy * scalar` 等操作仍会物化整个矩阵。这是显式兼容路径，不代表算法已经完成分块化。Phase 2 必须将高内存算法迁移到 chunk executor。

## 8. Phase 2

- 导航/姿态/测高规范组；
- 标注关键点、中心线、可见性和不确定度规范组；
- QC 与多分辨率预览金字塔；
- 算法块执行器、halo、两遍统计和频域分区；
- 产物引用计数、垃圾回收、磁盘预算和缓存 LRU；
- raw normalization 版本化，而非仅受控替换；
- 大项目断电恢复和 Windows 真机压测。
