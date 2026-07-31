# MyGPR 测试体系工业化与冗余清理：第十阶段实施报告

## 1. 阶段结论

本阶段没有推倒既有测试。原有回归资产被保留并重新分类，在此基础上新增工业验收层、需求与风险追踪、关键模块分支覆盖门禁、实测营山黄金基线、性质测试、性能资源边界测试、变异测试和外部验收证据门禁。

当前可确认：便携式、无 GUI 的工业自动化测试层已完成并通过。Windows 破坏性 I/O、真实 CUDA 和硬件在环属于目标设备外部验收，代码仓内已提供证据格式、校验器和商业发布强制门禁，但本环境没有伪造这些实机结果。

## 2. 新增工业测试结构

新增 `tests/industrial/`：

- `acceptance/`：后端端到端工作流；
- `reliability/`：任务回收、事务恢复和保留策略；
- `scientific_validation/`：营山实测数据与钻孔黄金基线；
- `property/`：输入契约和边界性质测试；
- `performance/`：分块执行资源边界；
- `static_contract/`：测试治理与冻结兼容接口。

每个工业测试必须声明 requirement、risk、level 和 domain marker；缺失时测试收集阶段直接失败。

## 3. 需求、风险与证据追踪

新增：

- `config/requirements_catalog.json`：22 项需求、14 项风险；
- `config/test_traceability.json`：自动化测试与外部证据映射；
- `scripts/check_test_traceability.py`：P0/P1 未覆盖直接失败；
- JSON/CSV 可审计追踪报告。

本阶段结果：22 项需求、14 项风险、16 个显式测试链接全部通过治理检查。

## 4. 覆盖率与测试有效性

新增 `config/coverage_policy.json` 和覆盖门禁。关键模块实测结果：

| 模块 | 行覆盖率 | 分支覆盖率 |
|---|---:|---:|
| `mygpr/application/jobs/runner.py` | 76.3% | 59.5% |
| `mygpr/application/processing/validation.py` | 98.8% | 96.9% |
| `mygpr/interfaces/backend.py` | 83.1% | 50.0% |
| `core/storage_primitives.py` | 76.6% | 61.5% |
| `core/hybrid_transaction_journal.py` | 85.3% | 57.1% |
| `core/project_integrity.py` | 66.6% | 53.4% |

新增变异契约对中央参数校验制造三种缺陷：接受未知参数、禁用最小值校验、禁用跨字段顺序校验。三种变异均被测试杀死。

## 5. 营山实测科学基线

从外部实测包读取 3、6、7、9、L1、X1 六条测线，建立：

- 六个原始文件 SHA-256；
- 每条测线 16 道、每道 501 点的确定性仓库内子集；
- 固定预处理流水线和量化结果指纹；
- ZK07、ZK08、ZK09 钻孔基覆界面真值；
- 基线解释结果与钻孔最大绝对误差 0.3 m，小于 1 m 验收阈值。

完整外部数据校验结果：六个文件全部通过 SHA-256、形状、数据头及确定性抽样验证。

## 6. 冗余清理

执行 AST 级测试体审计：

- 删除五个完全重复的兼容模块存在性测试；
- 用一个集中式冻结兼容契约替代；
- 精确重复测试体组降为 0；
- `misc` 测试模块从 70 降为 0；
- 94 个源码/配置读取型模块明确归入 `static_contract`，不再冒充行为验收证据；
- 删除重复生成的 mutation 与营山证据副本；
- 工业证据清单不再包含临时 `.coverage` 数据库或自引用清单。

清理后库存：290 个测试模块、1,517 个静态测试函数、0 个精确重复组、0 个未分类模块。

## 7. 自动化执行结果

便携式工业套件：49 项通过，0 失败，0 跳过。该套件包含工业测试、Backend API、任务生命周期、项目存储恢复、备份安全、中央参数校验和供应链测试。

额外后端/存储回归：71 项通过。

治理门禁通过：

- Python 编译：631 个文件；
- 架构、Schema、复杂度、技术债 ratchet；
- 测试策略、分类、追踪、冗余；
- Backend API v1；
- 依赖契约、源码包清单；
- 静态发布测试库存：290 模块、1,517 测试函数。

## 8. 发布门禁调整

`check_release_test_baseline.py` 现在分两层：

- 无可选 GUI/硬件依赖时，使用 AST 静态库存防止测试被静默删除；
- 正式 release gate 强制 `--runtime-collect`，要求安装完整发布依赖后 pytest 全量收集成功。

这避免后端开发节点因缺少 PyQt/PyWavelets产生伪失败，同时不降低正式发布 CI 的要求。

## 9. 外部验收状态

商业发布仍必须提供以下真实证据：

- Windows 破坏性 I/O 与大数据长稳；
- CUDA CPU/GPU 数值一致性、显存和取消；
- RTK/IMU/雷达或等效硬件在环。

`check_external_acceptance_evidence.py --require-all` 在证据缺失时失败。当前三个外部证据均未在本环境执行，因此不能宣称已完成实机工业认证。
