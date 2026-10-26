# MyGPR 开发文档

当前开发与交接入口（2026-10-14）。安装和精确依赖版本见根目录 [README](../../README.md)，执行约束见 [AGENTS](../../AGENTS.md) 与 [CLAUDE](../../CLAUDE.md)。

- [架构](ARCHITECTURE.md)与[后端 API](../architecture/BACKEND_API_V1.md)。
- [混合项目存储](HYBRID_PROJECT_STORE_V1.md)：现行 HDF5 + SQLite 协议。
- [项目数据协议](project_data_contract.md)：现行入口与旧布局说明。
- [数据演化](PROJECT_DATA_EVOLUTION.md)与[版本策略](versioning_policy.md)。
- [测试与门禁](TESTING_STRATEGY.md)、[测试数据层级](TEST_DATA_TIERS.md)。
- [发布流程](RELEASE_PROCESS.md)：已存在的包构建路径和未落地的 exe 发布环节。
- [架构决策](adr/)与[设计规格索引](../superpowers/specs/README.md)。
- [领域术语](../../CONTEXT.md)；[用户手册](../user/README.md)。

此前索引中的 `terminology_v0.8.74.md`、`field_ui/`、`docs/legacy/research/` 不在当前树中。历史记录从 Git 查询，不再作为可直接打开的现行入口。

- [处理算法展示命名规则](ALGORITHM_NAMING.md)：正式名称、作用说明、旧名检索及兼容约束。
