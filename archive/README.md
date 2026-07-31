# archive/history_v0.9.x

2026-07-30 历史残留清理归档。这里的内容均为**历史轮次产物**，不再参与
当前代码、测试或治理链（CI / schema_catalog / Makefile 均无引用）。
需要时可直接取回；确认无用后可整体删除。

- `release_notes/` — v0.9.26~v0.9.36 及 v1.4~v2.0 各轮 RELEASE/PATCH 笔记
- `implementation_reports/` — 后端 Phase2~8 实施报告、验证报告、瘦身报告
- `delivery_manifests/` — 后端交付快照佐证（文件清单与 SHA256）
- `docs_developer/` — field_ui 轮次开发日志、v0.8/v0.9 文档索引与术语表
- `docs_audit/` — v0.8.x 时代审计产物（含清理审计 CSV、findings）
- `scripts_oneoff/` — 一次性验证/迁移/基准脚本（保持原相对子路径）

注意：`scripts/` 下的 `run_mutation_contract.py`、`generate_schema_catalog.py`、
`audit_test_redundancy.py` 因被 `config/schema_catalog.json` 注册而**保留在原位**。
