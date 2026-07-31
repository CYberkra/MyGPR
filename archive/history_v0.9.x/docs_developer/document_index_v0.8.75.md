# MyGPR v0.8.75 文档分区索引

v0.8.75 已将原先散落在 `docs/` 顶层的主要文档实际归入以下分区。`docs/` 顶层只保留四个主入口目录：`user`、`developer`、`audit`、`legacy`。

## user

- `docs/user/README.md`：用户文档入口占位。后续正式用户手册应放在这里。

## developer

- `docs/developer/project_data_contract.md`：项目目录与数据协议。
- `docs/developer/terminology_v0.8.74.md`：当前术语和 UI 文案约束。
- `docs/developer/versioning_policy.md`：版本策略。
- `docs/developer/launcher_environment_fix_2026_06_09.md`：启动器环境选择修复说明。
- `docs/developer/field_ui/`：Field Workbench 迭代记录。
- `docs/developer/adr/`：架构决策记录。
- `docs/developer/config/`：配置相关说明。

## audit

- `docs/audit/algorithm_compatibility_v0.8.73.md`
- `docs/audit/algorithm_compatibility_v0.8.74.md`
- `docs/audit/code_health_findings_v0.8.74.md`
- `docs/audit/audits/`：历史审计结果。
- `docs/audit/evidence/`：交付证据和报告包说明。

## legacy

- `docs/legacy/research/`：AutoTune、gprMax、运动补偿、早期工作台、superpowers plans 等历史研究材料。

## 后续读取规则

新会话或新开发者默认优先读取：

1. 根目录 `README.md`
2. 根目录 `DEV_HANDOFF.md`
3. 根目录 `CURRENT_STATE.md`
4. 根目录 `CHANGELOG.md`
5. 根目录 `TODO.md`
6. `docs/developer/project_data_contract.md`
7. `docs/developer/terminology_v0.8.74.md`
8. `docs/audit/algorithm_compatibility_v0.8.73.md`

`docs/legacy/*` 仅在需要追溯历史方案、研究验证或旧 UI 决策时读取，不作为当前 Field Workbench 主线开发依据。
