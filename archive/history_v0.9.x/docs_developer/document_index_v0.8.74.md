# MyGPR v0.8.74 文档索引与分区建议

## 根目录文档

- `README.md`：用户入口和运行说明。
- `CURRENT_STATE.md`：当前版本状态。
- `DEV_HANDOFF.md`：新会话 / 新开发者交接。
- `CHANGELOG.md`：版本历史。
- `TODO.md`：任务池与技术债。

## 建议分区

```text
docs/
├─ user/              # 用户使用说明
├─ developer/         # 架构、数据协议、开发交接
├─ audit/             # 审计报告、兼容性报告、代码健康扫描
└─ legacy/            # 历史研究记录、旧实验说明
```

## 当前需要保护的开发文档

- `docs/project_data_contract.md`
- `docs/algorithm_compatibility_v0.8.73.md`
- `docs/field_ui_round8_v0.8.74_engineering_hygiene.md`
- `docs/code_health_findings_v0.8.74.md`
- `docs/terminology_v0.8.74.md`

## 当前历史文档处理建议

AutoTune、gprMax、motion compensation 相关文档仍有技术价值，不应删除。但它们不应干扰当前 MyGPR 现场工作台主流程，后续可逐步移动到 `docs/legacy/` 或 `docs/audit/`。
