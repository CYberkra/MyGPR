# MyGPR V0.8.67 Patch Notes

## Field-product UI positioning pass

This release changes the default product surface from a mixed research/workbench flow to a field exploration and positioning flow.

### Default user-facing workspaces

```text
项目管理 -> 测线处理 -> 目标定位 -> 空间成果 -> 成果报告
```

### Hidden by default

The following research/development surfaces no longer appear in the normal UI:

- Workbench `仿真验证` workspace
- AutoTune primary `研究验证` button
- AutoTune advanced `研究验证` tab
- Legacy AutoTune `真值验证` / `研究验证` segmented pages

They can still be enabled for development with:

```bat
set MYGPR_ENABLE_RESEARCH_UI=1
start_mygpr.bat
```

or:

```bat
set MYGPR_PRODUCT_MODE=research
start_mygpr.bat
```

### Engineering terminology updates

- `数据管理` -> `项目管理`
- `处理实验室` -> `测线处理`
- `解释工作台` -> `目标定位`
- `空间综合` -> `空间成果`
- `成果交付` -> `成果报告`
- Resource tree `解释层` -> `目标标注`
- Resource tree `成果包` -> `交付成果`
- Bottom drawer `证据` -> `交付文件`
- Classic AutoTune compatibility page uses recommendation/comparison wording instead of experiment-first wording.

### Preserved behavior

- Processing algorithms unchanged.
- AutoTune scoring and candidate generation unchanged.
- gprMax campaign tooling remains in the source tree and is available in research mode.
- Project file formats and delivery output schema are unchanged.
