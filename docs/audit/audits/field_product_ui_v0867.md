# MyGPR V0.8.67 Field Product UI Audit

## Decision

The default MyGPR application should present itself as field exploration / positioning software, not as a research validation platform.  Research features remain available for developers but are hidden from normal users.

## Product-mode gate

Implemented `core.product_mode`:

- Default: field mode.
- Enable research UI with `MYGPR_ENABLE_RESEARCH_UI=1`.
- Alternative enablement: `MYGPR_PRODUCT_MODE=research` or `dev`.

## Default Workbench surface

The normal Workbench now exposes five field workspaces:

```text
项目管理 -> 测线处理 -> 目标定位 -> 空间成果 -> 成果报告
```

The gprMax `仿真验证` workspace is not created or shown in default field mode.  It is instantiated only when research mode is explicitly enabled.

## AutoTune surface

Default mode hides:

- Primary `研究验证` button.
- Advanced `研究验证` tab.
- Legacy segmented `真值验证` / `研究验证` pages.

The underlying pages are retained for developer/research mode and existing non-UI tooling.

## Terminology cleanup

Field-facing terms now prefer:

- 项目管理 instead of 数据管理.
- 测线处理 instead of 处理实验室.
- 目标定位 / 目标标注 instead of 解释工作台 / 解释层.
- 空间成果 instead of 空间综合.
- 成果报告 / 交付成果 instead of 成果交付 / 成果包.
- 交付文件 instead of 证据.

## Validation

```text
python scripts/preflight_check.py
python scripts/check_version_consistency.py --expected 0.8.67
python -m compileall -q app_qt.py cli_batch.py core ui scripts tests PythonModule
```

Targeted regression:

```text
pytest tests/test_product_mode.py \
       tests/test_workbench_ui.py \
       tests/test_simulation_validation_page_ui.py \
       tests/test_autotune_recipe_ui.py::test_autotune_page_hides_research_console_in_field_mode \
       tests/test_gui_presets.py::test_auto_tune_defaults_live_in_auto_tune_page \
       tests/test_gui_presets.py::test_auto_tune_tab_hides_research_console_pages_in_field_mode \
       tests/test_version_consistency.py \
       tests/test_delivery_page_ui.py \
       tests/test_delivery_service.py \
       tests/test_spatial_synthesis_ui.py \
       tests/test_interpretation_workbench_ui.py \
       tests/test_processing_lab_ui.py
```

Result: 34 passed, 1 skipped.  The skipped test requires a Windows CJK font not available in the Linux/offscreen sandbox.

Research-mode smoke:

```text
MYGPR_ENABLE_RESEARCH_UI=1 QT_QPA_PLATFORM=offscreen python <workbench smoke>
# research_mode_smoke_ok
```
