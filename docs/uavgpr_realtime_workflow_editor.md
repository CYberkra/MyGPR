# UAV-GPR Realtime Workflow Editor

This note records the current MyGPR workflow editor contract after the
`codex/uavgpr-workflow-refactor` branch introduced the realtime workflow page.

## User-Facing Contract

- The main control tabs are ordered as `日常处理 -> 工作流 -> 调参与实验 -> 显示与对比 -> 质量与导出`.
- The old standalone workbench UI is removed. Long-chain method composition now lives in the `工作流` tab.
- Workflow steps can be reordered, enabled/disabled, and hidden.
- Hidden steps are preserved in the template but excluded from execution.
- Numeric parameters render as spinbox plus slider when the parameter range is suitable.
- User templates default to realtime preview. System templates keep the explicit `运行全部` path.
- Realtime preview updates the current preview result but must not create formal history entries.
- `保存结果` is the boundary that commits a realtime workflow result into formal history.

## Default UAV-GPR Chain

The high-quality default workflow currently starts with:

```text
set_zero_time
-> dc_shift
-> dewow
-> frequency_filter_1d
-> motion_compensation_v2
-> subtracting_average_2D
-> wavelet_svd
-> manual_velocity_model
-> geometry_depth_context
-> sec_gain
```

`kirchhoff_migration` remains present in the template as a hidden migration
step, so users can explicitly enable it when they are ready for heavier
processing.

## Gain Stage Rule

The gain stage includes:

- `sec_gain`
- `energy_decay_gain`
- `compensatingGain`
- `agcGain`

`agcGain` is intentionally available in the main gain stage, but the UI warns
that it is display-enhancement oriented and not strict amplitude-preserving.

## Minimal New Methods

- `dc_shift`: per-trace or global DC offset removal with mean/median estimators.
- `manual_velocity_model`: attaches a constant velocity model from velocity or
  dielectric constant.
- `geometry_depth_context`: validates velocity, trace spacing, time window, and
  optional AGL metadata, then passes context downstream.

`geometry_depth_context` is not a mature topographic correction algorithm. It is
only the first workflow-safe context checkpoint for later geometry/depth work.

## Regression Checks

Use these focused checks when changing workflow UI behavior:

```powershell
pytest tests\test_workflow_ui_contracts.py -q
pytest tests\test_workflow_refactor_methods.py -q
pytest tests\test_workflow_profile_alignment.py -q
pytest tests\test_shared_state_sync.py tests\test_daily_processing_smoke.py -q
```

Before packaging or merging larger workflow changes:

```powershell
pytest -q
python scripts\preflight_check.py
```

## Current Known Risk

The realtime workflow editor is now regression-covered at the contract level,
but it still needs human GUI review for interaction polish: drag behavior,
parameter density, scrolling, and how quickly B-scan previews feel responsive on
large real UAV-GPR lines.
