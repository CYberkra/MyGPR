# UI-SMOKE-001 Manual UI Smoke Report for UI-STAB-001

## Environment
- Repo: `D:\CDUT-UavGPR-Controller\MyGPR`
- Branch: `codex/research-gprmax-autotune`
- Commit tested: `47145197732479e12cebe271231acd08ba994938`
- Platform: Windows + Qt offscreen smoke capture
- Dataset used: `C:\Users\17844\Desktop\02_Preprocessed_Standard\2025-09_营山\Line9origin(36).csv`

## Scope and Method
- This run is a manual-smoke-oriented validation with scripted UI interaction and screenshot capture.
- No processing algorithm semantics were changed.
- No AutoTune/gprMax runs were executed.

## Window Sizes Tested
- Narrow: `620x960`
- Medium: `1000x800`
- Wide: `1400x900`

Screenshots are stored under `docs/ui_smoke_001_screenshots/`:
- `narrow_basic.png`, `narrow_auto_tune.png`, `narrow_advanced.png`, `narrow_quality.png`, `narrow_workbench.png`
- `medium_basic.png`, `medium_auto_tune.png`, `medium_advanced.png`, `medium_quality.png`, `medium_workbench.png`
- `wide_basic.png`, `wide_auto_tune.png`, `wide_advanced.png`, `wide_quality.png`, `wide_workbench.png`

## Scenario Results
1. Window size / layout smoke: `partial_pass`
- All target pages can be reached and rendered at narrow/medium/wide sizes.
- Workbench page can be opened and captured.
- Remaining risk: offscreen capture cannot fully replace live pointer-driven overlap/scroll usability checks.

2. Display/Comparison smoke: `pass`
- Theme toggle is accessible from `显示与对比`.
- Wiggle display is accessible from `显示与对比`.
- Theme switch and subsequent B-scan rendering did not crash.

3. Daily processing advanced parameter smoke: `pass`
- `显示高级参数` toggle exists in `日常处理`.
- Expanding reveals motion-compensation-v2 advanced fields including:
  - `apc_offset_x_m`, `apc_offset_y_m`, `apc_offset_z_m`
  - `resample_spacing_m`
- Expanding/collapsing did not force default mutation behavior in this smoke.

4. B-scan interaction smoke: `partial_pass`
- Raw B-scan renders.
- View interaction path keeps history unchanged in the smoke checks.
- ROI drag path works without crash.
- Remaining risk: true subjective smoothness and “rollback feel” still need live interactive verification on desktop GPU/display.

5. 3D preview smoke: `partial_pass`
- 3D preview panel and axes render.
- Title shows `三维地理参考预览`.
- Remaining risk: this dataset/path did not include a full interactive camera judgment in offscreen mode; live check still required for camera framing perception.

6. Processing lineage smoke: `pass`
- Raw state shows `处理链路: Raw`.
- After one method apply path, lineage updates (`Raw -> set_zero_time` in this run).
- Tooltip includes chain details.
- Undo and reset restore lineage to `Raw`.
- View interaction does not append processing lineage.

7. No-prior guard regression smoke: `partial_pass`
- Main guard methods exist in runtime.
- Workbench guard callback presence could not be reliably asserted via this offscreen script-level probe.
- Required follow-up: live action-trigger check in app session for blocked paths and diagnostics payload.

8. Workbench legacy smoke: `partial_pass`
- Entry label is legacy (`进入旧工作台（Legacy）`).
- Workbench still opens.
- Remaining risk: full template/single-method blocking behavior should be verified in a live interactive run with dialog handling.

## Acceptance Decision
- UI-STAB-001 smoke status: `partial_accept`.
- Reason:
  - Functional and structural checks passed for major requested items.
  - Remaining gaps are primarily live-interaction perception checks (stutter feel, visual clipping under actual desktop event loop, guard dialog path behavior).

## Remaining UI Issues / Risks
- Need one live desktop interactive sweep to fully close:
  - narrow layout readability and accidental scrollbar behavior,
  - B-scan drag/zoom smoothness perception,
  - 3D camera framing perception with real mouse operations,
  - no-prior guard dialogs and blocked-action logging in real click paths.

## Recommended Next Task
- `UI-SMOKE-001B`: live desktop interactive smoke capture with operator checklist + annotated screenshots/GIF for:
  - B-scan interaction smoothness,
  - 3D camera framing with mouse controls,
  - no-prior guard blocked dialogs in main + Workbench paths.

