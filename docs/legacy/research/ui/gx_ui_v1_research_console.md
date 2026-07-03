# GX-UI-V1 Research Console

## Purpose

GX-UI-V1 adds a read-only research console inside the existing PyQt6 / qfluentwidgets MyGPR main interface. It surfaces current GX-008 gprMax paired Evidence, standardized metrics, AT-BG background-suppression diagnostics, claim boundaries, and GX-008 model draft inspection without turning the app into Workflow Studio.

## Current UI Integration Point

The primary entry point is the main AutoTune recommendation page (`AutoTuneTuningPage`): click `研究验证` in the header, which opens `高级设置与审计明细 -> 研究验证`. The console remains read-only and hosts four pages:

- `仿真与验证 Dashboard`
- `背景抑制 AutoTune`
- `Evidence Viewer`
- `gprMax 模型编辑器 v0`

The legacy compatibility `AutoTunePage` still exposes the older segmented `研究验证` page for tests and backward-compatible callbacks, but the normal user path no longer depends on opening the hidden legacy page. The main window stack, right-side B-scan canvas, and legacy Workbench fallback remain unchanged.

## Pages Implemented

### Simulation And Validation Dashboard

Shows GX-008 scene progress cards, a scene status table, selected artifact details, manifest/report/metrics text tabs, warnings, and preview thumbnails when curated figures exist.

### Background Suppression AutoTune Viewer

Shows AT-BG selected trials, candidate method tables, trial rows, method-rank summary, and a visible claim-boundary banner. It reads diagnostic artifacts only; it does not run AutoTune.

### Evidence Viewer

Lists configured gprMax and AT-BG artifacts, then displays manifest, report, metrics/table summaries, and claim boundaries in read-only text panes.

### gprMax Model Editor v0

This page is a read-only protected model inspector. It loads GX-008 scene drafts, validates the current pair contract subset, shows raw/background text, materials, ROI, generated GPU wrapper command, and warning summaries.

## Data Sources

Default configuration lives in:

- `config/research_dashboard_defaults.json`

Read-only data models live in:

- `core/research_dashboard.py`
- `core/gprmax_model_inspector.py`

The dashboard searches configured Evidence roots and curated artifact paths. The model inspector reads local GX-008 model drafts under `experiments/gprmax/GX-008/models/`.

## Missing Path Behavior

Missing Evidence roots, missing artifacts, missing preview files, and malformed JSON are converted into warnings. They do not block GUI startup.

## Claim Boundary Rules

The UI must keep these constraints visible:

- synthetic paired diagnostic only
- background suppression diagnostic only
- not full AutoTune
- not production scoring
- not field validation
- not no-prior truth correctness
- not AutoTune superiority evidence
- not paper-candidate benchmark

## gprMax Model Editor v0 Boundary

Phase 0 is read-only protected mode:

- no direct edits to `raw_with_target.in`
- no direct edits to `background_only.in`
- no direct edits to `materials.txt`
- no writes to MyGPR-Evidence
- no gprMax execution from UI
- no destructive editing

Additional V0.8.61 completion:

- local model draft scene discovery now scans `experiments/gprmax/GX-008/models/` instead of limiting the inspector to the initial six scenes;
- the geometry tab reports parsed target directives from `raw_with_target.in` and confirms that `background_only.in` has no target directives when pairable;
- file-open actions use the native opener on Windows, macOS, and Linux, with a browser fallback.

Future phases:

1. Duplicate scene as draft.
2. Controlled form editing of draft scenes.
3. Pair contract check before save.
4. Dry-run integration.
5. Optional controlled run command execution.

Archived Evidence scenes must never be edited directly.

## Reference Mockup Notes

The user-provided mockup informed the layout direction: status cards at the top, scene/artifact list on the left, dense table plus details in the center/right, and lower report/manifest/metrics/warning tabs. The implementation stays restrained and operational rather than decorative.

## Non-goals

- No Workflow Studio.
- No node canvas.
- No gprMax execution from UI.
- No Evidence writing from UI.
- No production AutoTune scoring changes.
- No motion-compensation changes.
- No AutoTune superiority claim.

## Next UI Tasks

- `GX-UI-V1-SCREENSHOT-QA`: run the full app and capture visual QA screenshots.
- `UI-BG-AUTOTUNE-PANEL-SPEC`: specify a focused background suppression AutoTune panel.
- `GX-UI-MODEL-DRAFT-EDIT-SPEC`: design controlled draft editing without touching archived scenes.
