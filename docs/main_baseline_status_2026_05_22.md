#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MyGPR main baseline status note (2026-05-22)."""

# Main Baseline Status (2026-05-22)

## 1. Executive Summary
- `main` is now the active MyGPR source development branch.
- Current `main` HEAD: `b70d73114dac927976e6a48a08fb14d41b76b182`.
- Old main rollback tag: `pre-research-main-2026-05-22 -> 289c843d4c7a7d208070a4882d9c558bb2dfeaf1`.
- MyGPR-Evidence remains a separate repository and is not merged into this repo.
- This baseline is a research/development baseline, not a final public release.

## 2. What Is Included in the Main Baseline
- Workflow Studio direction and UI stabilization updates are included.
- `processing_engine` / method registry modernization is included.
- Metadata-aware processing behavior is included.
- `motion_compensation_v2` is the main UAV motion compensation path.
- AutoTune framework and diagnostics are included.
- gprMax benchmark/validation tooling is included.
- No-prior QC and recommendation warning labels are included.
- Evidence/export support in source code is included.
- Workbench remains available as legacy fallback.
- `UI-WIGGLE-001` is completed on main.

## 3. What Is Not Included / Not Claimed
- No MyGPR-Evidence artifacts are stored inside this MyGPR source repo.
- MyGPR-Evidence remains separate and independently versioned.
- This is not a final public release note.
- No AutoTune universal superiority claim is made.
- No field-performance validation claim is made.
- No target-detection or underground-correctness claim is made for no-prior field data.
- No ML/DL module is included yet.
- Workbench is not removed.

## 4. Important Behavior Changes from Old Main
- Metadata-aware runtime parameter injection can produce output differences vs old main.
- Shape-changing methods (such as `time_cut` / resampling) may change output shape by design.
- `motion_compensation_v2` may resample traces based on trace metadata.
- No-prior high-risk background AutoTune results now emit advisory/manual-review labels rather than silent “safe best” wording.
- Wiggle display uses trace sampling for display only; it does not modify underlying data.

## 5. Current Validation Status
- `MR-ALG-001` passed with no P0 algorithm compatibility blocker.
- `MR-001` merge readiness and fast-forward promotion were completed.
- `UI-WIGGLE-001` local validation on main:
  - `tests/test_wiggle_ui_behavior.py`: passed
  - `python scripts/preflight_check.py`: passed
  - `tests/test_gui_presets.py`: passed
  - `tests/test_import_export_report.py`: passed
  - `tests/test_auto_tune_recommendation_labels.py`: passed
- Remote CI for `b70d73114dac927976e6a48a08fb14d41b76b182` may be pending/not confirmed in this note.

## 6. Known Remaining Risks
- **P1**
  - Real desktop interactive UI smoke is still recommended.
  - Main is now a research baseline, not the old stable baseline behavior.
  - No-prior warning labels are heuristic, not ground truth.
  - AutoTune scoring should not be claimed as globally optimal.
- **P2**
  - MyGPR-Evidence cleanup is separate follow-up work.
  - Workbench retirement is incomplete.
  - Wiggle sampling maximum trace count is fixed for now.
  - ML/DL integration remains future work.

## 7. Development Policy Going Forward
- Continue development on `main` as the active source branch.
- Use short-lived feature branches only for risky or high-impact changes.
- Keep MyGPR-Evidence separate from MyGPR source.
- For algorithm changes, require tests and Evidence/claim-boundary updates.
- For UI changes, require focused tests and/or smoke notes.
- Frozen-by-default modules:
  - `motion_compensation_v2` algorithm semantics must not change without explicit task approval.
  - AutoTune scoring must not change without explicit diagnostic evidence and task scope.
- Keep durable audit/status decisions in `docs/`, not only in chat history.

## 8. Recommended Next Tasks
- `UI-SMOKE-001B`: real desktop interactive smoke on main.
- `EV-001`: MyGPR-Evidence cleanup plan for historical DS-001 noise/large files.
- `RELEASE-002` or `HANDOFF-001`: concise agent handoff for main branch.
- `UI-WIGGLE-002` (optional): configurable max wiggle traces if needed.
- `ML-000` (later): ML/DL integration design, not immediate.
