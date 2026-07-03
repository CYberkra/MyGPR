#!/usr/bin/env markdown
# GX-UI-005 Rollback Embedded 3D Viewer Integration Audit

## Task
- Task ID: `GX-UI-005-ROLLBACK-EMBEDDED-3D-VIEWER-INTEGRATION`
- Scope: MyGPR source repo only
- Strategy: manual cleanup (no history rewrite)

## Finding Classification

### A. Keep
- `experiments/gprmax/GX-008/**` and `experiments/gprmax/GX-009/**` model/campaign/audit files.
- Existing conversion scripts and evidence/report workflow code.
- Existing georeference export button and backend georeference payload processing.

### B. Remove or Revert
- Main UI segmented control item `georef3d` ("三维预览") from primary chart navigation.
- Embedded viewer dependency path that preferred `PyVista/pyvistaqt` in expanded preview dialog.
- Dev dependency declaration that implied `pyvista/pyvistaqt` are required baseline packages.

### C. Defer
- Future external standalone 3D viewer integration design.
- Optional docs describing future external launch, with explicit "not integrated yet" boundary.

## Applied Changes
- Updated `ui/gui_quality_log.py`
  - Removed main segmented entry for embedded 3D preview in normal workflow navigation.
  - Removed visual-route mapping for `georef3d`.
  - Changed expanded 3D dialog path to always use Matplotlib fallback view; no PyVista launch path in normal app flow.
- Updated `requirements-dev.txt`
  - Removed `pyvista` and `pyvistaqt` from baseline dev dependency list.

## Responsibility Boundary (post-cleanup)
MyGPR owns:
- gprMax scene/campaign management
- conversion and processing chain
- evidence/report workflow

Standalone 3D viewer:
- remains external and optional
- install path not fixed
- not a startup/runtime requirement for MyGPR

## Claim Boundary
- This change rolls back embedded/experimental viewer integration from MyGPR primary path.
- This is not deletion of GX-008/GX-009 model history.
- This does not alter gprMax model physics or evidence data outputs.
