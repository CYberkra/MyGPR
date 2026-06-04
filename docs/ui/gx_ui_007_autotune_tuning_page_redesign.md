#!/usr/bin/env markdown
# GX-UI-007 AutoTune Tuning Page Redesign

## Task
- ID: `GX-UI-007-AUTOTUNE-TUNING-PAGE-REDESIGN`
- Scope: UI-only refactor of `调参与实验` entry.
- Boundary: no gprMax run, no GX-008/GX-009 model changes, no Evidence repo changes.

## What changed
- Added new focused page class: `ui/autotune_tuning_page.py`
  - class: `AutoTuneTuningPage`
  - title: `AutoTune 调参`
  - structure:
    1. Top AutoTune Session Header
    2. Left Parameter Panel
    3. Center Preview Workspace
    4. Right Result/Risk Inspector
    5. Bottom Drawer
- Navigation binding update:
  - `app_qt.py` now routes “调参与实验” tab to `AutoTuneTuningPage`.

## Legacy compatibility strategy
- `ui/gui_auto_tune_page.py` kept unchanged as legacy page.
- `ui/research_console_page.py` kept unchanged as legacy dependency.
- `AutoTuneTuningPage` embeds a hidden legacy `AutoTunePage` instance and forwards legacy attributes/methods via `__getattr__`.
  - Reason: keep existing `app_qt.py` signal wiring and state update code working without backend changes.

## Non-goals in this task
- No production AutoTune scoring logic changes.
- No 3D viewer embedding or launcher.
- No PyVista/PyVistaQt reintroduction.
- No broad Research Lab productization.

## Claim boundary
- This redesign is a UI focus adjustment for AutoTune tuning workflow clarity.
- It is not an AutoTune algorithm upgrade and not a performance claim.
