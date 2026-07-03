# GX-UI-014 Dark Theme Polish and Workbench Retirement

## Scope

This pass improves dark-theme visual consistency and removes the legacy Workbench fallback from the active MyGPR shell.

## Visual changes

- `ui/theme.py` now provides a theme-aware polish stylesheet.
- The polish layer no longer forces light cards, light inputs, or light preview panels when dark mode is active.
- Dark mode now has dedicated card, input, tab, chip, table, empty-state, and preview-canvas colors.
- `ui/autotune_tuning_page.py` now applies local AutoTune styling based on the current light/dark theme.
- AutoTune header, status chips, preview cards, group boxes, tabs, tables, and form controls now remain readable in dark mode.

## Workbench retirement

- The visible “进入旧工作台（Legacy）” button was removed from the main application shell.
- `WorkbenchPage` is no longer imported or instantiated during normal startup.
- Legacy workbench run/save methods remain as safe no-op compatibility guards so historical signal references do not crash the application.
- Existing `ui/gui_workbench.py` is intentionally left in the source package as historical code, but it is no longer part of the active main UI.

## Not changed

- No processing algorithm was modified.
- No AutoTune production scoring logic was modified.
- No AutoTune production execution path was enabled or changed.
- No gprMax scene, campaign, or Evidence artifact was modified.
- No PyVista/PyVistaQt dependency was added.

## Claim boundary

This is a UI/theme and shell cleanup pass only. It does not change scientific results, processing outputs, model physics, Evidence contents, or research claims.
