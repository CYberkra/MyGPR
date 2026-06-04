# GX-UI-015/016 Theme Consistency, B-scan Workspace Polish, and Workbench Cleanup

## Scope

This pass focuses on UI finish only:

- deep/light theme consistency audit and polish,
- main B-scan workspace readability and empty-state cleanup,
- legacy Workbench active-code cleanup.

## Theme consistency changes

- Added additional theme-aware QSS for toolbar buttons, scrollbars, progress bars and empty-state labels.
- Added object-name selectors for the main empty-state card to avoid dark-theme inherited label backgrounds.
- Improved dark-theme readability for input widgets, scrollbars, tabs, cards and toolbar controls.

## Main B-scan workspace changes

- Kept the productized B-scan card structure.
- Improved empty-state label styling so dark mode does not render black text bands over the onboarding panel.
- Kept Matplotlib plotting behavior unchanged.

## Legacy Workbench cleanup

- The active MyGPR UI no longer creates or displays the old Workbench page.
- The heavy `ui/gui_workbench.py` implementation has been replaced by a small retired compatibility shim so old imports/tests fail gracefully instead of loading the retired UI.
- `app_qt.py` keeps only no-op compatibility methods for historical calls.
- `scripts/preflight_check.py` no longer depends on Workbench execution.

## Not changed

- No processing algorithms changed.
- No AutoTune production scoring changed.
- No gprMax models or campaign files changed.
- No Evidence files changed.
- No PyVista/PyVistaQt dependency was introduced.

## Claim boundary

This is a UI polish and maintenance cleanup pass. It does not change scientific processing behavior or research claims.
