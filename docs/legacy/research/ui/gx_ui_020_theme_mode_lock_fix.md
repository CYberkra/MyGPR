# GX-UI-020 Theme Mode Lock Fix

This patch fixes the mode mismatch where the AutoTune page could render dark in light mode, or light in dark mode.

## Root cause

The previous effective-theme helper inspected the Qt widget palette even when `theme_manager` explicitly reported LIGHT. Some qfluentwidgets child palettes can be dark while the application is light, so only the AutoTune page switched to dark colors.

## Fix

- `ui.theme.is_dark_ui()` now treats explicit LIGHT/DARK theme-manager values as authoritative.
- Palette probing is only used for automatic/system/unknown theme states.
- `AutoTuneTuningPage.refresh_theme()` was added so app-level theme switches force local page styles to refresh.
- `app_qt.py` calls the AutoTune refresh after applying the global app theme.

## Boundary

No processing logic, AutoTune scoring, gprMax, Evidence, or model files were changed.
