# GX-UI-019 Code Audit and Performance Optimization

## Scope

This pass audits and optimizes the current UI-focused MyGPR package after the dark-theme consistency work. It is intentionally limited to UI responsiveness and maintainability.

## Files reviewed

- `app_qt.py`
- `ui/theme.py`
- `ui/autotune_tuning_page.py`
- `ui/gui_workbench.py`
- `scripts/preflight_check.py`

## Key findings

### 1. Repeated stylesheet construction and application

`app_qt.py` rebuilt and re-applied the full application stylesheet during theme refresh even when the effective stylesheet did not change. This is relatively expensive in Qt because it can trigger repolish of a large widget tree.

Resolution:

- `ui/theme.py` now caches the generated polish stylesheet by effective light/dark theme.
- `app_qt.py` now skips `setStyleSheet(...)` when the stylesheet is unchanged.

### 2. Theme-manager label can be stale

The previous dark-theme fix already detected that the qfluentwidgets theme label can report `LIGHT` while the actual Qt palette is dark. This behavior is retained and now formalized through `get_effective_theme_key(...)`.

Resolution:

- `ui/theme.py` exposes `get_effective_theme_key(...)`.
- Local style code uses the effective Qt palette key instead of trusting the theme label alone.

### 3. Repeated dynamic-property repolish on chips

Status chips in `AutoTuneTuningPage` and the main B-scan header were repolished on every refresh, even when text and tone were unchanged. This can cause unnecessary UI work during ROI spinbox changes, candidate toggles, data sync, and theme refresh.

Resolution:

- Added `set_dynamic_property(...)` and `repolish(...)` helpers in `ui/theme.py`.
- Chip update methods now skip work when text/tone are unchanged.

### 4. AutoTune control initialization caused avoidable signal churn

`_sync_controls_from_state()` set many controls during initialization without blocking signals. This could trigger repeated refresh calls before the page reached a stable initial state.

Resolution:

- `_sync_controls_from_state()` now blocks signals while synchronizing controls, then restores them.

### 5. Candidate row calculation repeated across panels

Candidate rows were recalculated for ranking, recommendation text, and trial table refresh. This was small but unnecessary and grows with future candidate expansion.

Resolution:

- `AutoTuneTuningPage` now caches candidate rows by candidate-method/rank configuration.
- Candidate cache invalidates when recommendation-affecting controls change.

### 6. Legacy workbench is retired but preserved as compatibility shim

`ui/gui_workbench.py` remains a lightweight retired shim. No further action was taken in this optimization pass because it already avoids main UI instantiation.

## Changes made

- `ui/theme.py`
  - Added cached stylesheet generation.
  - Added dynamic-property helper functions.
  - Added effective theme key helper.

- `app_qt.py`
  - Passes `widget=self` to theme stylesheet resolution.
  - Skips repeated global `setStyleSheet(...)` when unchanged.
  - Skips repeated direct workspace restyling when effective theme did not change.
  - Avoids unnecessary chip repolish when B-scan status chips are unchanged.

- `ui/autotune_tuning_page.py`
  - Added candidate-row cache.
  - Added signal blocking during state-to-control sync.
  - Avoids repeated chip repolish.
  - Avoids repeated local stylesheet rebuild/application when effective theme is unchanged.
  - Removed duplicate QSS color declaration in the status chip rule.

## What was intentionally not changed

- Processing algorithms.
- Production AutoTune scoring.
- AutoTune execution backend.
- gprMax model files and campaign assets.
- Evidence repository/content.
- PyVista / 3D viewer behavior.
- Legacy page deletion.

## Validation

Static syntax check passed:

```text
python -m py_compile app_qt.py ui/theme.py ui/autotune_tuning_page.py ui/gui_workbench.py scripts/preflight_check.py
```

The current execution sandbox does not include PyQt6, so full GUI/preflight runtime validation should be run locally.

## Known limitations

- `ui/gui_quality_log.py` still contains optional PyVista dialog code. It is not on the normal startup path, but a future dependency-boundary audit should decide whether to move it behind a plugin boundary.
- Older legacy pages still contain hard-coded local styles. They are not the current primary UI, but a future cleanup could migrate them to `ui/theme.py` tokens.
- This pass optimizes UI refresh overhead, not numerical processing performance.
