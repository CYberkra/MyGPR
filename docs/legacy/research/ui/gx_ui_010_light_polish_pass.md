# GX-UI-010 Light Polish Pass

## Scope

This pass applies a lightweight visual polish layer to the existing MyGPR UI without changing the application structure.

The current four main tabs remain unchanged:

- 日常处理
- 调参与实验
- 显示与对比
- 质量与导出

## Changed

- Added `ui/theme.py` as a centralized lightweight QSS polish module.
- Appended the polish stylesheet in `app_qt.py` after the existing theme manager stylesheet.
- Refined the AutoTune 参数推荐 page visual style:
  - more comfortable spacing
  - softer cards and borders
  - more readable status chips
  - compact Chinese tab labels
  - single-column comparison preview cards to avoid narrow right-panel squeezing
  - cleaner table/header styling
  - clearer Chinese candidate labels

## Intentionally unchanged

- No processing algorithms changed.
- No AutoTune scoring logic changed.
- No AutoTune execution enabled.
- No gprMax execution added.
- No GX-008/GX-009 model files changed.
- No Evidence files changed.
- No PyVista/PyVistaQt dependency added.
- No embedded 3D viewer reintroduced.
- Legacy pages remain present.

## Boundary

This is a UI polish pass only. It improves visual comfort and readability while preserving the current functionality and safety boundaries.
