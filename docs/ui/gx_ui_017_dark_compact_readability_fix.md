# GX-UI-017 Dark Compact Readability Fix

## Purpose

This patch fixes the usability issues observed after the dark-theme/workbench cleanup pass:

- AutoTune header was too wide for the current right-side work area.
- AutoTune tabs were crowded and partially clipped.
- The B-scan empty-state card could remain visually light while the right-side panel was dark.
- Matplotlib toolbar styling needed a stronger local dark/light override.

## Changes

- AutoTune page header changed to a compact stacked layout.
- Header action buttons use shorter labels: 数据 / 推荐 / 报告.
- AutoTune tab labels are shortened: 配置 / 对比 / 推荐 / 审计.
- Global tab minimum width reduced for narrow control panels.
- Main B-scan card and empty-state card receive a direct theme reinforcement after global QSS application.
- Matplotlib toolbar receives local theme-aware styling.

## Boundary

This is a visual/readability patch only.

It does not change:

- GPR processing algorithms
- AutoTune production scoring
- AutoTune execution
- gprMax models or runs
- Evidence artifacts
- PyVista / 3D viewer behavior
- Legacy workbench retirement state
