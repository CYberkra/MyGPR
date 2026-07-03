# GX-UI-009 AutoTune Recommendation Console Redesign

## Purpose

This task redesigns `AutoTuneTuningPage` into a focused **AutoTune 参数推荐** console.

The page is intentionally scoped to AutoTune parameter recommendation only. It is not a broad Research Lab, not a gprMax simulation page, and not a 3D scene viewer.

## Design model

The page follows a four-region layout:

1. Header status bar
   - data source
   - workflow step
   - ROI state
   - candidate count
   - recommendation readiness
   - risk level

2. Left tuning setup
   - processing step
   - candidate space
   - SVD rank sweep
   - ROI settings
   - scoring metrics
   - safety / claim-boundary policy

3. Center candidate comparison
   - Raw/Input preview placeholder
   - Candidate preview placeholder
   - Recommended preview placeholder
   - Top-3 candidate ranking table

4. Right recommendation explanation
   - recommended parameters
   - scoring explanation
   - risk warnings
   - claim boundary

Bottom drawer:

- Trial Table
- Metrics
- Logs
- Warnings
- Claim Boundary

## Scope

Implemented:

- UI-local `AutoTuneRecommendationState`
- dynamic ROI status
- dynamic candidate count
- dynamic scoring metric count
- dynamic risk level
- dynamic recommended candidate preview
- dynamic Top-3 ranking table
- dynamic Trial Table
- dynamic warnings and claim-boundary text
- legacy `AutoTunePage` compatibility layer retained

Not implemented:

- production AutoTune execution
- production scoring changes
- real B-scan loading
- Evidence export
- gprMax execution
- PyVista / embedded 3D viewer

## Safety boundary

This page does not claim AutoTune superiority. It only prepares a clearer UI for fixed-workflow parameter recommendation and manual review.

## Compatibility

`ui/gui_auto_tune_page.py` and `ui/research_console_page.py` remain as legacy modules. The new page still instantiates a hidden legacy `AutoTunePage` to satisfy existing `app_qt.py` signal/state calls during the transition.
