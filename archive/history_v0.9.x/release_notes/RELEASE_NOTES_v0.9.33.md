# MyGPR 0.9.33 — Phase 18A Project Data & QC Migration

## Scope

This release begins the capability-parity migration from the frozen legacy frontend to MyGPR Studio. It closes the project-data and quality-control slice without importing legacy Qt classes.

## Added

- Formal `ProjectMaintenanceService` and `ProjectMaintenanceServiceProtocol`.
- Project metadata editing.
- Multi-file batch line import with per-file diagnostics.
- Single-line and whole-project quality checks.
- Source-file evidence verification, relink and CSV manifest export.
- Transactional B-scan orientation transpose with automatic backup and post-fix QC.
- Safe line deletion into the project-local recycle area.
- Studio project-workspace entries for all above operations.

## Fixed

- Corrected HDF5 transpose progress callback argument ordering.

## Boundary

The frozen `ui/` and `compatibility/legacy_app_qt.py` remain unchanged. Processing, interpretation, GIS/3D, report authoring and research-console parity are scheduled for later Phase 18 slices.
