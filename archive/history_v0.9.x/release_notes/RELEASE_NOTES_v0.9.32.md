# MyGPR 0.9.32 — Release Consistency Hardening

This patch closes release-metadata and launcher drift found in the 0.9.31 frozen source package.

## Fixed

- Release tests no longer hard-code version 0.9.28.
- The cross-platform runner now launches `mygpr_studio.py`, not the frozen `app_qt.py` frontend.
- Windows launchers, package metadata and current documentation now declare 0.9.32 consistently.
- The version gate now verifies `VERSION`, `pyproject.toml`, packaging specs, launcher banners, current-state documents, changelog and release notes.

## Unchanged

- Backend API v1, project storage schema, numerical algorithms and legacy frontend freeze hashes are unchanged.
- The historical Qt frontend remains migration-only and excluded from production wheels.

## Remaining external evidence

Real PyQt6/Windows startup, DPI, installer and seven-workspace end-to-end acceptance still require the target Windows environment.
