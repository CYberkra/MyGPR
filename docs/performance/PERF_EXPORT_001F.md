# PERF-EXPORT-001F Export / Report Smoothing

Version: 0.8.60

## Scope

This pass audits export paths that can block the Qt main thread during ZIP, PNG,
JSON, HTML or VTK/CSV writes. It does not change processing arrays, AutoTune
scoring, candidate generation, gprMax contracts or Evidence schema semantics.

## Changes

- Added `ui.export_worker.ExportTaskWorker` and `start_export_task()` for pure
  background export callables.
- Added `MainWindowExportMixin._run_background_export()` with synchronous
  fallback for tests/headless compatibility.
- Moved these user-triggered export actions to background tasks:
  - AutoTune comparison evidence export.
  - Replay evidence ZIP export.
  - UAV georeference 3D export.
- Added `core.export_performance.write_json_sidecars()` and
  `write_text_sidecars()` for report sidecar batch writing.
- Instrumented report export timing:
  - `export.report_figure_600dpi_ms`
  - `export.report_html_ms`
  - `export.report_sidecar_json_ms`
  - `export.report_sidecar_text_ms`
  - `export.autotune_comparison_bundle_ms`
  - `export.replay_evidence_zip_ms`
  - `export.airborne_georeference_3d_ms`

## Boundaries

- `generate_report()` still performs the current Matplotlib figure export on the
  main GUI thread because it uses the live figure/canvas. This is intentional for
  safety; moving live Qt/Matplotlib state into a worker would be unsafe.
- Long pure export actions now run in worker threads after GUI state has been
  snapshotted.
- Report and manifest schemas remain unchanged.

## Next candidates

- Dedicated report figure export path from a detached Matplotlib Agg figure.
- Optional report-export progress dialog.
- Large HTML/PNG export cancel support.
