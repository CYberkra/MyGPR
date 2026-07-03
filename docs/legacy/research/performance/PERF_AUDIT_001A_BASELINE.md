# PERF-AUDIT-001A Baseline

Version: 0.8.55
Scope: performance audit foundation and first low-risk UI smoothness safeguards.

## Baseline methodology

This pass does not change numerical processing algorithms, AutoTune scoring, candidate generation, gprMax data contracts, or Evidence schema.  It establishes instrumentation and identifies first-order UI hotspots.

Baseline checkpoints for manual desktop validation:

1. Application startup to main window ready.
2. CSV import to first B-scan render.
3. Mouse hover coordinate readout on a loaded B-scan.
4. Manual ROI drag responsiveness.
5. AutoTune recommendation generation UI responsiveness.
6. AutoTune run with stepwise B-scan preview.
7. Slider comparison drag responsiveness.
8. Runtime log panel updates during processing.

## Added instrumentation

- `core/perf_monitor.py` adds an optional aggregate `PerfMonitor`.
- `plot_data()` records `display.plot_data_ms`.
- `_prepare_view_data()` records cache-hit/miss display preparation timings.
- `_compute_vmin_vmax()` records vmin/vmax timing and cache hit timing.

These timings are display-path observability only. They do not affect processing arrays or scoring metrics.

## First low-risk optimizations in this pass

1. Prepared-view cache
   - Caches display-only cropped/preprocessed B-scan payloads by data revision and display settings.
   - Avoids repeating display preprocessing when only overlays/view refreshes change.

2. vmin/vmax cache
   - Caches display range calculation by data revision, array identity, and display stretch settings.
   - Avoids repeated full-array finite filtering and percentile calculations on unchanged display payloads.

3. Runtime log batching
   - Batches visible QTextEdit log appends at an 80 ms cadence.
   - Structured `LogEventBuffer` remains immediate and unchanged for audit/export purposes.

## Non-goals

- No algorithm formula changes.
- No AutoTune scoring changes.
- No candidate generator changes.
- No Evidence manifest schema changes.
- No display downsampling yet.
- No PyQtGraph/GPU/C++ rewrite.

## Offscreen smoke measurement

Environment: container/offscreen Qt, synthetic display-only B-scan shaped `501 x 2378`.

Observed timing snapshot:

- First `plot_data`: included full Matplotlib draw path; observed max about 145 ms in the offscreen run.
- Repeat `plot_data` on unchanged data/display settings: prepared-view cache hit about 0.28 ms; vmin/vmax cache hit about 0.16 ms.
- First vmin/vmax computation on the same array: about 4.9 ms.

These values are not a substitute for Windows desktop profiling, but they verify that the caches activate without changing processing arrays.
