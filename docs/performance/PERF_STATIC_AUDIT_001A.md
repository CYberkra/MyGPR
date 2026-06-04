# PERF-AUDIT-001A Static Hotspot Audit

## High-probability hotspots

| Area | Risk | Evidence / code area | First response |
|---|---:|---|---|
| B-scan render path | High | `plot_data()`, `_prepare_view_data()`, `_compute_vmin_vmax()`, Matplotlib `imshow` | Add display caches and timing counters |
| Mouse/ROI/slider events | High | `ui/bscan_interaction_controller.py` | Existing throttling present; keep and refine after measurement |
| Runtime logs | Medium | `_log()` appends to multiple QTextEdit-like widgets | Batch visible appends; keep structured logs immediate |
| AutoTune progress/table updates | Medium | AutoTune UI controllers and worker progress callbacks | Defer to PERF-AUTOTUNE-001C; do not change scoring path now |
| Display/compare page slider | Medium | `_render_slider_compare_panel()`, `_try_update_slider_compare_lightweight()` | Existing lightweight clip update present; validate under real data |
| Large tables / reports | Medium | candidate/trial/report pages | Move to lazy/top-N rendering in later pass |
| Import I/O | Medium | CSV/A-scan loaders | Defer worker/memory audit to PERF-IO-001D |

## Existing positive findings

- Plot refresh already uses a short single-shot timer via `_refresh_plot()`.
- Mouse panning, ROI preview, coordinate readout, and slider compare already include throttle intervals.
- Slider compare already uses lightweight artist updates during drag instead of rebuilding the full figure every motion event.
- Selected-trace markers and hover crosshair drawing have been retired, reducing persistent overlay churn.

## Remaining risks

- Matplotlib full redraw is still the dominant cost for large B-scans.
- Some display operations still convert to float arrays; cache reduces repetition but not first-render cost.
- Cache correctness depends on data revision and display override revision being updated whenever processing arrays change.
- Full GUI runtime measurements are still needed on Windows with real project data.
