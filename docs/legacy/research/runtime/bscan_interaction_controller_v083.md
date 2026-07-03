# BscanInteractionController v0.8.3

`ui.bscan_interaction_controller.BscanInteractionController` owns the main B-scan interaction layer.

## Responsibilities

- Mouse press / motion / release on the main Matplotlib canvas.
- Wheel zoom with shift/ctrl axis modifiers.
- Pixel-based pan.
- Manual ROI drag preview and commit.
- Slider-compare split drag.
- Hover crosshair and current trace highlight.
- Selected-trace marker.
- Manual ROI marker.
- Main view limit capture, clamp, and reset.
- Coordinate readout label updates.

## Compatibility

`GPRGuiQt` keeps thin wrapper methods such as `_on_main_canvas_press(...)`, `_draw_manual_roi_marker(...)`, and `_update_hover_crosshair(...)` so existing connections, tests, and UI call sites remain stable.

## Design boundary

This is a conservative extraction.  The controller still reads and writes host-window attributes.  It is not a pure core module; it belongs in `ui/` because it depends on Matplotlib artists and Qt canvas behavior.

## Next architecture step

After this split, the next recommended controller extraction is AutoTune/no-prior synchronization:

```text
ui/autotune_sync_controller.py
```
