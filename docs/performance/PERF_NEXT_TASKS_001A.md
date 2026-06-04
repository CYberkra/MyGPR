# PERF Next Tasks after 001A

## PERF-RENDER-001B

- Add explicit performance summary export for display timings.
- Measure first render vs repeat render on the same B-scan.
- Add safe draw-idle coalescing if Matplotlib still queues excessive redraws.
- Consider optional display-only preview downsampling during drag/slider interaction.

## PERF-AUTOTUNE-001C

- Audit progress-signal frequency.
- Batch trial table and log updates while a run is active.
- Ensure candidate scoring does not trigger full B-scan redraw per candidate.

## PERF-IO-001D

- Audit CSV/A-scan loading and dtype conversion.
- Separate processing arrays from display preview arrays.
- Avoid unnecessary copy/transpose chains.

## PERF-COMPARE-001E

- Validate slider compare on large arrays.
- Cache left/right display panels by lineage step and display settings.

## PERF-EXPORT-001F

- Move expensive report/image exports to worker threads with progress feedback.
