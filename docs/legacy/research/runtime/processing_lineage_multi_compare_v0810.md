# Processing lineage multi-compare design (V0.8.10)

V0.8.10 replaces the previous single “compare with current” action with a lightweight compare tray.

## Interaction model

- Click a lineage step chip to preview that step.
- Click the small dot on the right side of a step to add/remove it from the compare tray.
- Select exactly two steps for slider compare or difference view.
- Select two to four steps for grid compare.
- Click the active compare mode again to return to the formal current result.
- Use Clear to empty the compare tray.

## Display modes

### Slider compare

Requires exactly two selected steps. The main B-scan renders a display-only slider split between the selected steps.

### Difference view

Requires exactly two selected steps. The main B-scan renders `|A - B|` after size alignment. This is display-only and must not be used as a processing step without explicit export context.

### Grid compare

Supports two to four selected steps. Two panels are shown horizontally; three or four panels use a compact grid layout. Display settings and color-scale decisions follow the current B-scan display controls.

## Evidence boundary

The compare tray does not alter formal data, processing history, AutoTune state, or gprMax evidence. It is a display-only diagnostic view. Export sidecars mark the active compare state and selected steps when present.
