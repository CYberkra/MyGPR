# ADR-004: Native project processing session

## Status

Accepted

## Context

The classic processing window owns mature algorithms but also owns widget state,
which prevents the project-first workbench from editing and replaying processing
chains independently.

## Decision

Use `ProcessingSessionService` as the UI-independent processing boundary. A
session loads one project line, preserves immutable original data, invokes the
shared processing engine, injects runtime header and trace metadata, and writes
results only through explicit version saves. Chain edits replay deterministically
from the original data. Preview and saved-version browsing never mutate the
formal current result.

The native processing laboratory calls the session service from background
workers. The classic window remains available only as a migration fallback.
Manual-baseline versus AutoTune comparison also belongs to the session boundary:
it reuses the existing core comparison and evidence-export APIs, stores only the
last comparison in memory, does not mutate current processing data, and writes
exported evidence under the project `exports` tree.
Display controls such as colormap, symmetric limits, percentile clipping,
shared color scale, and colorbar visibility remain local to the processing
laboratory UI. They must not create processing-chain steps or modify session
arrays.

## Consequences

- The new workbench can process data without reading classic `page_xxx` widgets.
- AutoTune and motion-compensation methods reuse their existing core contracts.
- Processing chains, parameters, warnings, metadata, and arrays remain
  versioned and auditable.
- AutoTune comparison evidence can be produced without opening the classic
  window or reading classic `page_auto_tune` widget state.
- Basic display and comparison decisions can be made in the native laboratory
  without opening the classic advanced display page.
- Expensive processing operations do not block the workbench event loop.
