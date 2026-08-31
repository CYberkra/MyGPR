# ADR-005: Interpretation, spatial synthesis, and delivery services

## Status

Accepted

## Context

The project-first workbench must support field interpretation and delivery
without returning to page-widget contracts. Interpretation objects, spatial
summaries, reports, and evidence packages need durable project records.

## Decision

Implement three UI-independent services:

- `InterpretationService` stores point, interface-line, and interval objects as
  line-level GeoJSON feature collections with confidence and optional result
  linkage.
- `SpatialSynthesisService` aggregates real trace metadata into tracks,
  terrain/height summaries, and located interpretation features. It reports
  unlocated lines explicitly instead of inventing coordinates.
- `DeliveryService` runs outcome checks, blocks packages on hard errors, and
  builds a report, evidence index, spatial synthesis JSON, manifest, and
  SHA-256 checksum list under `exports/`.

The corresponding workbench pages call these services directly and remain
project-context aware.

## Consequences

- Interpretation, spatial, and delivery data remain inspectable outside the UI.
- Evidence packages are reproducible and auditable from project files.
- Missing spatial metadata is visible as a data-quality state, not hidden by
  placeholder coordinates.
