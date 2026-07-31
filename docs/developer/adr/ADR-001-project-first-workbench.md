# ADR-001: Project-first exploration workbench

## Status

Accepted

## Context

The historical GUI organizes behavior around page widgets and hidden feature
tabs. Controllers read `page_xxx` controls directly, which makes multi-line
field projects and lifecycle-oriented workflows difficult to introduce safely.

## Decision

The normal application entry creates `MyGPRWorkbenchWindow`. It keeps project
resources, data documents, contextual inspection, and task/QC evidence visible
while switching between lifecycle workspaces. Document tabs represent data
objects only. The legacy `GPRGuiQt` remains available through an explicit
processing bridge during migration.

## Consequences

- New application services must not depend on page widgets.
- Existing processing algorithms and the legacy window remain reusable.
- Future workspaces can replace legacy capabilities incrementally.
