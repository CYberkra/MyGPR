# ADR-003: Explicit legacy processing bridge

## Status

Accepted

## Context

The legacy processing UI remains feature-rich but is coupled to its page
widgets. Replacing every processing workflow before delivering project and QC
management would create unnecessary regression risk.

## Decision

Open `GPRGuiQt` only on request for a selected formal, QC-ready line. The bridge
loads a copy into the legacy window. Processing never writes back automatically;
the user must explicitly save the current result as a versioned project result.
Temporary or QC-blocked lines cannot enter formal processing.

## Consequences

- Raw project data remains immutable.
- Intermediate legacy actions do not flood the project with versions.
- The bridge can be removed after the new processing laboratory reaches parity.
