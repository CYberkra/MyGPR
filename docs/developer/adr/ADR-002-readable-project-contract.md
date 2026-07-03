# ADR-002: Readable portable project contract

## Status

Accepted

## Context

Field projects must be portable, auditable, recoverable, and understandable
without a database inspection tool. Existing nearby `manifest.json` files have
acquisition, simulation, or report-specific meanings and cannot serve as the
project root contract.

## Decision

Use versioned JSON records rooted at `project.mygpr.json`. Formal project paths
are relative to the project root. Primary data and discovered or manually
assigned sidecars are copied under `raw/<line_id>`, made read-only, and verified
by SHA-256. Copying completes before a separate background integrity pass, so
users may browse while records show `pending_integrity`. Large processing arrays
remain separate `.npy` files. JSON writes are atomic and a lock file enforces
one writer. Creating a project never overwrites an existing project manifest.
Explicit recovery may replace a lock only after proving that its writer process
is no longer active.

## Consequences

- Project content is inspectable and migration-friendly.
- Large arrays do not inflate JSON files.
- Formalization and integrity checks run in background workbench tasks.
- Integrity mismatch is a hard QC error. Pending integrity is a warning that
  requires an acknowledgement note before formal processing.
- Integrity verification fills only pending hashes. It never replaces an
  established SHA-256 baseline; changed raw data is marked as a mismatch and
  blocked from formal processing.
