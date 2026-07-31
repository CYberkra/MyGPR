# Project data evolution

Persistent documents use `mygpr.<family>.vN`. Mutable families are registered in
`core/schema_registry.py`. Opening an older document creates a migration snapshot,
applies sequential idempotent migrations, validates the result, and commits it
atomically. Corrupt documents are copied to quarantine and never converted into
an empty/default state. A document from a newer MyGPR version opens read-only.

Project writes require a single-writer project session. Report packages freeze a
project snapshot and are immutable after sealing.
