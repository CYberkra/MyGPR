# MyGPR Backend API v1

`MyGPRBackend` is the sole presentation-facing composition root. API v1 is frozen by
`config/backend_api_v1.json` and `scripts/check_backend_api_contract.py`.

Compatibility rules:

- Existing method names, required parameters, DTO field names, field order, job states and error-code semantics are stable.
- New optional fields and new methods may be added without changing API v1.
- Breaking changes require a new API major version and a migration adapter.
- Job events and snapshots are serializable through `to_dict()`.
- Failures use `mygpr.error_info.v1`; frontends must branch on `error_code`, not Python exception class names.
- Project-scoped jobs hold a session lease from queueing through terminal cleanup.
- Large in-memory results may be replaced by `JobResultSummary`; persistent workflows return project artifact references.

Job retention guarantees:

- Per-job event history, terminal-job count, terminal TTL, per-result size and aggregate retained-result bytes are bounded.
- Oversized results are represented by `mygpr.job_result_summary.v1` rather than embedded matrices.
- `JobEvent.to_dict()` and `JobSnapshot.to_dict()` produce JSON-safe structures and summarize numerical arrays.
- `forget()`, `prune()` and `release_result()` are explicit lifecycle operations for presentation layers.
- Static release attestations are bound to the current Git commit (or a deterministic source-tree digest outside CI), so they cannot be reused for different source.
