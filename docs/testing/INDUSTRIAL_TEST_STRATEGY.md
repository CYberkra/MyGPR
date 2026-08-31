# MyGPR Industrial Test Strategy

The test system uses five evidence layers:

1. **Legacy regression** preserves historical defects and compatibility knowledge.
2. **Static contracts** verify architecture, packaging, schemas and prohibited dependencies.
3. **Industrial automated tests** verify end-to-end workflows, crash recovery, bounded resources, input properties and scientific fingerprints.
4. **External acceptance** records Windows destructive I/O, CUDA and hardware-in-loop evidence. Commercial release gates fail when required evidence is absent or stale.
5. **Traceability and coverage** bind requirements and risks to tests and prevent regression of critical modules.

Industrial tests must declare at least one `requirement` and one `risk` marker. P0/P1 requirements with automated verification must have a passing automated test. External requirements must have signed/hashed evidence in the path declared by `config/industrial_acceptance_matrix.json`.

Coverage is a ratchet, not a vanity target. Critical transaction, job, validation and backend modules have explicit branch thresholds. New code is expected to meet the diff thresholds in `config/coverage_policy.json`.

Scientific validation uses immutable hashes for the six Yingshan field lines, a repository-safe deterministic trace subset, a frozen preprocessing fingerprint, and borehole comparison baselines. Full raw files remain external acceptance assets and are verified by SHA-256 when available.
