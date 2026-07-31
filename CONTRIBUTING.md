# Contributing to MyGPR

1. Use Python 3.11-3.13 and install from `requirements-dev.txt`.
2. Run `python scripts/run_quality_gate.py affected --plan` and then `affected`.
3. New persistent JSON structures require an owned schema entry and migration policy.
4. New algorithms require `AlgorithmSpec`; placeholders are not allowed in the production registry.
5. Domain/application code may not import Qt or UI packages.
6. File writes must use the project repository/storage primitives.
7. Broad exception handling, silent handlers, `sys.path` mutation, and oversized modules are debt-ratcheted and may not increase.
8. Compatibility code requires an owner, replacement, removal version, and removal condition.
9. Release changes require the full release gate and packaged-application smoke test.
