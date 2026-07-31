# Backend handoff

1. Keep `core/` and `mygpr/` free of Qt imports.
2. A redesigned frontend must call public interfaces/application services rather than storage internals.
3. Preserve project schemas, artifact lineage and evidence SHA-256 contracts.
4. Run `backend_smoke.py`, `backend_project_smoke.py` and the backend test suite before integration.
5. Frontend code is intentionally absent from this delivery.
