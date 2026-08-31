# Backend handoff

1. Keep `core/` and `mygpr/` free of Qt imports.
2. A redesigned frontend must call public interfaces/application services rather than storage internals.
3. Preserve project schemas, artifact lineage and evidence SHA-256 contracts.
4. Run `backend_smoke.py`, `backend_project_smoke.py` and the backend test suite before integration.
5. The Qt frontend (`ui/`) is active and ships with this delivery. All UI→core/domain/application imports must route through `ui/desktop_backend_facade.py`.
