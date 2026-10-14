# MyGPR Architecture

- `ui/`: PyQt6 pages, widgets, controllers and coordinator wiring. The frontend reaches core/application/domain capabilities through `ui/desktop_backend_facade.py`.
- `mygpr/interfaces/`: public composition root (`MyGPRBackend`) and CLI interfaces.
- `mygpr/application/`: use cases and ports, without Qt or presentation imports.
- `mygpr/domain/`: domain models and engineering rules.
- `mygpr/infrastructure/`: adapters for processing, persistence, acquisition, GIS and reporting.
- `core/`: active legacy kernels and project storage, reached through the infrastructure adapters.
- `PythonModule/`: legacy wrappers; dynamic registry imports remain in use.

The native algorithm catalog lives in `mygpr/infrastructure/processing/algorithms/methods.py`.
Project storage uses per-line HDF5 plus SQLite, described in [Hybrid Store](HYBRID_PROJECT_STORE_V1.md).
The public contract is `config/backend_api_v1.json`; the architecture policy is `config/architecture_policy.toml`.

`scripts/check_architecture.py` and `scripts/check_debt_budget.py` enforce dependency direction and growth limits. See [CLAUDE.md](../../CLAUDE.md) for UI assembly and threading rules, and the [specification index](../superpowers/specs/README.md) for design history.
