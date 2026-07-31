# MyGPR Architecture

- **domain**: pure data and engineering rules; no Qt or filesystem dependencies.
- **application**: use cases and ports; no presentation imports.
- **infrastructure**: project repository, durable storage, GIS, report rendering, device/file adapters.
- **presentation/qt**: desktop views and Qt adapters.
- **plugins**: typed processing extensions split into production and research catalogs.
- **compatibility**: temporary adapters with explicit retirement contracts.

The current codebase is being migrated incrementally. `scripts/check_architecture.py`
and `scripts/check_debt_budget.py` enforce dependency direction and prevent debt growth.
