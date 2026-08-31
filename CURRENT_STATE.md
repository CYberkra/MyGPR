# Current State — Desktop Qt GUI + headless backend

Version: 0.9.37

This package now ships both the Qt6 desktop frontend and the headless backend.

- Desktop entry: `app_qt.py` or console script `mygpr`.
- Backend interfaces: `mygpr/interfaces/` and `config/backend_api_v1.json`.
- Application services: `mygpr/application/`.
- GUI packages: `ui/`, launched by `MyGPRMainWindow`.

Install with GUI dependencies:

```bash
pip install -e ".[gui]"
```

Run the desktop application:

```bash
python app_qt.py
# or, after installation:
mygpr
```

## Architecture status

- `ui/` → `core/` violations have been removed via `ui/desktop_backend_facade.py`.
- `MyGPRMainWindow` (was 1360 lines) has been split into the core assembler
  (`ui/main_window.py`, ~715 lines) plus signal-handler mixins
  (`ui/page_coordinator.py`) grouped by domain:
  project lifecycle, line/artifact, import/preflight, processing,
  interpretation, delivery, and job center.
- `config/architecture_policy.toml` now declares `[layers.ui]` with a migration
  exception for `desktop_backend_facade.py`.
