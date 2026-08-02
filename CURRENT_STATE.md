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

## Known architecture debt

The Qt frontend (`ui/`) currently imports several `core/` modules directly:

- `ProjectController` → `core.gpr_data_model`, `core.gui_rendering`
- `ProcessingController` → `core.method_registry_metadata`, `core.methods_registry`
- `ProcessingPage` → `core.gui_rendering`
- `ProjectPage` → `core.gpr_format_registry`

This means the GUI is not fully decoupled from the backend kernel. A clean
fix requires introducing adapter layers in `mygpr/interfaces/` or `ui/adapters/`
so that pages/controllers only depend on the backend public API. This is a
larger refactor and is intentionally left for a follow-up branch.
