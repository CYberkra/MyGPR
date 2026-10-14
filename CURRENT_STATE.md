# Current State — Desktop Qt GUI + headless backend

Version: 0.9.38. Source-tree review: 2026-10-14 (includes acquisition HDF5 compatibility changes).

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
- `MyGPRMainWindow` assembles eight pages behind a top tab bar, a project tree,
  and a bottom output panel. `page_coordinator.py` delegates signal wiring to
  `coordinator_project.py`, `coordinator_processing.py`, and `coordinator_jobs.py`.
- New projects use per-line HDF5 arrays plus a SQLite catalog; legacy project
  layouts remain a compatibility path. See [Hybrid Store](docs/developer/HYBRID_PROJECT_STORE_V1.md).
- Project imports accept CSV/TXT/NPY/NPZ/H5/HDF5. CDUT acquisition HDF5 retains
  timing, component and aligned sensor gaps. Vendor binary readers are present
  but are not wired into the project import wizard.
- CLI jobs use explicit `methods`; `recommended_profile` has been removed.
- `config/architecture_policy.toml` now declares `[layers.ui]` with a migration
  exception for `desktop_backend_facade.py`.

## Environment and specifications

- Python 3.12–3.13; direct runtime versions are in [README](README.md),
  [requirements.txt](requirements.txt), and [requirements-gui.txt](requirements-gui.txt).
- Design specifications remain in [docs/superpowers/specs](docs/superpowers/specs/README.md).
  The July three-page UI is historical. The PyInstaller `gpr_gui.spec` was
  deleted in `a9fb92ef`; an executable release flow still needs implementation
  and validation. Source/wheel installation is the current documented route.
- Local checks and historical CI do not establish Windows user acceptance or
  physical validation of field results.
