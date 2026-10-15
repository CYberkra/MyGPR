# Current State — Desktop Qt GUI + headless backend

Version: 0.10.0. Source-tree review: 2026-10-14 (includes acquisition HDF5 compatibility changes).
Platform integration documentation review: 2026-10-15.

This package now ships both the Qt6 desktop frontend and the headless backend.
It is the processing, imaging/interpretation and deliverables component of the
[CDUT-UavGPR platform](https://github.com/MuseLinn/CDUT-UavGPR-Controller).
Acquisition hardware and airborne/ground communications belong to the parent repository.
See [integration boundaries and pending alignment](docs/developer/CDUT_PLATFORM_INTEGRATION.md).

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
  timing, component and aligned sensor gaps for completed files marked `real`/`magnitude`.
  Parent `fdata` and acquisition files without a component marker remain unsupported.
  Vendor binary readers are present
  but are not wired into the project import wizard.
- CLI jobs use explicit `methods`; `recommended_profile` has been removed.
- `config/architecture_policy.toml` now declares `[layers.ui]` with a migration
  exception for `desktop_backend_facade.py`.

## Environment and specifications

- Python 3.12–3.13; direct runtime versions are in [README](README.md),
  [requirements.txt](requirements.txt), and [requirements-gui.txt](requirements-gui.txt).
- Design specifications remain in [docs/superpowers/specs](docs/superpowers/specs/README.md).
  The July three-page UI is historical. The PyInstaller `gpr_gui.spec` was
  deleted in `a9fb92ef`; the current Windows installer recipe is
  `packaging/windows/MyGPR.spec` + Inno Setup. See [build and validation](docs/releases/windows-build.md).
- Local checks and historical CI do not establish Windows user acceptance or
  physical validation of field results.
