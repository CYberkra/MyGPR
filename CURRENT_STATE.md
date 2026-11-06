# Current State — Desktop Qt GUI + headless backend

Source version: 0.11.4. The [v0.11.4 release notes](docs/releases/v0.11.4.md) describe the installation and upgrade path. The [specification index](docs/superpowers/specs/README.md) distinguishes current contracts from historical design records.

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
- `MyGPRMainWindow` assembles seven pages behind a top tab bar, a project tree,
  and a bottom output panel. `page_coordinator.py` delegates signal wiring to
  `coordinator_project.py`, `coordinator_processing.py`, and `coordinator_jobs.py`.
- Home and Project share one project workspace. Processing uses manually docked
  B-scan tab groups; the active saved result is the next input, with separate
  drafts and versioned direct dependencies for new result branches.
- The project landing page supports searching and pinning recent projects. Projects, lines, processing artifacts and spatial results have permanent deletion with a confirmed scope. Original imported data and sensor source files are protected, including sources within the project directory.
- The Windows installer includes signed application updates. Settings → About opens the user guide and release downloads. The graphite and honey-apricot icon depicts a UAV surveying a landslide cross-section and is shared by the desktop application and installer.
- New projects use per-line HDF5 arrays plus a SQLite catalog; legacy project
  layouts remain a compatibility path. See [Hybrid Store](docs/developer/HYBRID_PROJECT_STORE_V1.md).
- Project imports accept single or multiple CSV/TXT/NPY/NPZ/H5/HDF5 files and provide an application-side quality center. CDUT acquisition HDF5 retains
  timing, component and aligned sensor gaps for completed files marked `real`/`mag`/`magnitude`/`fdata`;
  `fdata` remains explicitly identified as instrument-display sampling. Acquisition files without a component marker remain unsupported.
  Vendor binary readers are present but are not wired into the project import wizard.
- The application exposes batch processing with resource estimates, saved-step results by default and optional block-friendly final-only execution,
  sensor calibration templates, annotation confirmation and borehole checks, spatial exports,
  configurable report packages, and restore-to-new-project. Processing result tabs show compact
  processing actions and parameters from persisted lineage rather than opaque result names.
- Processing supports inclusive trace-range cropping with aligned distance and sensor metadata,
  cross-line raw/result split-screen previews, and continued processing from a selected result.
  B-scan time axes use readable physical-time intervals; Hilbert envelope is a display-only option
  unless the corresponding processing method is explicitly run.
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
