# MyGPR Backend Source 0.9.37

This delivery intentionally contains no desktop or GUI frontend.

## Included

- `core/`: processing, synchronization, AutoTune, reporting and gprMax algorithms.
- `mygpr/`: application services, domain models, infrastructure adapters and public backend interfaces.
- `PythonModule/`: retained algorithm modules.
- `scripts/`: backend validation, benchmarking, migration and gprMax utilities.
- `experiments/`, `sample_data/`, `config/`, `configs/`: backend research and reproducibility assets.
- `tests/`: backend and algorithm tests that do not import frontend packages.
- `cli_batch.py`, `backend_smoke.py`, `backend_project_smoke.py`: headless entry points.

## Intentionally removed

`ui/`, `studio_app/`, `frontend_sdk/`, `compatibility/`, Qt presentation code, GUI launchers, desktop packaging specifications, UI assets, UI documentation and frontend-specific tests.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements-core.txt
python -m pip install -e .
```

## Validation

```bash
python backend_smoke.py
python backend_project_smoke.py
python -m pytest -q
```

## New frontend integration

Use the public backend boundaries under `mygpr/interfaces/`, the application services under `mygpr/application/`, and `config/backend_api_v1.json`. Do not import persistence internals directly from a new frontend.
