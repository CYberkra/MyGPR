# Field UI Round 3: Project Persistence and Data Loop

Implemented in Round 3:

- Added `core/field_project_store.py` for a product-facing field project contract.
- Standardized the UI project folder around `project.json`, `raw`, `processed`, `targets`, `spatial`, `reports`, and `logs`.
- The 1080P field workbench now creates/opens `runtime_projects/field_demo_project` automatically.
- Project management reads line records from disk-backed project state.
- `open_loose_path()` imports a source CSV into `raw/L03/` and updates `project.json`.
- “保存处理结果” writes a `.npy` result and parameter JSON under `processed/L03/`, then updates the line status.
- Target add/save/delete operations persist to `targets/L03_targets.csv`.
- Saving targets also regenerates `spatial/L03_targets_xy.csv`.
- Added tests for structure creation, target persistence/spatial export, and processed-result manifest updates.

Validation commands:

```bash
python -m py_compile core/field_project_store.py ui/field_workbench_window.py
python -m pytest tests/test_field_project_store.py -q
```
