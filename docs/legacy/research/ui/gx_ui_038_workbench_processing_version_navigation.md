# GX-UI-038 Workbench Processing-Version Navigation Completion

Base version: 0.8.63  
Output version: 0.8.64

This pass closes an unfinished project-first Workbench UI path: saved processing versions were visible as a resource count but could not be opened directly from the resource tree.

## Completed

- Replaced ad-hoc `results/*/result.json` scanning in the Workbench tree with `ProjectService.list_processing_results()`.
- Stored result-tree payloads as `("result", line_id, result_id)` so they can be resolved safely through the project service.
- Added a read-only processing-version document view with:
  - B-scan preview for two-dimensional result data;
  - result ID / line ID / data shape / created timestamp metadata;
  - compact processing-chain table;
  - inspector context for the selected version.
- Hardened global splitter restore after Qt show events and kept the bottom task/QC/evidence drawer at a readable minimum height.
- Added regression coverage in `tests/test_workbench_ui.py`.

## Boundaries

- No processing algorithm changes.
- No AutoTune scoring changes.
- No gprMax execution changes.
- No Evidence schema changes.
- No project-file schema change.

## Verification

- `QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q tests/test_workbench_ui.py`
- `QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q tests/test_version_consistency.py`
- Additional targeted UI smoke tests were run individually because the full Qt/Matplotlib set can exhaust offscreen resources when run as one long process.
