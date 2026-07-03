# MyGPR V0.8.64 Patch Notes

## Scope

This pass continues the UI completion work from the uploaded package. It focuses on the project-first Workbench resource tree and environment-validated UI regression closure. It does not change processing algorithms, AutoTune scoring, gprMax execution, or Evidence schema semantics.

## Findings

1. Saved processing versions were counted under the Workbench `处理版本` tree group, but the tree children were inert: selecting them did not open the saved result.
2. The result-tree refresh path previously depended on raw `result.json` filesystem scanning instead of the canonical `ProjectService.list_processing_results()` API.
3. The global splitter layout restore could leave the bottom task/QC/evidence drawer too short in offscreen/first-show Qt geometry negotiation, making the restored UI feel unfinished.

## Changes

- Bumped the package to `0.8.64` and added a matching changelog entry.
- Populated Workbench processing-version tree nodes from `ProjectService.list_processing_results()`.
- Added `open_result_document()` so selecting a saved processing version opens a read-only preview tab with B-scan image, result metadata, and processing-chain table.
- Updated the inspector context for saved processing versions.
- Hardened the bottom drawer layout restore with a readable minimum height and a post-show splitter reapply.
- Added a regression test that saves a processing version, reopens the project, selects it from the tree, and verifies the document/inspector update.

## Validation run in this sandbox

```text
python -m compileall -q app_qt.py ui core scripts cli_batch.py
passed

python scripts/check_version_consistency.py --expected 0.8.64
version_check_ok: 0.8.64

python scripts/preflight_check.py
[OK] Preflight passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 pytest -q \
  tests/test_workbench_ui.py \
  tests/test_environment_and_workbench_polish.py \
  tests/test_workbench_entry_and_bridge.py \
  tests/test_delivery_page_ui.py \
  tests/test_version_consistency.py -q
18 passed, 1 skipped

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 pytest -q tests/test_processing_lab_ui.py -q
6 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 pytest -q tests/test_interpretation_workbench_ui.py -q
2 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 pytest -q tests/test_spatial_synthesis_ui.py -q
1 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 pytest -q tests/test_app_qt_controller_boundaries.py -q
4 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 pytest -q tests/test_autotune_recipe_ui.py -q
7 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 pytest -q tests/test_autotune_tuning_page_target_response.py -q
1 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 pytest -q tests/test_auto_tune_result_dialog.py -q
2 passed

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_theme_polish_stylesheet.py -q
1 passed
```

The full Qt/Matplotlib GUI test set was not run as one monolithic process in this sandbox because several long-lived offscreen GUI runs were terminated by the host resource guard. The targeted UI and preflight checks above completed successfully.

## Environment note

Dependencies were installed in this sandbox from `requirements-dev.txt` for validation. The delivered source does not include a virtual environment. On Windows, use `install_mygpr_environment.bat` / `安装MyGPR环境.bat`, then launch with `start_mygpr.bat`.
