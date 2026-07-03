# MyGPR V0.8.66 Deep Line-Level Audit

Base package: V0.8.65  
Output package: V0.8.66  
Scope: source package text/code audit plus targeted runtime regression.

## What was audited

A line-level static audit scanned every line in auditable source/text files with suffixes:

- Python, batch/cmd/PowerShell, Markdown, QSS, YAML/YML, JSON, TXT, INI, SPEC.
- Generated/runtime-heavy folders were excluded from source risk scoring: `.venv`, cache folders, `dist`, `build`, `output`, `experiments`.
- The scan covered 595 auditable files and 157,845 source/text lines.
- Python parse coverage covered 388 Python files and 123,563 Python lines.
- Python AST parse result: 0 syntax errors.

Raw machine-readable evidence is archived in:

- `docs/audits/deep_line_audit_v0866_raw_summary.json`
- `docs/audits/deep_line_audit_v0866_findings.jsonl`

## Finding triage

The automated scanner initially flagged these classes:

| Class | Count | Triage |
|---|---:|---|
| Empty implementation/pass lines | 94 | Mostly defensive optional dependency or UI fallback blocks; no syntax failure. |
| Silent exception-looking lines | 413 | Requires future hygiene pass; current product flow tests continue to pass. |
| TODO/placeholder wording | 175 | Mostly docs and future roadmap language; visible V0.8.66 scope items were closed. |
| Qt `exec()` calls | 5 | False positives from Qt dialog/app event loop APIs, not Python `eval/exec`. |
| Debug/CLI `print()` lines | 321 | Mostly CLI tools/tests/scripts; not a GUI completeness blocker. |
| Local absolute path-looking text | 428 | Mostly historical docs/tests/audit artifacts; preflight excludes historical docs. |
| Subprocess calls | 55 | Expected in launchers, CLI, gprMax runner and tests. |

No real Python `eval()` or Python dynamic `exec()` calls were found in runtime source. The five high-severity raw hits are Qt APIs named `.exec()` / `app.exec()`, which are normal modal dialog/application event-loop calls.

## Gaps found and fixed in V0.8.66

1. Standard runtime requirements file was missing.
   - Added `requirements.txt` for runtime use.
   - Converted `requirements-dev.txt` to layer `pytest` on top of runtime dependencies.
   - Updated Windows installer, checker guidance and README to reference runtime requirements for normal users.

2. gprMax simulation validation page accepted malformed GPU device text too silently.
   - Invalid GPU tokens now block command generation and copying.
   - The UI shows a clear `GPU 设备格式无效` message.
   - Added regression coverage.

3. `cli_batch.py resume` was still a placeholder while the CLI advertised the command.
   - Implemented `resume --summary <summary.json>`.
   - It reads the previous summary, finds failed job IDs, reloads the original config, reruns only those failed jobs, and writes a fresh resumed summary.
   - Added regression coverage for no-failed-jobs and missing-summary paths.

4. Unsupported method/family errors used wording that looked like unfinished work.
   - Changed guardrail messages from unfinished-style wording to explicit unsupported-method wording.

## Boundaries preserved

- No processing algorithm changes.
- No AutoTune scoring changes.
- No Evidence schema changes.
- No change to gprMax execution semantics except safer UI command planning.
- No long-running gprMax simulation was launched from the GUI.

## Verification run

The following checks passed in this sandbox after the fixes:

```text
python scripts/preflight_check.py
python scripts/check_version_consistency.py --expected 0.8.66
python -m compileall -q app_qt.py cli_batch.py core ui scripts tests PythonModule

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q \
  tests/test_cli_batch_profiles.py \
  tests/test_simulation_validation_page_ui.py \
  tests/test_requirements_files.py
# 20 passed, 1 warning

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q \
  tests/test_workbench_ui.py \
  tests/test_delivery_page_ui.py \
  tests/test_processing_lab_ui.py \
  tests/test_interpretation_workbench_ui.py \
  tests/test_spatial_synthesis_ui.py \
  tests/test_gprmax_campaign_loader.py \
  tests/test_gprmax_campaign_validator.py \
  tests/test_gprmax_campaign_preview.py \
  tests/test_gprmax_campaign_cli.py
# 40 passed, 1 skipped

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q \
  tests/test_gprmax_campaign_runner_execution.py::test_runner_success_writes_logs_and_manifest \
  tests/test_gprmax_campaign_runner_execution.py::test_runner_failure_return_code_recorded \
  tests/test_gprmax_campaign_runner_execution.py::test_runner_supports_executable_with_inline_args \
  tests/test_gprmax_campaign_runner_execution.py::test_runner_timeout_recorded \
  tests/test_gprmax_campaign_runner_execution.py::test_runner_cancelled_recorded \
  tests/test_gprmax_campaign_runner_execution.py::test_cli_refuses_invalid_scene_for_execution \
  tests/test_gprmax_campaign_runner_execution.py::test_cli_run_valid_scene_with_fake_executable \
  tests/test_gprmax_campaign_runner_execution.py::test_cli_num_runs_forwards_gprmax_n_argument
# 8 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q \
  tests/test_gprmax_campaign_runner_execution.py::test_cli_rejects_non_positive_num_runs \
  tests/test_gprmax_campaign_runner_execution.py::test_cli_gpu_flag_passthrough_without_device \
  tests/test_gprmax_campaign_runner_execution.py::test_cli_gpu_single_device_passthrough \
  tests/test_gprmax_campaign_runner_execution.py::test_cli_gpu_multi_devices_passthrough \
  tests/test_gprmax_campaign_runner_execution.py::test_cli_num_runs_and_gpu_passthrough_combined \
  tests/test_gprmax_campaign_runner_execution.py::test_cli_gprmax_python_builds_python_module_command \
  tests/test_gprmax_campaign_runner_execution.py::test_cli_gprmax_python_missing_path_fails_fast \
  tests/test_gprmax_campaign_runner_execution.py::test_cli_rejects_gpu_device_and_gpu_devices_together
# 8 passed

QT_QPA_PLATFORM=offscreen MYGPR_TEST_MODE=1 python -m pytest -q \
  tests/test_workbench_project_core.py \
  tests/test_processing_session_service.py \
  tests/test_delivery_service.py \
  tests/test_spatial_synthesis_service.py \
  tests/test_interpretation_service.py \
  tests/test_gpr_format_registry_and_readers.py \
  tests/test_gpr_io_airborne_contract.py
# 48 passed
```

Notes:

- One UI test is skipped because this Linux/offscreen host does not provide the Windows CJK font fallback used by that test.
- `pip check` in the shared sandbox reports an unrelated global `moviepy`/`pillow` conflict. MyGPR does not depend on `moviepy`; this is not treated as a package failure.
- This is an automated line-by-line static audit plus targeted manual triage of flagged lines. It is not a formal proof that every possible runtime path is bug-free.
