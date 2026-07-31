#!/usr/bin/env markdown
# MAINT-002 Stabilization Pass

Date: 2026-05-23

Branch: `main`

Base commit: `68908c1be3fcac64410fe90c7f63b1663839df7a`

Remote HEAD: `68908c1be3fcac64410fe90c7f63b1663839df7a` at audit start

## Scope

This pass reviewed repository hygiene, gprMax campaign backend robustness,
paired output validation, preview/report diagnostics, GX-007 status documents,
and low-risk UI surface status. It intentionally avoided AutoTune scoring,
motion-compensation semantics, `processing_engine`, GUI restructuring, Evidence
repository operations, and generated artifact commits.

## Files Inspected

- `.gitignore`
- `scripts/preflight_check.py`
- `scripts/gprmax_campaign_runner.py`
- `scripts/gprmax_campaign_convert_scene001.py`
- `scripts/diagnose_gprmax_gpu_env.py`
- `core/gprmax_campaign/`
- `experiments/gprmax/GX-007/`
- `docs/gprmax_integration_architecture.md`
- `ui/gui_workbench.py`
- `tests/test_gprmax_campaign_runner_execution.py`
- `tests/test_gprmax_campaign_pairing.py`
- `tests/test_gprmax_campaign_preview.py`
- `tests/test_wiggle_ui_behavior.py`
- `tests/test_auto_tune_recommendation_labels.py`

## P0 Findings

None found in the reviewed scope.

No reviewed issue indicated app startup failure, processing data corruption, or
an already committed generated native artifact.

## P1 Findings

1. Staged native gprMax outputs were not blocked by preflight.
   - Symptom: `.gitignore` had local GX-007 output rules, but preflight did not
     inspect staged paths before commit.
   - Source: `scripts/preflight_check.py`.
   - Consequence: a future commit could accidentally include `.out`, `.h5`,
     `.vti`, `.vtk`, or generated campaign arrays/previews if manually staged.
   - Remedy: implemented staged generated artifact guard in preflight and added
     focused tests.

## P2 Findings

1. Paired output shape mismatch messages were too terse.
   - Symptom: validation reported only raw/background shapes.
   - Source: `core/gprmax_campaign/pairing.py`.
   - Consequence: users had to inspect command arguments to identify the bad
     file path in pairing failures.
   - Remedy: include raw/background paths and shapes in the mismatch issue.

2. Existing target_response preview load failures could escape structured
   preview reporting.
   - Symptom: unsupported target response suffixes in preview mode raised from
     the loader path instead of returning an invalid summary JSON.
   - Source: `core/gprmax_campaign/preview.py`.
   - Consequence: CLI `--json` callers could lose a parseable failure report.
   - Remedy: target response load errors now return structured invalid results
     and write `paired_report_summary.json`.

3. gprMax architecture status was stale after GX-007 CPU 2D completion and
   explicit runtime-Python support.
   - Symptom: the status section still treated the first paired benchmark as a
     future step and did not mention `--gprmax-python`.
   - Source: `docs/gprmax_integration_architecture.md`.
   - Consequence: handoff readers could follow obsolete next-step guidance.
   - Remedy: updated current status, runtime Python command, GPU optional
     boundary, and next recommended Evidence task.

## P3 Backlog

- Keep GPU diagnostics separate from CPU Evidence packaging; the PowerShell
  toolchain failure should not block GX-007-EVIDENCE-002.
- Consider a general gprMax output conversion command after GX-007 has one or
  two stable model shapes; current `scene001` converter is deliberately narrow.
- Add richer synthetic full-reference metrics after Evidence archiving, rather
  than changing AutoTune scoring immediately.
- Defer GUI `仿真与验证` work until the backend Evidence loop is stable.
- Keep Workbench legacy fallback in place until a separate retirement plan
  exists.

## Changes Implemented

- Added `.gitignore` coverage for native gprMax outputs and campaign generated
  output folders.
- Added `check_staged_generated_artifacts()` to `scripts/preflight_check.py`.
- Added direct path classifier tests for generated artifact staging guard.
- Improved paired raw/background shape mismatch messages.
- Improved preview handling for invalid existing target response inputs.
- Updated gprMax architecture status and next-step text.

## Tests Added/Updated

- Added `tests/test_preflight_generated_artifact_guard.py`.
- Updated `tests/test_gprmax_campaign_pairing.py`.
- Updated `tests/test_gprmax_campaign_preview.py`.

## Commands Run

Initial baseline:

```text
git checkout main
git pull --ff-only origin main
git status --short
git log -1 --format=%H
git ls-remote origin main
python scripts/preflight_check.py
python -m pytest tests/test_gprmax_campaign_runner_execution.py tests/test_gprmax_campaign_pairing.py tests/test_gprmax_campaign_preview.py -q
```

Focused verification during implementation:

```text
python -m pytest tests/test_preflight_generated_artifact_guard.py -q
python -m pytest tests/test_gprmax_campaign_pairing.py tests/test_gprmax_campaign_preview.py -q
```

Final validation is recorded in the task closeout.

## Tests Run

Final test results are intentionally kept in the task closeout after the final
commands are run against the committed diff.

## Repository Hygiene Checks

- Generated native output suffixes `.out`, `.h5`, `.vti`, `.vtk`, `.vtu` are now
  ignored.
- Generated GX-007 conversion and pairing folders are ignored.
- Preflight now checks staged paths and fails when generated native outputs are
  staged.
- Curated fixtures and docs screenshots remain allowed.

## Generated Artifacts Excluded

No generated `.out`, `.h5`, `.vti`, `.vtk`, `.csv`, `.npy`, or `.png` artifacts
were added by this stabilization pass.

## Known Risks

- The generated artifact guard is path/suffix based. It intentionally does not
  block all `.csv`/`.png` files because the repo already contains curated
  fixtures and documentation screenshots.
- GPU readiness remains shell-dependent. The current reliable benchmark path is
  CPU, and GPU should remain optional until VS x64 shell diagnostics are green.
- The GX-007 CPU artifact is small-scale synthetic diagnostic evidence only.

## Deferred GPU Work

- `GX-RUN-GPU-DIAG-003`: repeat GX-007 GPU checks in VS x64 Native Tools Command
  Prompt, where `cl.exe`, `nvcc`, and the gprMax runtime Python are all known to
  be available.

## Deferred Evidence Work

- `GX-007-EVIDENCE-002`: selectively archive the CPU complete `936 x 21`
  diagnostic artifact in MyGPR-Evidence with the correct claim boundary.

## Claim Boundary

- This pass is repository stabilization and backend robustness work.
- It is not field validation.
- It is not AutoTune evaluation.
- It is not an AutoTune superiority claim.
- It is not a paper-candidate benchmark.
- It does not modify production AutoTune scoring, `processing_engine`, or
  motion-compensation scientific semantics.

## Recommended Next Tasks

1. `GX-007-EVIDENCE-002`: archive CPU complete `[936, 21]` 2D diagnostic artifact.
2. `GX-RUN-GPU-DIAG-003`: verify GX-007 GPU only in a VS x64 shell.
3. `METRIC-CLUTTER-001`: add MAE/MSE/PSNR/MS-SSIM or energy metrics for synthetic paired artifacts.
4. `GX-008-MINI-BENCHMARK-PLAN`: plan paper-inspired mini benchmark.
5. `GX-UI-001`: minimal read-only `仿真与验证` page.
