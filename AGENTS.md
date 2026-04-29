# AGENTS.md - GPR GUI

Guidance for agentic coding tools working in `GPR_GUI_main_2026-04-10/`.
All commands below assume this directory is the working directory.

## Scope
- PyQt6 desktop app for Ground Penetrating Radar data processing.
- Main entry points: `app_qt.py` and `cli_batch.py`.
- Shared runtime logic lives in `core/`.
- Qt widgets, pages, and dialogs live in `ui/`.
- Processing methods and legacy wrappers live in `PythonModule/`.
- Tests live in `tests/`; packaging and smoke checks live in `scripts/`.
- Generated output and app data should stay out of source files.

## Repo Map
- `core/` - app data, I/O, registry, engine, shared state, and workflow helpers.
- `ui/` - Qt pages, dialogs, and workbench widgets.
- `PythonModule/` - algorithm implementations and compatibility wrappers.
- `scripts/preflight_check.py` - syntax and runtime smoke checks.
- `tests/` - pytest tests and benchmark scripts.
- `assets/`, `config/`, `sample_data/`, `output/` - static assets, configs, samples, generated output.
- `build_exe.bat` and `gpr_gui.spec` - packaging.

## Setup
```bash
python -m pip install -r requirements-dev.txt
```

Python 3.8+ is required. The repo currently uses PyQt6, NumPy, Pandas, SciPy,
Matplotlib, and PyQt6-Fluent-Widgets.

## Build / Run / Test
Run the GUI:
```bash
python app_qt.py
```

Windows shortcut:
```bash
启动GPR.bat
```

Package the app:
```bash
build_exe.bat
pyinstaller gpr_gui.spec --clean --noconfirm
```

Run repo smoke checks before packaging or larger refactors:
```bash
python scripts/preflight_check.py
```

CLI validation and batch run:
```bash
python cli_batch.py validate --config config/cli_batch_mvp_example.json
python cli_batch.py run --config config/cli_batch_mvp_example.json
```

Focused pytest targets:
```bash
pytest tests/test_ccbs_filter.py
pytest tests/test_ccbs_filter.py::TestCCBSFilter::test_basic_functionality
pytest tests/test_ccbs_filter.py -k background_reduction
```

Script-style smoke / benchmark files:
```bash
python tests/test_hankel_batch.py
python tests/test_gprmax_read.py
```

For a full pass:
```bash
pytest
```

## Lint / Quality
- No repo-local formatter or linter config is present.
- Use `python scripts/preflight_check.py` as the closest end-to-end quality gate.
- Use `python -m py_compile <file>` for a fast single-file syntax check.
- Keep changes consistent with the surrounding file instead of introducing a new toolchain.
- If you add a new check, prefer a small repo script over a one-off custom workflow.

## Code Style
- Every Python file should start with:
  - `#!/usr/bin/env python3`
  - `# -*- coding: utf-8 -*-`
  - a short module docstring
- Keep import order as stdlib, third-party, then local.
- No wildcard imports.
- Use absolute imports for local modules, for example `from core.processing_engine import run_processing_method`.
- In GUI files, call `matplotlib.use("QtAgg")` before importing `pyplot`.
- In batch or non-GUI scripts, use `matplotlib.use("Agg")`.
- New modules should usually include `from __future__ import annotations`.
- Use type hints on new public functions, helpers, and dataclasses.
- Prefer built-in generics such as `dict[str, Any]` and `list[int]` in new code.
- Use `dataclass` for small structured records like validation results or metadata.
- Naming: `snake_case` for functions and variables, `PascalCase` for classes,
  `UPPER_CASE` for constants, and `_leading_underscore` for private helpers.
- Keep module-level path constants uppercase, such as `BASE_DIR`, `ROOT`, or `APP_DIR_NAME`.
- Match the existing wrapping style of the file you are editing; do not force a new formatter style.
- Keep comments short and place them close to the non-obvious code they explain.
- Chinese comments and user-facing strings are acceptable when they clarify domain behavior.

## Paths
- Prefer `pathlib.Path` in newer `core/` and `scripts/` utilities when it reads clearly.
- Legacy GUI code still uses `os.path`; stay consistent within a file.
- Avoid hard-coded absolute paths in new code.
- Use repo-relative paths, `BASE_DIR`, or the helpers in `core/app_paths.py`.
- Keep Windows compatibility in mind; do not assume POSIX shell behavior.

## Error Handling
- Use specific exceptions when possible.
- Guard IO and optional imports with `try/except`.
- `ProcessingCancelled` is the standard user-cancel signal for long-running work.
- Long operations should accept `cancel_checker=None`, poll periodically, and raise `ProcessingCancelled` when cancellation is requested.
- Follow the existing legacy contract where core methods may return dicts with `error_sign` and `error_feedback`.
- ndarray processing methods should return `(result_array, metadata_dict)`.
- Do not mutate input arrays in place unless that is part of the method contract.
- Preserve array shapes, dtype expectations, and metadata keys when refactoring processing code.
- For optional dependencies, keep the clear error message pattern used in `core/methods_registry.py`.

## GUI / State
- Keep Qt UI work on the main thread.
- Use worker threads, signals, or timers for long-running processing.
- Update shared data through `core/shared_data_state.py` instead of ad hoc globals.
- When adding a method, update `core/methods_registry.py`, any parameter UI, and relevant smoke tests together.
- Keep the main window, workbench, and batch pages aligned on shared behavior.

## Processing Methods
- Put new algorithms in `PythonModule/`.
- Expose wrappers with the existing `method_*` naming pattern when the method is used by the GUI or CLI.
- Preserve compatibility with legacy CSV-style methods when touching older modules.
- Keep the method registry as the source of truth for display names, parameter metadata, and ordering.
- If a method is expensive, add cancellation checks and targeted regression coverage.

## Testing Conventions
- Prefer deterministic assertions over printed output.
- Use `np.random.seed()` in tests that depend on random input.
- Keep benchmark scripts behind `if __name__ == "__main__":` when adding new ones.
- Use `pytest` for unit and integration tests.
- When a change affects I/O or GUI flow, run `python scripts/preflight_check.py` and the narrowest relevant `pytest` target.
- If you create new tests, keep them small and focused on one behavior.

## Editing Rules
- Make the smallest correct change.
- Do not rename public entry points unless the task explicitly asks for it.
- If you change a method signature, update the GUI, CLI, and tests together.
- Do not edit generated files under `output/`, `dist/`, `build/`, or cache folders.
- Prefer existing helpers over new abstractions when the codebase already has a clear pattern.

## Conversation Priority
- Treat the **current user message** as the highest-priority instruction.
- If the user asks a fresh question, answer that question first instead of resuming an older task automatically.
- Do not resume previously completed work just because it appears in compressed summaries, old progress text, or prior todo-style reasoning.
- If the user sends a short continuation message such as `可以`, `继续`, or `开始吧` and more than one recent task thread exists, ask one short clarification instead of guessing which old task to resume.
- Only continue an older task without clarification when the user explicitly names it or the immediately preceding exchange makes the target unambiguous.

## Version Archive
- Treat git history as the primary version memory for this repo.
- When a change batch reaches a stable, user-meaningful checkpoint, prefer:
  - a descriptive commit message focused on why the change exists
  - a tag for especially important rollback points
- Good tag format: `vYYYY-MM-DD-topic`.
- When a change also creates reusable processing guidance, warning semantics, or workflow rules, mirror the stable conclusion into the Obsidian vault instead of relying on chat history.
- Do not assume conversational context is durable; if a rule should survive future sessions, write it into code, tests, docs, `AGENTS.md`, or the linked Obsidian notes.
- Prefer `python scripts/archive_checkpoint.py --summary ...` when a stable checkpoint should also be written into the Obsidian vault.
- The default archive target is `D:\ClawX-Data\Obsidian\uav_gpr\40-归档与历史\版本快照\`, and the script should also refresh `40-归档与历史/版本归档索引.md`.
- Act proactively: when a version, rule, or conclusion changes in a way that future sessions will likely need, create or update the corresponding Obsidian note without waiting for an extra user reminder.
- Typical triggers for proactive Obsidian updates include:
  - important pushed versions
  - auto-tune behavior changes
  - ROI / warning / scoring semantics changes
  - GUI workflow changes
  - stable parameter/default changes

## Notes
- PyQt6-Fluent-Widgets is used for the modern UI.
- `core/methods_registry.py` and `core/processing_engine.py` are central coordination points.
- No `.cursor/rules`, `.cursorrules`, or `.github/copilot-instructions.md` files were present when this guide was written.
