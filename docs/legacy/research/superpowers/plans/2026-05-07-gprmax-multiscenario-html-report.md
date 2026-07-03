# GPRMAX Multi-Scenario HTML Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable gprMax validation workflow that runs several simple forward-model scenarios, compares manual baseline parameters with MyGPR auto-tuned parameters step by step, and emits a static HTML research report.

**Architecture:** Add a self-contained benchmark/report script under `scripts/gprmax_benchmark/` so the research workflow can run without changing GUI behavior. The script owns scenario definitions, gprMax command construction, output conversion, per-step processing capture, visualization, metric summarization, and HTML rendering. Keep existing one-scenario smoke files intact for compatibility.

**Tech Stack:** Python 3.10+, gprMax 3.1.7, NumPy, Matplotlib Agg, MyGPR `core.auto_tune`, `core.processing_engine`, `core.methods_registry`, and pytest.

---

### Task 1: Multi-Scenario Report Runner

**Files:**
- Create: `scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`
- Modify: none
- Test: `python -m py_compile scripts/gprmax_benchmark/gprmax_multi_scenario_report.py`

- [ ] **Step 1: Define scenario records**

Create a `ScenarioDefinition` dataclass with fields for `scenario_id`, `label`, `description`, `domain_m`, `dx_m`, `total_time_ns`, `trace_step_m`, `runs_default`, `model_in_text`, `materials`, `targets`, and `analysis_roi_hint`.

- [ ] **Step 2: Add three simple gprMax scenes**

Implement `build_scenario_definitions()` returning:

```python
{
    "cylinder_single_v1": ScenarioDefinition(...),
    "cylinder_double_v1": ScenarioDefinition(...),
    "layered_interface_v1": ScenarioDefinition(...),
}
```

The scenes must be simple enough to run repeatedly: one PEC cylinder, two PEC cylinders, and a two-layer soil interface. Each scene includes a human-readable true-structure description for the HTML report.

- [ ] **Step 3: Build gprMax command helper**

Implement `build_gprmax_command()` with explicit support for `-n`, `--geometry-fixed`, `-mpi`, and `-gpu`. The command helper should not force MPI or GPU; it only adds them when the CLI requests them.

- [ ] **Step 4: Convert `.out` files into MyGPR BScan CSV**

Use `core.gpr_io.read_gprmax_out()` on the numerically first scenario `.out` file. Write `mygpr_bscan.csv`, `scenario.json`, `ground_truth.json`, and `structure.png` under the run output directory.

- [ ] **Step 5: Run manual and automatic branches step by step**

Resolve pipeline and manual parameters from `RECOMMENDED_RUN_PROFILES["uav_gpr_experience_baseline_v1"]`. For each method, capture:

```python
{
    "method_key": "dewow",
    "method_name": "Dewow",
    "manual_input": np.ndarray,
    "auto_input": np.ndarray,
    "manual_output": np.ndarray,
    "auto_output": np.ndarray,
    "manual_params": {...},
    "auto_params": {...},
    "auto_tune_summary": {...},
    "manual_warnings": [...],
    "auto_warnings": [...],
}
```

- [ ] **Step 6: Save locked-scale BScan images**

For each step, render `manual_before`, `auto_before`, `manual_after`, and `auto_after` images into an `assets/` directory. Use the same symmetric percentile scale per step so visual differences are comparable.

- [ ] **Step 7: Render static HTML**

Write `index.html` containing:

- scenario overview and true geologic structure
- gprMax command and acceleration settings
- final manual vs auto metric summary
- selected manual and automatic parameters for every step
- per-step BScan image panels before and after each method
- notes explaining that the ground truth is simulation-derived and intended for auto-tune validation

### Task 2: Tests

**Files:**
- Create: `tests/test_gprmax_multi_scenario_report.py`

- [ ] **Step 1: Test scenario definitions**

Assert that at least three scenarios are available, each has non-empty model text, target/structure metadata, and a stable `scenario_id`.

- [ ] **Step 2: Test command construction**

Assert that `build_gprmax_command(..., mpi=4, gpu=["0"])` includes `-mpi 4 -gpu 0`, and that default command construction omits MPI/GPU.

- [ ] **Step 3: Test HTML rendering contract**

Build a tiny fake report payload and assert the generated HTML contains the required Chinese headings: `真实地质结构`, `人工选参`, `自动选参`, `逐步骤 BScan 对比`, and `gprMax 运行设置`.

### Task 3: Real Run and Verification

**Files:**
- Generated: `output/gprmax_multi_scenario_reports/<timestamp>/index.html`
- Generated: `output/gprmax_multi_scenario_reports/<timestamp>/summary.json`

- [ ] **Step 1: Check local gprMax acceleration support**

Run:

```powershell
E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe -m gprMax --help
```

Use `--geometry-fixed` by default. Use `-mpi` or `-gpu` only if the local environment has working `mpi4py` or `cupy`.

- [ ] **Step 2: Run real multi-scenario report**

Run:

```powershell
python scripts\gprmax_benchmark\gprmax_multi_scenario_report.py --gprmax-root E:\gprMax\gprMax-v.3.1.7 --runs 36 --geometry-fixed
```

Expected: all selected scenarios produce converted BScan CSV files and a static HTML report.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
python -m pytest tests\test_gprmax_multi_scenario_report.py tests\test_gprmax_benchmark_smoke.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Run repo smoke gate**

Run:

```powershell
python scripts\preflight_check.py
```

Expected: preflight completes without errors.

- [ ] **Step 5: Commit and push**

Run:

```powershell
git add scripts\gprmax_benchmark\gprmax_multi_scenario_report.py tests\test_gprmax_multi_scenario_report.py docs\superpowers\plans\2026-05-07-gprmax-multiscenario-html-report.md
git commit -m "feat: add multi-scenario gprMax HTML benchmark"
git push
```

Expected: branch contains the reusable report runner and tests; generated `output/` artifacts remain local unless intentionally tracked.
