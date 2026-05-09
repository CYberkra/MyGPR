# GPRMAX Cylinder Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first clean MyGPR GPRMAX benchmark package, `cylinder_single_v1`, with scenario metadata, gprMax input, ground-truth manifest, MyGPR CSV conversion, and auto-tune evidence export.

**Architecture:** Keep benchmark-generation logic in a standalone script under `scripts/gprmax_benchmark/` so it does not depend on the old `E:\gprMax\gprMax-v.3.1.7` experiment tree. The script writes deterministic files under `sample_data/gprmax_benchmarks/cylinder_single_v1/` and can optionally convert a supplied gprMax `.out` into MyGPR CSV. Tests use small synthetic HDF5 `.out` fixtures instead of running gprMax.

**Tech Stack:** Python 3.10+, NumPy, h5py for `.out` fixtures, Matplotlib Agg for preview/evidence PNGs, pytest, existing `core.gpr_io.read_gprmax_out`, existing `core.auto_tune_comparison_export`.

---

### Task 1: Benchmark Contract Tests

**Files:**
- Create: `tests/test_gprmax_benchmark_package.py`
- Create later: `scripts/gprmax_benchmark/generate_cylinder_single_v1.py`

- [ ] **Step 1: Write failing package-generation test**

The test imports `generate_package` and asserts that it writes:

```python
expected = {
    "scenario.json",
    "model.in",
    "ground_truth.json",
    "mygpr_bscan.csv",
    "preview.png",
    "README.md",
}
```

It should assert `ground_truth.json["schema"] == "mygpr_gprmax_ground_truth_v1"`, one target of type `hyperbola`, and a bounded ROI.

- [ ] **Step 2: Write failing `.out` conversion test**

Use `h5py` to create three tiny `.out` files with `rxs/rx1/Ez`, then call the benchmark converter and assert the output CSV equals `np.column_stack(traces)`.

- [ ] **Step 3: Run test and confirm failure**

Run: `python -m pytest tests\test_gprmax_benchmark_package.py -q`

Expected: FAIL because `scripts.gprmax_benchmark.generate_cylinder_single_v1` does not exist.

### Task 2: Generator Implementation

**Files:**
- Create: `scripts/gprmax_benchmark/generate_cylinder_single_v1.py`
- Create: `scripts/gprmax_benchmark/__init__.py`

- [ ] **Step 1: Implement deterministic scenario writer**

The module should expose:

```python
def generate_package(package_dir: Path | str = DEFAULT_PACKAGE_DIR, *, raw_out_path: Path | str | None = None) -> PackageResult:
    ...
```

It writes a gprMax-compatible `model.in` based on the local `cylinder_Bscan_2D.in` pattern, but with MyGPR-owned parameters and clean metadata.

- [ ] **Step 2: Implement synthetic fallback B-scan**

If `raw_out_path` is not supplied, write a deterministic synthetic B-scan containing a known hyperbola. This keeps tests and docs independent from actually running gprMax.

- [ ] **Step 3: Implement `.out` conversion**

If `raw_out_path` is supplied, call `core.gpr_io.read_gprmax_out` and write the resulting `mygpr_bscan.csv`.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests\test_gprmax_benchmark_package.py tests\test_gprmax_read.py -q`

Expected: PASS.

### Task 3: Auto-Tune Evidence Smoke

**Files:**
- Modify: `tests/test_gprmax_benchmark_package.py`

- [ ] **Step 1: Add comparison-export smoke test**

Generate the package in `tmp_path`, load `mygpr_bscan.csv`, run `run_auto_tune_comparison(..., pipeline=["dewow"], search_mode="fast")`, and export via `export_auto_tune_comparison_artifacts(...)`.

- [ ] **Step 2: Assert GPRMAX metadata is carried into evidence**

Assert the summary JSON contains `input_ref` pointing at `mygpr_bscan.csv`, and the exported report exists.

- [ ] **Step 3: Run export-adjacent tests**

Run: `python -m pytest tests\test_gprmax_benchmark_package.py tests\test_auto_tune_comparison_export.py -q`

Expected: PASS.

### Task 4: Documentation

**Files:**
- Modify: `docs/gprmax_auto_tune_validation_plan.md`
- Create generated package files under `sample_data/gprmax_benchmarks/cylinder_single_v1/`

- [ ] **Step 1: Generate the repository sample package**

Run: `python scripts\gprmax_benchmark\generate_cylinder_single_v1.py`

- [ ] **Step 2: Update docs with actual commands**

Document:

```powershell
python scripts\gprmax_benchmark\generate_cylinder_single_v1.py
python -m pytest tests\test_gprmax_benchmark_package.py -q
```

- [ ] **Step 3: Explain the boundary**

State that the bundled sample is deterministic and contract-oriented; a later optional smoke can run real gprMax from `E:\gprMax\gprMax-v.3.1.7`.

### Task 5: Verification and Archive

**Files:**
- No additional source files unless tests reveal a bug.

- [ ] **Step 1: Compile changed modules**

Run: `python -m py_compile scripts\gprmax_benchmark\generate_cylinder_single_v1.py`

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests\test_gprmax_benchmark_package.py tests\test_gprmax_read.py tests\test_auto_tune_comparison_export.py -q`

- [ ] **Step 3: Run broader gates**

Run:

```powershell
python -m pytest -q
python scripts\preflight_check.py
git diff --check
```

- [ ] **Step 4: Archive checkpoint**

Run `python scripts\archive_checkpoint.py` with a summary for the first GPRMAX benchmark package.
