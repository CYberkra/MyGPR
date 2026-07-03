# gprMax Dataset Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect gprMax-produced manifest folders to MyGPR ground-truth metrics and AutoTune pipeline smoke validation.

**Architecture:** Add `core/gprmax_dataset_contract.py` as a narrow adapter layer. It reads a gprMax dataset manifest, resolves relative sidecar paths, loads the primary `.out` through existing `read_gprmax_out()`, converts zero-based closed YAML ROIs into MyGPR Python-slice ROIs, and returns a package that can be passed directly into `compute_ground_truth_metrics()` and `run_auto_tune_pipeline()`.

**Tech Stack:** Python 3.10+, JSON, PyYAML, h5py-backed existing gprMax reader, NumPy, pytest, existing `core.gprmax_truth_metrics` and `core.auto_tune_pipeline`.

---

### Task 1: Contract Reader And ROI Adapter

**Files:**
- Create: `core/gprmax_dataset_contract.py`
- Modify: `requirements-dev.txt`
- Test: `tests/test_gprmax_dataset_contract.py`

- [ ] Write tests for manifest path resolution, YAML closed-interval ROI conversion, and `.out` loading.
- [ ] Implement manifest JSON loading with relative path resolution.
- [ ] Implement `ground_truth.yaml` loading through PyYAML.
- [ ] Convert `target_roi.sample_range=[s0,s1]` and `trace_range=[t0,t1]` to MyGPR `time_end_idx=s1+1`, `dist_end_idx=t1+1`.
- [ ] Return a dataclass package with data, header info, trace metadata, raw manifest, metadata sidecar, converted ground truth, and resolved paths.

### Task 2: AutoTune Smoke Contract

**Files:**
- Test: `tests/test_gprmax_dataset_contract.py`

- [ ] Build a minimal temporary gprMax HDF5 `.out`.
- [ ] Build a matching `*_manifest.json`, metadata JSON, and `ground_truth.yaml`.
- [ ] Load the package.
- [ ] Run `compute_ground_truth_metrics(raw, raw, package.ground_truth)`.
- [ ] Run `run_auto_tune_pipeline(..., ground_truth=package.ground_truth, pipeline=["dewow"])`.
- [ ] Assert truth metrics and AutoTune summary contain the converted ground truth.

### Task 3: Verification And Checkpoint

**Files:**
- Modify: `AGENTS.md` only if project workflow instructions need updating.

- [ ] Run `python -m py_compile core\gprmax_dataset_contract.py`.
- [ ] Run `pytest tests\test_gprmax_dataset_contract.py tests\test_gprmax_truth_metrics.py tests\test_auto_tune_pipeline.py -q`.
- [ ] Run `python scripts\preflight_check.py`.
- [ ] Commit on `codex/research-gprmax-autotune` only, using `scripts/git_checkpoint.py`.
