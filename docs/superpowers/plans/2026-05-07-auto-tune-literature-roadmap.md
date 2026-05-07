# Auto-Tune Literature And Roadmap Implementation Plan

> **For agentic workers:** Implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Produce a rigorous research report on whether UAV-GPR/GPR systems already provide MyGPR-like automatic parameter selection, then define a publishable long-term roadmap for MyGPR auto-tuning.

**Architecture:** Create one durable Markdown report in `docs/` and keep all conclusions source-backed. Separate evidence into UAV-GPR literature/software/patents, general GPR literature/software/patents, and MyGPR roadmap. Treat isolated automatic methods differently from an end-to-end multi-step processing auto-tune system.

**Tech Stack:** Web research with citations, repository inspection of `core/auto_tune.py`, `core/methods_registry.py`, and current gprMax validation artifacts, plus Markdown documentation.

---

### Task 1: Define Research Criteria

**Files:**
- Create: `docs/uav_gpr_auto_tune_prior_art_and_roadmap.md`

- [x] **Step 1: Write the report scope**

Add a scope section that defines "automatic parameter selection" as software or algorithms that choose GPR processing parameters from data, not merely default presets. Include these levels:

```text
L0 default presets only
L1 automatic single-parameter or single-step estimation
L2 automatic method family selection
L3 multi-step pipeline parameter optimization
L4 closed-loop benchmark/ground-truth-driven auto-tuning
```

- [x] **Step 2: Write search methodology**

Document that UAV-GPR is searched first, then general GPR, with separate categories for literature, software/products, patents, and adjacent methods.

### Task 2: UAV-GPR Prior Art

**Files:**
- Modify: `docs/uav_gpr_auto_tune_prior_art_and_roadmap.md`

- [x] **Step 1: Search UAV-GPR literature**

Use queries including:

```text
UAV ground penetrating radar automatic parameter selection
drone GPR automatic processing parameters
airborne GPR automatic gain background removal parameter optimization
UAV-GPR autonomous processing pipeline
```

Record every directly relevant source with title, year, contribution, parameter automation level, and whether it resembles MyGPR.

- [x] **Step 2: Search UAV-GPR software/products**

Search vendor and project pages for UAV-GPR processing automation. Record whether the software provides presets, assisted processing, or actual data-driven auto-tuning.

- [x] **Step 3: Search UAV-GPR patents**

Search patent databases/web results for UAV/drone/airborne GPR automatic data processing and parameter optimization. Record whether claims cover parameter tuning or only platform/control/detection.

### Task 3: General GPR Prior Art

**Files:**
- Modify: `docs/uav_gpr_auto_tune_prior_art_and_roadmap.md`

- [x] **Step 1: Search general GPR literature**

Look for automatic time-zero correction, dewow/window selection, background removal, AGC/gain selection, denoise rank/window selection, migration velocity selection, hyperbola fitting, and ML/Bayesian optimization.

- [x] **Step 2: Search general GPR software/products**

Review GPRPy, RGPR, GPR-SLICE, GSSI RADAN, Sensors & Software EKKO_Project, ReflexW/Sandmeier, and other visible packages for auto-processing or parameter automation claims.

- [x] **Step 3: Search patents**

Record patents that claim automatic GPR processing, object detection, or adaptive parameter/control, then classify their relevance to MyGPR auto-tune.

### Task 4: MyGPR Technical Roadmap

**Files:**
- Modify: `docs/uav_gpr_auto_tune_prior_art_and_roadmap.md`

- [x] **Step 1: Inspect current implementation**

Summarize weaknesses in `core/auto_tune.py` and `core/methods_registry.py`, including candidate bounds that can exceed trace count, weak data-shape awareness, per-step local scoring, limited UAV metadata use, and lack of global pipeline optimization.

- [x] **Step 2: Define target architecture**

Write a roadmap with milestones:

```text
M1 data-aware candidate constraints
M2 scenario-aware metrics and gprMax benchmark corpus
M3 pipeline-level optimization and Pareto scoring
M4 UAV metadata-aware motion compensation and quality model
M5 publishable validation protocol and ablation studies
```

- [x] **Step 3: Define paper/column contribution angle**

Frame the long-term contribution as UAV-GPR processing auto-tuning with simulation-ground-truth validation, expert baseline comparison, and sensor-metadata-aware processing.

### Task 5: Verify And Archive

**Files:**
- Modify: `docs/uav_gpr_auto_tune_prior_art_and_roadmap.md`

- [x] **Step 1: Validate citations**

Every external factual claim must have a source URL. Mark inference explicitly when a source supports only partial automation.

- [ ] **Step 2: Commit and archive**

Run:

```powershell
git status --short
git diff --check
git add docs\uav_gpr_auto_tune_prior_art_and_roadmap.md docs\superpowers\plans\2026-05-07-auto-tune-literature-roadmap.md
git commit -m "docs: add UAV-GPR auto-tune prior art roadmap"
python scripts\archive_checkpoint.py --summary "新增 UAV-GPR/GPR 自动选参前沿调研与 MyGPR 长期路线。"
git push
```

Expected: report is committed, archived, and pushed.
