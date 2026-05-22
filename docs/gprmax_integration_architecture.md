# gprMax Integration Architecture and UI Specification

Status: design specification only  
Scope: MyGPR source repository only  
Target UI area: optional `仿真与验证` control page  
Implementation status: GX-RUN-001 implemented (backend dry-run only)  
Evidence repository: separate; not modified by this specification

## 1. Executive Summary

MyGPR should integrate gprMax support as an optional simulation and validation workspace inside the existing MyGPR application. The proposed user-facing entry is a fifth main control page named `仿真与验证`.

This should not become a separate GUI application. It should also not be mixed into `日常处理`, AutoTune production scoring, or `processing_engine`. The first implementation should be a backend campaign runner and dry-run validator. The GUI should be added only after the backend contract is stable.

The intended role of the gprMax module is to manage long-running simulation campaigns, paired synthetic benchmarks, and Evidence-ready outputs. It is a validation and benchmark source, not a dependency of normal MyGPR data processing.

Key architectural rule:

```text
gprMax integration must be optional and removable.
Core MyGPR processing, import, AutoTune, display, and export must not depend on gprMax modules at import time.
```

## 2. Current MyGPR UI Baseline

The current MyGPR `main` branch does not use a dark left-sidebar Workflow Studio layout yet. The actual current UI structure is:

- The main window uses `QTabWidget` control pages:
  - `日常处理`
  - `调参与实验`
  - `显示与对比`
  - `质量与导出`
- The right side contains the Matplotlib B-scan canvas.
- The main canvas area includes:
  - Matplotlib toolbar
  - processing lineage label
  - coordinate label
  - empty-state card before data import
- The bottom/right runtime drawer shows:
  - global log
  - quality summary
- Workbench is a separate `content_stack` page and remains a Legacy fallback.

Therefore, gprMax should be added as a fifth control page:

```text
仿真与验证
```

It should follow the same visual grammar as the current pages: title, hint text, callout flow, segmented subpages, task tables, logs, and preview panels.

## 3. Proposed User-Facing UI

### 3.1 Page placement

Add a fifth control tab/page beside the existing pages:

```text
日常处理 | 调参与实验 | 显示与对比 | 质量与导出 | 仿真与验证
```

This page should not replace Workbench and should not introduce a new global sidebar.

### 3.2 Top page structure

The `仿真与验证` page should contain:

- Page title: `仿真与验证 / gprMax`
- Hint text explaining that this page manages long-running gprMax campaigns and paired benchmarks.
- Flow callout:
  - `① 场景配置`
  - `② 队列运行`
  - `③ 结果检查`
  - `④ Evidence`

The first version should not attempt to be a visual gprMax model editor. It should manage existing `model.in`, `materials.txt`, ROI annotations, task queues, and outputs.

### 3.3 Inner segmented subpages

The page should use an inner segmented layout:

- `任务队列`
- `场景管理`
- `环境配置`
- `运行记录`

#### 任务队列

Main purpose: manage campaign tasks and long-running execution state.

Task queue table fields:

- `task_id`
- `campaign_id`
- `scene_id`
- `variant`
- `status`
- `progress`
- `runtime`
- `output_path`
- `warnings`

Recommended task statuses:

- `pending`
- `ready`
- `running`
- `success`
- `failed`
- `skipped`
- `invalid`

#### 场景管理

Main purpose: inspect paired scene definitions.

Suggested fields:

- `scene_id`
- `description`
- `raw_model`
- `background_model`
- `materials`
- `target_roi`
- `tags`
- `pair_validation_status`

#### 环境配置

Main purpose: configure gprMax executable/environment and output roots.

Suggested fields:

- gprMax executable or environment name
- output root
- maximum parallel tasks
- timeout policy
- resume policy
- log retention policy

#### 运行记录

Main purpose: show campaign-level logs and run history.

Suggested content:

- campaign load log
- dry-run validation log
- task execution stdout/stderr summary
- failed task reason
- resume/retry records

### 3.4 Right-side canvas usage

The current right-side Matplotlib canvas should be reused for preview, not replaced by a separate viewer.

Preview modes:

- `raw_with_target`
- `background_only`
- `target_response = raw - background`
- diff / QC view

The preview should be read-only and diagnostic. It must not imply that simulation data has been imported into the normal MyGPR processing chain unless the user explicitly imports it.

### 3.5 Runtime drawer usage

The existing runtime drawer pattern should be reused for:

- gprMax stdout/stderr
- QC
- Evidence
- metrics

This avoids long logs permanently occupying the main canvas area.

## 4. Backend Architecture

The backend should be independent from the Qt UI and removable.

Proposed structure:

```text
core/gprmax_campaign/
  __init__.py
  schema.py
  campaign_loader.py
  runner.py
  manifest.py
  pairing.py
  preview.py
  metrics.py

scripts/gprmax_campaign_runner.py

ui/gui_gprmax_simulation_page.py
```

`ui/gui_gprmax_simulation_page.py` is a future implementation target, not part of this specification task.

Required dependency direction:

```text
GUI -> core/gprmax_campaign backend
CLI -> core/gprmax_campaign backend
core/gprmax_campaign -> no app_qt import
processing_engine -> no gprmax_campaign import
AutoTune production scoring -> no gprmax_campaign import
```

The GUI and CLI should call the same backend functions. The backend must not import `app_qt.py`.

The backend should support dry-run validation before any gprMax process is launched.

## 5. Campaign YAML Schema

A campaign YAML file should define a reproducible simulation campaign. The schema should be small in the first version.

Required top-level fields:

- `campaign_id`
- `output_root`
- `gprmax_executable` or environment name
- `scenes`

Required scene fields:

- `scene_id`
- `description`
- `raw_model`
- `background_model`
- `materials`
- `target_roi`
- `expected_outputs`

Optional scene fields:

- `tags` (default `[]` when omitted)

Example:

```yaml
campaign_id: GX-007_paired_background_benchmark
output_root: D:/CDUT-UavGPR-Controller/MyGPR-Evidence/gprmax/GX-007
gprmax_executable: gprMax

scenes:
  - scene_id: single_pipe_rough_surface
    description: single shallow pipe under rough surface clutter
    raw_model: models/single_pipe_rough/raw_with_target.in
    background_model: models/single_pipe_rough/background_only.in
    materials: models/single_pipe_rough/materials.txt
    target_roi: annotations/single_pipe_rough_roi.json
    expected_outputs:
      - raw_with_target
      - background_only
      - target_response
    tags:
      - paired_benchmark
      - background_suppression
      - target_preservation
```

The first backend task should validate this YAML and report missing files, malformed fields, invalid scene pairs, and unwritable output directories.

## 6. Paired Benchmark Contract

The paired benchmark is the core scientific contract.

Each scene contains two gprMax runs:

```text
raw_with_target
background_only
```

The two runs must share:

- domain
- `dx`, `dy`, `dz`
- time window
- waveform
- source / receiver configuration
- scan path
- output sampling
- material background

The only intentional difference is:

```text
target present only in raw_with_target
```

The derived target-response reference is:

```text
target_response = raw_with_target - background_only
```

Validation rules:

- Shape mismatch is a hard failure.
- Missing raw/background output is a hard failure.
- Missing ROI is a high-risk warning for visualization, but a hard failure for ROI-based metrics.
- Metadata mismatch is a high-risk warning or hard failure depending on field severity.
- If the background scene changes material background, scan path, waveform, or output sampling, the pair should be invalid.

This contract is required before using the benchmark for AutoTune evaluation.

## 7. Evidence / Output Contract

Each completed scene should produce an Evidence-ready package.

Required outputs:

- copied raw model `.in` file
- copied background model `.in` file
- copied materials file
- copied ROI annotation if available
- run manifest JSON
- gprMax version
- command line
- runtime
- return code
- raw output metadata
- background output metadata
- converted CSV if available
- raw preview PNG
- background preview PNG
- target_response preview PNG
- metrics CSV/JSON
- report markdown/html
- claim boundary note

Large simulation outputs should not be committed to the MyGPR source repository.

Policy:

```text
MyGPR stores code, docs, and tests.
Large .out/.h5 simulation outputs live under external output_root or policy-controlled MyGPR-Evidence paths.
```

The Evidence package must clearly distinguish:

- synthetic gprMax benchmark
- real field data
- hybrid data
- diagnostic proxy metric
- full-reference metric
- paper-candidate figure

## 8. Metrics Plan

### 8.1 Synthetic paired data

For synthetic paired data with known target response, full-reference metrics are allowed.

Candidate metrics:

- MAE
- MSE
- PSNR
- SSIM / MS-SSIM if available
- target ROI energy preservation
- background ROI suppression
- false-positive energy outside ROI
- shape checks
- hash checks

Recommended interpretation:

- Background suppression should not be scored only by clutter reduction.
- Target response preservation must be measured separately.
- Large-window background suppression should be flagged if it suppresses target or interface energy.

### 8.2 Real no-prior field data

For real no-prior field data, ground truth is unavailable unless explicit target ROI or independent annotation exists.

Allowed metric class:

- proxy metrics
- warning flags
- manual review labels
- no-prior advisory labels

Not allowed:

- ground-truth target correctness claim
- field-performance superiority claim
- automatic proof that a detected structure is real

## 9. Integration Boundaries and Removability

The gprMax integration must remain optional and removable.

Lazy import rule:

```text
Only import gprMax campaign modules when opening 仿真与验证 or running the CLI runner.
```

To remove the integration later, the expected deletion set should be limited to:

- `core/gprmax_campaign/`
- `ui/gui_gprmax_simulation_page.py`
- `scripts/gprmax_campaign_runner.py`
- `tests/test_gprmax_campaign_*.py`
- fifth tab registration in the main window
- related docs, archived if desired

Removing the gprMax module must not break:

- daily processing
- AutoTune
- motion compensation
- import/export
- display
- normal Evidence export
- Workbench fallback

Anti-patterns to avoid:

- top-level `app_qt.py` import of gprMax campaign modules
- `processing_engine` calling gprMax
- AutoTune production scoring requiring gprMax artifacts
- storing large `.out` / `.h5` files in MyGPR source repo
- mixing gprMax manifest schema into generic import schema

## 10. Implementation Roadmap

Recommended sequence:

### GX-RUN-001: backend campaign loader and dry-run validator

Goal:

- load campaign YAML
- validate required fields
- check file existence
- check output directory writability
- identify raw/background pairs
- report ready / invalid tasks

No gprMax execution yet.

### GX-RUN-002: single local task execution wrapper

Goal:

- run one gprMax task through a subprocess wrapper
- capture stdout/stderr
- capture return code
- write run manifest
- support safe cancellation and timeout

### GX-RUN-003: paired raw/background validation and target_response generation

Goal:

- verify raw/background output compatibility
- convert outputs to CSV or internal arrays if available
- compute target_response
- generate previews and basic metrics

### GX-UI-001: minimal UI page

Goal:

- add `仿真与验证` as the fifth control tab
- load campaign YAML
- display task queue
- show logs
- show previews
- avoid blocking Qt main thread

### GX-007: first paired benchmark dataset

Goal:

- create the first small paired gprMax benchmark set
- store Evidence-ready reports and manifests
- avoid overclaiming paper-level validation

### AT-022: AutoTune background suppression evaluation using GX-007

Goal:

- evaluate background suppression AutoTune candidates on paired synthetic ground-truth data
- separate clutter reduction from target preservation
- avoid field-performance overclaims

## 11. Risks

### P1 risks

- Long-running gprMax processes can freeze the GUI if not isolated.
- Pair mismatch can invalidate `target_response`.
- Output files may be too large for repository storage.
- gprMax version/environment drift can affect reproducibility.
- A GUI-first implementation may hide backend reproducibility problems.

### P2 risks

- UI may become crowded if too many controls are added at once.
- Campaign YAML schema may evolve after the first real benchmark.
- MyGPR-Evidence cleanup remains a separate task.
- Future hybrid data design may need a separate schema.

## 12. Acceptance Criteria for This Specification

This document is considered sufficient if:

- It references the real current MyGPR UI structure, not a fictional dark sidebar UI.
- It defines gprMax as an optional fifth control page named `仿真与验证`.
- It defines backend / CLI / UI separation.
- It defines the campaign YAML schema.
- It defines the paired benchmark contract.
- It defines the Evidence/output contract.
- It defines metrics boundaries for synthetic and real no-prior data.
- It defines removability boundaries.
- It does not claim the GUI or runner is already implemented.
- It does not claim gprMax benchmarks already exist.

## 13. Current Status

As of the current source baseline:

- `core/gprmax_campaign/` backend loader/validator is implemented for dry-run.
- `scripts/gprmax_campaign_runner.py` dry-run CLI is implemented.
- gprMax execution mode is not implemented yet in GX-RUN-001.
- `仿真与验证` page is not implemented yet.
- No new gprMax benchmark is created by this document.
- MyGPR-Evidence is not modified by this document.

Current dry-run command:

```text
python scripts/gprmax_campaign_runner.py --campaign path/to/campaign.yaml --dry-run
```

The next recommended task is:

```text
GX-RUN-002: single local gprMax task execution wrapper (backend-only)
```
