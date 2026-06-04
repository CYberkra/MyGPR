## 0.8.60 - Export and report non-blocking path audit

- Added a reusable background export worker for long-running GUI export actions.
- Moved AutoTune comparison, replay evidence ZIP and UAV georeference exports to background tasks with synchronous fallback for tests.
- Added export timing instrumentation for report figure, HTML and sidecar writes.
- Added batch sidecar write helpers for report JSON/text artifacts without changing report schemas.

## 0.8.59 - Display/compare performance smoothing

- Added a processing-lineage stepper signature cache so high-frequency plot refreshes do not rebuild step buttons when the visible lineage state is unchanged.
- Optimized compare-panel shared colour limit calculation to avoid large temporary concatenations in normal symmetric display mode.
- Added compare combo update skipping when snapshot labels are unchanged, reducing display/compare page signal churn.
- Added performance monitor keys for lineage stepper rebuild/skip and compare combo refresh/skip paths.

## 0.8.58 - Import and memory-copy performance audit

- Added lightweight I/O performance helpers for array memory summaries, CSV import context, conservative CSV dtype policy, and float32 sanitisation.
- Reduced matrix-only CSV import memory pressure by reading numeric chunks as float32 when no airborne header or sidecars require precision-preserving fields.
- Added import memory summaries to loaded header info and user logs for auditability.
- Throttled loading dialog progress emissions to reduce UI churn during large chunked imports.
- Kept processing algorithms, AutoTune scoring, candidate generation, gprMax linkage, and Evidence schema unchanged.

## 0.8.57 - AutoTune runtime refresh smoothing

- Throttled AutoTune progress UI updates to reduce QLabel/QProgressBar churn during candidate sweeps.
- Added signature-based QTableWidget refresh skips and batched table writes for AutoTune ranking/candidate/trial panels.
- Deferred hidden advanced audit table/text refresh until the advanced panel is opened.
- Added performance counters for AutoTune page refresh, table refresh, progress request and progress flush paths.
- No scoring formula, candidate generator, algorithm output, gprMax linkage, or Evidence schema changes.

## 0.8.56 - Render interaction coalescing and B-scan gesture smoothing

- Added a coalesced main-canvas draw scheduler for high-frequency B-scan interactions.
- Routed pan, ROI preview, slider-compare and transient overlay redraws through bounded draw requests.
- Replaced hover coordinate nearest-point scans with monotonic-axis search to reduce mouse-move overhead.
- Kept all changes display-only; processing arrays, AutoTune scoring, candidate generation and Evidence schema are unchanged.

## 0.8.55 - Performance audit foundation and display-cache safeguards

- Added lightweight performance-monitor instrumentation for render-path timing.
- Added display-only prepared-view and vmin/vmax caches keyed by data revision and display settings.
- Batched visible runtime log widget updates to reduce QTextEdit churn during long operations.
- Added performance audit baseline, hotspot report, and first-pass optimization plan.

## 0.8.54 - AutoTune editable bounded workflow UI

- Removed the visible top-level AutoTune report button while keeping hidden compatibility wiring for legacy export paths.
- Reordered the workflow table to 步骤 / 参数 / 处理方式 / 说明 and made the parameter column editable by double click.
- Added bounded workflow customization state: parameter overrides, workflow order overrides, manual review flags, and payload metadata for recipe execution.
- Enabled row drag behavior on the workflow table with validation guards for unsafe order changes; scoring formulas, algorithms, candidate generation, and Evidence schemas remain unchanged.

## 0.8.53 - AutoTune page compact ROI and tab cleanup

- Simplified AutoTune ROI row labels: ROI / 全图 / 自动 / 手动 with compact 自动 and 框选 buttons.
- Removed the visible recommendation card block from the AutoTune page to reduce text density.
- Removed the duplicate 参数 primary tab; the primary AutoTune tabs are now 流程 / 候选 / 说明.
- Kept hidden compatibility buffers for existing report/test update paths; no scoring, algorithm, candidate generator, or Evidence format changes.

## 0.8.52 - AutoTune page workflow-first UI cleanup

- Reworked the AutoTune tab around target goal, focus area, recommendation card, and four primary tabs: workflow, parameters, candidates, and notes.
- Replaced the ambiguous manual ROI checkbox with an explicit checkable “开始图上框选” action button and renamed automatic ROI to “自动建议区域”.
- Removed the duplicate top-level run action; the recommendation card now owns the single “应用并运行推荐流程” action.
- Moved ROI coordinates and report/audit details into the advanced settings panel while keeping report/export backend compatibility.
- Added full-workflow parameter and candidate summary tables without changing scoring, algorithms, candidate generation, or Evidence schemas.

## 0.8.51 - AutoTune V1 manifest and trial export closure

- Added AutoTune V1 candidate-space metadata to exported trial tables.
- Added candidate_space_hash/profile/config/recipe ids and manual_review_required to evidence manifests.
- Added scoring-boundary and claim-boundary section to AutoTune comparison reports.
- Preserved V1 candidate metadata through compact AutoTune comparison summaries.

# Changelog

## 0.8.50 - AutoTune V1 candidate-space backend hookup

- Connected the AutoTune V1 bounded candidate generator to the background candidate runner through an opt-in backend path used by the AutoTune recommendation page.
- Added V1 candidate-space metadata to background trial rows: `candidate_space_hash`, profile id, config version, recipe ids, candidate id, candidate parameters and candidate warnings.
- Propagated candidate-space context into workflow recipe rows and `autotune_scoring_record` so future trial tables/manifests can verify the exact bounded search space.
- Added executable support for V1 sliding mean/median background candidates while preserving the legacy runner interface and tests.
- Kept scoring formulas, UI layout, gprMax/GPR Scene Studio contracts and core processing algorithms unchanged.

## 0.8.49 - AutoTune V1 bounded candidate generator

- Added `core/autotune_candidate_generator.py` for bounded AutoTune V1 candidate generation from fixed tables plus lightweight data-adaptive diagnostics.
- Generates profile-aware background, Dewow, bandpass, gain, denoise and migration candidate declarations without executing processing algorithms or changing scoring defaults.
- Added conservative profile caps for interface, landslide, wet-zone and deep-weak-reflector profiles, including SVD rank limiting and display-only AGC exclusion flags.
- Added deterministic `candidate_space_hash` output for manifests, trial tables and future Evidence export.
- Added backend contract tests for stable hashes, metadata-only generation, display-only exclusion and landslide SVD caps.

## 0.8.48 - AutoTune V1 profile/recipe config contract

- Added `configs/autotune_v1_profiles.yaml` as the AutoTune V1 final-candidate profile/recipe configuration contract.
- Added `core/autotune_v1_config.py` for schema validation, profile alias resolution, recipe lookup, and scoring-boundary inspection.
- Added smoke tests for profile weight completeness, recipe references, AGC display-only boundaries, and real no-prior forbidden truth metrics.
- Added `docs/autotune/autotune_v1_final_candidate_design.md` for the backend design reference.

## 0.8.47 - AutoTune live step B-scan preview

- Added display-only live B-scan preview updates after each sequential processing step, including AutoTune recommended workflow execution.
- Added a dedicated worker `step_completed` signal so intermediate arrays are shown without committing formal current data/history until the run finishes.
- Restored the formally committed B-scan if a live-preview run is cancelled, preventing intermediate previews from being mistaken for applied results.
- Updated live-preview plot titles and regression coverage for step preview emission.
- Kept AutoTune scoring, candidate search space, processing algorithms and gprMax data contracts unchanged.

## 0.8.46 - AutoTune scoring v2 lineage/evidence closure

- 写入 AutoTune scoring v2 record 到推荐流程执行上下文、处理链路 header metadata 和最近运行摘要。
- 报告导出新增 `autotune_scoring_v2.json`，manifest / workflow / params sidecar 同步记录 scoring v2。
- 质量页处理记录在执行 AutoTune 推荐流程后追加 scoring v2 摘要。
- 修复 pytest 收集兼容：GUI runtime seam 默认 offscreen，PyWavelets 缺失时跳过 wavelet kernel tests。
- 源码包发布时排除 pycache / pyc / pytest cache。

## 0.8.45 - AutoTune scoring v2 output closure

- Added serializable AutoTune scoring v2 records for workflow recommendations, including goal weights, workflow score terms, background score terms, diagnostics and processing notes.
- Surfaced scoring v2 breakdowns in the AutoTune parameter details, candidate record table and recommendation report text.
- Recipe payloads now carry `autotune_scoring_record` for future processing records, reports and Evidence manifests.
- Kept the existing workflow search space and processing algorithms unchanged.

## 0.8.44 - Final tab responsibility audit and UI hygiene

- Added final tab responsibility audit coverage for processing, AutoTune, display, quality, and spatial pages.
- Cleaned remaining internal assistant-name variable/docs references from runtime-visible source package.
- Confirmed display page remains display-only and spatial sidecar controls remain under the spatial page.
- No processing algorithm, AutoTune scoring, recipe planner, gprMax data contract, or data format changes.

## 0.8.43 - Quality/space compact layout and spatial empty state

- Reworked the Quality page into a compact section switcher: data quality, processing record, report export, and advanced diagnostics are shown as focused cards instead of one long page.
- Added a clear Space page empty-state banner for ordinary B-scan/profile data without spatial metadata, and clarified that spatial outputs require trajectory/elevation/height or multi-line grid inputs.
- Fixed the Space page bottom status text duplication and tightened QC status-card updates for metadata and chart availability.
- Added layout contract tests for the new Quality section switcher and Space empty-state behavior.

## 0.8.42 - Quality/space page responsibility split

- Moved visible RTK/IMU/altimeter sidecar selection from the display page to the Space page. The display page keeps hidden compatibility fields only and remains display-only.
- Tightened the Quality page wording toward QC, processing records, run summaries, and report export; spatial maps and complete trajectory views remain assigned to the Space page.
- Kept sidecar state synchronization compatible with older controller paths while making the Space page the visible owner of spatial auxiliary files.


## 0.8.41 - Page responsibility contract and processing-page stage filter

- Replaced the static daily-processing flow explanation card with an actionable processing-stage filter. The bottom lineage bar remains the only source of real executed processing history.
- Added explicit page operation contracts so the display page is constrained to display-only/compare/screenshot responsibilities, while processing and AutoTune pages own data-mutating operations.
- Tightened display-page wording: display enhancement now states it only changes presentation, not processed arrays.
- Added regression tests for page-operation contracts and processing-stage method filtering.


## 0.8.40 - GPR input formats, UI fit, lineage compare tray, method taxonomy audit

- Added a GPR input format registry and native lightweight readers for MALA RD3/RD7, ImpulseRadar IPRB/IPRH, fixed-length SEG-Y, ENVI BSQ, NumPy NPY/NPZ, plus existing CSV/gprMax inputs.
- Updated import menu/file filters to route common GPR profile files through a single loader.
- Reworked the AutoTune recommendation drawer layout for narrow side panels: stacked controls, shorter tab labels, larger detail tables, and no clipped target/ROI controls.
- Moved lineage compare actions from always-visible buttons into a compact overflow menu so selected compare actions no longer squeeze the processing-chain chips.
- Audited and tightened method taxonomy labels for F-K, sharp clutter suppression, and vibration artifact suppression.

## 0.8.39 - 2026-06-03

- Introduce AutoTune scoring v2 backend modules for goal profiles, normalized metric terms, score context, and auditable score breakdowns.
- Background candidate ranking now records high-level goal-profile weights and v2 terms while preserving legacy UI metric columns.
- Workflow recipe planner now records scoring v2 terms and workflow score weights for generated recipes.
- Add a current AutoTune parameter-selection rules report for development and audit use.

## 0.8.38 - 2026-06-03

- AutoTune workflow planner now keeps baseline as a reference-only candidate and always recommends a real background-suppression method.
- When background suppression has weak estimated benefit, the UI reports this as a processing note while still using a mild background method.
- Legacy recipes containing a skipped/baseline background step are defensively converted to median background removal at execution time.

## 0.8.37 - 2026-06-03

- 修复“重置原始”后处理链路仍保留旧步骤、出现 Raw → processed → Raw 的问题；主界面重置现在回到单一 Raw 链路状态。
- 修复 AutoTune 推荐流程一键运行后链路只显示一个浓缩步骤的问题；顺序 recipe/pipeline 会记录每个执行步骤。
- 调整处理链路条高度和滚动区尺寸，避免步骤 chip 在中文 UI 下显示不全。

## 0.8.36 - 2026-06-03

- 修复 AutoTune 推荐方案一键运行后处理完成路径中的 mixin 全局导入缺失问题。
- 补充版本迭代规则：每次交付源码 zip 必须同步更新 `VERSION`、变更记录和包名。
- 新增版本一致性检查，防止 `VERSION`、changelog、打包 spec 或发布包命名脱节。

