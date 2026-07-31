## 0.9.36 Interpretation Lab Migration（2026-07-25）

- Added reversible interpretation editing sessions with assisted whole-line/interval tracing, signal snapping, smoothing, vertical shift, metadata and uncertainty bands.
- Added semantic-zone split/delete/boundary editing, full undo/redo, full-view/visibility tools and formal version persistence.
- Added `mygpr.interpretation_labels.v1` exports containing JSON labels, training NPZ, lineage, edit audit and SHA-256 integrity records.
- Moved tracing into the pure interpretation domain, preserved Backend API v1 and all 81 frozen legacy-frontend hashes, and raised weighted legacy capability coverage to 71.3%.

## 0.9.35 Advanced AutoTune & Processing Evidence（2026-07-25）

- Added target/search/ROI/profile/weight-aware AutoTune ranking with Top-3 candidates, confidence, risk and preference audit while retaining the scientific score.
- Added step-level processing diagnostics, current-versus-recommended candidate comparison and SHA-256 sealed `mygpr.processing_evidence.v1` exports.
- Added selectable line-subset batch processing, per-line failure isolation, retry and result summaries.
- Preserved the frozen legacy frontend boundary and Backend API v1 while extending the clean-room SDK and Processing Workbench contracts.

## 0.9.34 Processing Lab Interactive Migration（2026-07-25）

- Added UI-independent interactive processing sessions with method preview, step application/editing, atomic downstream replay, enable/reorder/remove, undo, redo and reset.
- Added branch-isolated processing drafts, automatic recovery, project templates, cross-line reuse, branch forking and versioned artifact commits.
- Added A-scan, spectrum, original/current grid and difference analysis plus project-wide per-line batch job submission.
- Preserved the frozen legacy frontend boundary and the backend API v1 facade while expanding the clean-room frontend SDK processing contract.

## 0.9.32 Release Consistency Hardening（2026-07-25）

- Fixed release tests and launcher contracts that still hard-coded 0.9.28 after the 0.9.31 frontend freeze.
- Extended version consistency checks across VERSION, pyproject metadata, packaging specs, Windows launchers, current handoff documents, changelog and release notes.
- Corrected the cross-platform runner so it launches the production MyGPR Studio entrypoint instead of the frozen legacy Qt frontend.
- Updated current-state, startup, TODO and developer-handoff documentation to reflect the seven-workspace Studio mainline and migration-only legacy boundary.

## 0.9.31 Legacy Frontend Freeze（2026-07-25）

- Froze the historical Qt frontend behind explicit migration-only opt-in and prohibited new production dependencies on it.
- Switched the reproducible Windows build to the clean-room Studio specification and added bounded Studio smoke-test startup.
- Added a hash-locked legacy frontend manifest, archive and release quality gate so the frozen implementation cannot drift silently.
- Preserved the legacy frontend as a separate read-only recovery bundle pending real Windows/PyQt acceptance and final removal.

## 0.9.30 Studio Industrial Hardening（2026-07-25）

- Closed Phase-13 P0 defects in interpretation write safety, full-resolution B-scan coordinate mapping and borehole depth conversion.
- Added formal backend interpretation/spatial/artifact services and removed frontend access to private project-store internals.
- Made report selections and history authoritative, added borehole evidence, and moved spatial generation and project restore to cancellable jobs.
- Added headless traceability tests for the repaired safety contracts and split production desktop/backend packaging surfaces.

## 0.9.29 Studio Frontend Integration（2026-07-24）

- Added the clean-room PyQt6 Studio frontend with project, processing, interpretation, spatial, delivery, task and simulation workspaces.
- Preserved the Phase-12 backend API v1 and added the optional dielectric-constant field to line import without removing or renaming existing fields.
- Added real processed-artifact preview, versioned interpretation/borehole/spatial adapters, research-mode gprMax export, and Mock/real backend composition.
- Added Studio contract, integration and clean-room boundary tests plus a dedicated packaging entry point and PyInstaller specification.

# Backend Phase 8 — Native Processing Closure（2026-07-22）

- Completed native backend coverage for all 34 public processing methods; the legacy executor remains only as a compatibility/fallback seam.
- Added native CPU Kirchhoff migration, optional CuPy GPU selection/budget/fallback policy, and deterministic regression evidence.
- Added an explicitly experimental scalar 2-D zero-offset RTM baseline with CFL stabilization, absorbing boundaries, cancellation and resource caps.
- Migrated all six motion-compensation methods and the remaining eleven processing methods into UI-independent backend algorithm packages.
- Added a release contract test that prevents any public processing method from silently falling back to the historical engine.
- Verified 428 selected non-GUI backend tests, all architecture/schema/format/complexity/debt/package/test-policy/version gates, and five Qt-free CLI smoke workflows.

# V2.0.0 Hybrid Project Store Phase 1（2026-07-15）

- 新建项目默认使用 `mygpr.field_project.v3`、`catalog.sqlite` 和每测线独立 HDF5。
- 增加 HDF5 分块写入/切片代理、处理分支 DAG、导出登记、完整性双向核对和失败回滚。
- 增加旧项目显式无损迁移、备份 checkpoint、删除级联和 UI 分支选择。
- 保留源文件不可变证据；标准化原始矩阵采用带备份的受控替换策略。

# V1.9.8 正式工作台去演示化（2026-07-15）

- 删除隐式演示工程、合成项目种子和硬编码项目树。
- 启动后保持“未打开项目”，仅允许显式新建或打开真实项目。
- 删除演示地图、演示 B-scan、固定测线/版本/日期及演示任务条回退。
- 测试改为显式测试夹具，不再依赖产品演示接口。

## 2026-07-22 - Architecture / AutoTune Phase 1

- Migrated AutoTune orchestration from the 3414-line `core/auto_tune.py` into focused domain and application modules.
- Preserved `core.auto_tune` as a registered compatibility facade.
- Added architecture policy V2 with layer direction, cycle detection, migration ownership, frozen-module growth limits and new-code size checks.
- Replaced the ineffective debt-reduction target with measurable lower targets and tightened the release ratchet baseline.
- Verified deterministic AutoTune output equivalence for zero-time, dewow, background removal, SEC gain and F-K filtering.

## 0.9.28 Module Linkage V1.8 — 2026-07-14

- Added persistent five-page workspace context for selected line, processing source, annotation, spatial result and report version.
- Added semantic processing-to-annotation-to-spatial-to-report handoffs with version-aware destination focus.
- Added upstream/downstream dirty-state propagation and navigation badges that clear sequentially as derived artifacts are regenerated.
- Added page-level linkage actions and selective affected-module refresh instead of unconditional full-page rebuilds.
- Added close/reopen restoration and reproducible cross-module linkage evidence capture.

## 0.9.28 Report Closure V1.7 — 2026-07-14

- Added immutable `report_vNNN` report versions with a current-version index.
- Bound every formal report to one project data revision, template/software version and immutable spatial-result version.
- Added transactional PDF/HTML/XLSX/table/figure generation, report sealing, file audit, SHA-256 manifests and standalone delivery ZIP hashes.
- Added report staleness checks and dynamic report-page controls for snapshot, approval and delivery status.
- Preserved legacy report manifests and the approved five-page Golden Reference layout.

## 0.9.28 UI V1 - Approved five-page visual baseline implementation — 2026-07-13

- Replaced the permanent left navigation rail with the approved horizontal five-page product navigation and page-specific context toolbars.
- Rebuilt project management around the project tree, engineering map, line register, recent activity, notes and project/storage inspector.
- Reframed line processing around a left project/version tree, dominant B-scan work area, explicit apply workflow and contextual processing control panel.
- Preserved the continuous basal-interface annotation workflow while integrating it into the common five-page shell.
- Rebuilt spatial results with a persistent results tree, plan/profile/3-D view tabs and configuration/layer/export inspector.
- Preserved the report outline, document preview, report configuration and delivery output workflow in the unified shell.
- Archived the five accepted reference images under `docs/ui_reference/five_page_v1/` and upgraded capture diagnostics to `five_page_visual_reference.v1`.

## 0.9.28 - Long-term maintainability and production foundations — 2026-07-13

- Unified project sessions, locking, transactions and durable atomic storage across the production and compatibility APIs.
- Added central schema migration/quarantine, durable job journals/resource locks, clock-domain sensor synchronization, GIS cache/CRS contracts and report package sealing.
- Added external backup/restore verification, large-source Merkle/full hashes, typed algorithm catalogs, structured support diagnostics and reproducible package supply-chain evidence.
- Replaced the heavy startup closure with lazy compatibility loading and added architecture, schema-catalog and technical-debt ratchet gates.
- Intentionally left formal/demo mode separation unchanged for a separate product-design decision.
- Removed CLI dependence on implicit `PythonModule` path injection and serialized Qt/VTK modules across local release shards to eliminate native teardown races.
- Final Linux/offscreen release gate: 1339 passed, 0 failed, 0 errors.

## 0.9.27 - Risk-based layered quality gates — 2026-07-13

- Replaced routine monolithic acceptance runs with L0, affected, smoke, merge, nightly and release gates.
- Added the central `config/test_impact.toml` source-to-test impact map with conservative escalation for storage/schema, Job Manager and radar-RTK-IMU synchronization changes.
- Added manifest-driven pytest tier/domain markers and policy validation for all test modules.
- Added `scripts/select_tests.py` and `scripts/run_quality_gate.py` with Git-diff selection, machine-readable plans/results, native GUI process isolation, CI sharding and safe fallback for unmapped source changes.
- Added Linux/Windows PR smoke, nightly four-shard regression and Linux/Windows release workflows.
- Added Ruff critical checks, focused Mypy checks, pytest timeouts, Makefile/Windows wrappers, test data tiers and developer documentation.

## 0.9.26 - Field production systems: jobs, synchronization, GIS, annotation and formal delivery — 2026-07-13

- Added a unified Qt Job Manager and task center for large imports, preflight, quality checks, processing, sensor synchronization, GIS operations, spatial exports, annotation persistence, formal reports, backups and large project-maintenance operations.
- Replaced blocking full-matrix import preflight with header/mmap inspection or bounded CSV sampling; batch import no longer scans every source twice.
- Added cooperative cancellation, progress reporting and transactional commit boundaries to source verification, B-scan orientation correction, GIS copying, spatial-coordinate export and other long-running operations.
- Added radar-trace timestamp synchronization for RTK, IMU and altimeter streams with independent offsets, trigger delay, residual thresholds, coverage/fixed-solution diagnostics, gap/jump detection and lever-arm correction.
- Added project-local synchronized per-trace metadata, trajectory CSV and synchronization manifest outputs; out-of-range or high-residual samples are not silently clamped to sensor endpoints.
- Replaced decorative spatial backgrounds and pseudo-DEM strips with real offline GIS layers: GeoTIFF/DEM, KML, GeoJSON, Shapefile, GeoPackage and coordinate CSV, including CRS registration, reprojection, layer visibility and engineering map export.
- Upgraded the continuous basal-interface workstation with mmap viewport slicing, full-resolution local windows, A-scan evidence, local zoom, keyboard navigation, keypoint tables, whole-curve shift, robust smoothing and recovery drafts.
- Added formal report package schema `mygpr.report_package.v3` with PDF, HTML, Excel workbooks, approval metadata, synchronization/GIS/processing/interface summaries, engineering figures, audit CSV and SHA-256 checksum manifest.
- Preserved the v0.9.25 chunked import, directory-backed mmap storage, continuous basal-interface labels and old point-target read compatibility.

## 0.9.24-r2 - Visual-reference polish and responsive closure

## v0.9.24 UI freeze finalization — 2026-07-12

- Centralized cross-platform CJK font selection and measurable font diagnostics.
- Added critical text-fit checks to every five-page screenshot capture.
- Upgraded Windows diagnostics to record real DPR, taskbar available geometry, native frame margins and physical screenshot sizes.
- Added one-command Windows 125%/150% capture and two-profile freeze manifest verification.
- Validated eight responsive/DPI profiles with zero layout, visual-comfort and text-fit issues.


- Replaced the thin radar/target brand mark with a high-contrast pulse tile that remains legible from compact navigation rails through high-DPI report covers.
- Standardized the full workbench on the blue visual-reference palette and a cross-platform Chinese sans-serif font fallback chain.
- Refined the top project selector, engineering context bands, workstation cards, semantic status badges, task cockpit and processing-step chain.
- Improved target-list column sizing and compact inspector proportions while preserving the B-scan as the dominant interpretation surface.
- Refined the report cover, page thumbnails, template controls and export hierarchy to match the approved visual reference more closely.
- Verified layout and visual-comfort rules at 1280×720, 1366×768, 1536×816, 1920×1080 and 2560×1440 logical resolutions, plus 125% and 150% Qt scale factors.

## 0.9.24 - Three-page workstation redesign and processing fit

- Rebuilt the project-management, line-processing and spatial-results pages around the approved three-page workstation contract instead of the former equal-weight metric-card layout.
- Added shared `EngineeringContextBand` and `WorkstationSection` components, with one primary action per page layer and low-frequency operations moved into contextual menus.
- Reorganized project management into a dominant line worklist, ordered task dock and narrow intelligence tabs.
- Reorganized line processing into a dominant B-scan, basic/advanced parameter cockpit and tabbed processing-chain/message/overview band.
- Reorganized spatial results into a dominant map, tabbed elevation/DEM/association analysis band and compact layer/summary cockpit.
- Added `three_page_redesign.v1` capture metadata and automated checks for contract presence, context-band ratio, primary-workspace dominance and map dominance.
- Added deterministic module-isolated release testing for the mixed PyQt6/Matplotlib/VTK suite and restored the bundled CLI batch MVP configuration.
- Added PyVista to the development environment for the 3-D grid contract tests.

- Rebalanced the line-processing side panel for 1080P/125% Windows captures after user screenshot review.
- Increased the manual-processing side-panel width just enough to keep spin boxes and action buttons readable while preserving the two B-scan plots as the visual focus.
- Tightened parameter label widths and enforced minimum control sizes for spin boxes, combo boxes and line edits.
- Added stable layout keys and checks for continuous-processing buttons and the continuous-processing action card.
- Updated layout diagnostics so the manual-processing side panel must be neither too narrow nor too wide, and button heights are checked automatically.

## 0.9.23 - Manual chain layout closure

- Tightened the manual processing-chain side panel so it no longer pushes the bottom processing-history/log area out of the compact 1080P workspace.
- Kept the B-scan comparison as the primary visual area while aligning the processing settings side panel to the main plot height.
- Reorganized continuous-processing actions into a compact hierarchy: execute current step, undo/reset row, before/after compare, save current result, and a small recommend-parameters action.
- Preserved the v0.9.22 manual processing-chain data model and traceability; this release is a UI/layout closure pass, not a template/pipeline rollback.

## 0.9.22 - Manual processing chain

- Replaced the abandoned processing-template direction with a practical manual step-by-step processing chain for the default workbench.
- Added `core/manual_processing_chain.py` so the processing page can keep an in-memory sequence of executed methods, support one-step undo, full reset, and save the final result with complete chain traceability.
- Reworked the processing side panel around continuous manual processing: execute current step, undo one step, reset to original, compare before/after, save current result, and recommend parameters.
- Added a processing-history tab so users can see the already stacked steps instead of relying on hidden template state.
- Saved processing manifests now include `processing_mode=manual_step_chain`, `chain_step_count`, and `chain_steps` for downstream audit and interpretation-source traceability.

## 0.9.20 - Module linkage closure

- Added project-level event primitives and a persistent `metadata/project_state.json` tracker for cross-module dirty/stale state.
- Added dependency rules so line import, source-file relink/check, trajectory import, QC, B-scan orientation fix, processing save, target changes, spatial refresh/export and report generation update downstream module state consistently.
- Integrated a lightweight `ProjectLinkageController` to record events, update stale report metadata and refresh affected UI modules without new visible pages.
- Added explicit spatial/report stale notices and report-check rows so users can see when spatial成果需刷新 or成果报告需重新生成.
- Enhanced project-tree navigation: processed-result, target-annotation, spatial-result and report nodes now navigate to their corresponding workspace while keeping the selected line synchronized.

## 0.9.19 - Project navigation declutter

- Moved project-level actions into the top current-project drop-down: switch project, create/open project, settings, backup, and delete project.
- Removed the always-visible recent-project block from the project-management right sidebar to reduce visual clutter.
- Reworked the project-management right action area into compact grouped sections: data import and project maintenance.
- Kept line-level actions close to the current project tree through a contextual right-click menu for opening, locating, quality review, B-scan orientation fix, source-file operations, export, and deletion.
- Preserved source-file provenance and project deletion semantics from v0.9.18 while changing only entry hierarchy and UI density.

## 0.9.18 - Direct project-local deletion

- Changed line deletion from soft archive/trash behavior to direct deletion of project-local line artifacts under `raw/`, `processed/`, `targets/` and line-specific `spatial/` outputs.
- Changed project deletion from backup-and-trash movement to direct removal of the active MyGPR project folder.
- Added guards and tests ensuring original import source files outside the project directory are not deleted by line/project deletion.
- Kept "项目备份" as an explicit separate operation; deletion no longer creates automatic archives or `.mygpr_trash` folders.

## 0.9.16 - 2026-06-10

- Added initial safe line archiving/deletion: selected lines were removed from `project.json` while raw, processed, target and spatial artifacts moved to `trash/lines/<line_id>_<timestamp>/` with a manifest.
- Added initial safe project deletion by backing up and moving the active project to `.mygpr_trash/projects/`.
- Added recent-project removal from the project operation panel.
- Added GUI actions in Project Management for deleting the selected line, removing a recent project, and deleting the current project with typed project-name confirmation.
- Added regression coverage for line archive/delete, project trash move, backup creation, and recent-project cleanup.

## 0.9.15 - 2026-06-10

- Added a visual-comfort polish pass for the field workbench after the v0.9.14 Windows capture review.
- Reworked field table rendering so selected/alternate rows remain light on Windows inactive focus; state is now conveyed by text/icon color rather than heavy full-row fills.
- Improved sparse trajectory/spatial preview viewport handling so empty, single-point, and short-line plots do not appear as oversized blank charts.
- Enlarged the target-positioning B-scan work area and tightened the target inspector into compact key-value rows.
- Strengthened the delivery report preview so the report cover carries more visual weight and the table-of-contents panel no longer dominates the preview area.
- Added visual-comfort diagnostics via `ui.field_panels.visual_comfort_rules` and `scripts/check_visual_comfort.py`; screenshot capture now writes `visual_check_report.json`.

## 0.9.14 - 2026-06-10

- 增加右侧辅助栏折叠机制：项目管理操作栏、测线处理参数栏、空间成果辅助栏、成果报告检查/导出栏均可收起，低分辨率下可释放主工作区宽度。
- 增加主图放大查看入口：项目/测线预览、测线处理双 B-scan、空间成果主图及辅助图可通过图卡右上角按钮打开独立大图窗口。
- 新增 `CollapsibleSidePanel`、`PlotViewerDialog` 和 PlotCard 标题栏动作接口，避免在各页面重复实现折叠/放大逻辑。
- 保留 v0.9.13 布局诊断规则，截图诊断继续输出 `layout_check_report.json` 并保持 pass/fail 判定。

## 0.9.13 - 2026-06-10

- 修复 `projectQuickPreviewMapCard` 和 `processingLineOverviewMapCard` 在 1080P/125% 紧凑布局下被父容器压缩、导致小图卡画布裁切的风险。
- `PlotCard` 增加与固定画布高度一致的最小卡片高度，避免画布高度大于所属卡片高度。
- 调整自适应布局参数：项目管理底部预览区和测线处理底部辅助区改为可容纳小图卡的稳定高度，同时保持底部区不超过页面高度 22% 的约束。
- 新增 `ui.field_panels.layout_diagnostics_rules` 和 `scripts/check_layout_diagnostics.py`，将 `layout_diagnostics.json` 从“记录尺寸”升级为自动 pass/fail 规则检查。
- 截图服务新增 `layout_check_report.json`，并在 `layout_diagnostics.json` 内嵌 `check` 字段；后续 Windows 真机回传可直接运行 `python scripts/check_layout_diagnostics.py <capture_dir或layout_diagnostics.json>` 判定布局是否合格。

## 0.9.12 - 2026-06-10

- 基于用户回传的 Windows 真机 `windows_fit_check_v0911` 复核 `1536×816 / 125%` 缩放下的 v0.9.11 自适应布局效果。
- 继续收口测线处理页布局参数：双 B-scan 主图区进一步增高，右侧参数栏和底部消息区略压缩，减少主图与底部区域之间的无效空白。
- 新增 `layout_diagnostics.json` 输出：截图诊断现在会记录关键控件真实几何尺寸，后续可按规则比较真机布局，而不是只靠肉眼截图判断。
- 为项目管理、测线处理、空间成果、成果报告的关键卡片和画布添加稳定 `layoutKey` 动态属性；保留原有 `objectName` 供 QSS 样式使用。
- 新增 `tests/test_layout_diagnostics_v0912.py`，确保布局诊断采集不会破坏卡片样式标识。

## 0.9.10 - 2026-06-10

- 按用户圈选的 4 个核心标签页完成定向布局优化：项目管理、测线处理、空间成果、成果报告。
- 项目管理页强化“项目概览 + 测线清单 + 项目操作”三栏主轴，底部任务区与快速预览改为 7:3 辅助布局。
- 测线处理页将原始/处理后 B-scan 作为主工作区，右侧参数栏收窄，底部检查提示与当前测线概览收敛。
- 空间成果页放大主空间图，右侧整合剖面曲线、DEM、测线关联与空间信息，底部测线汇总表收紧。
- 成果报告页改为报告预览主导，检查结果与导出信息合并为右侧辅助栏，交付文件/任务/日志收到底部。
- 版本号从 v0.9.9 迭代到 v0.9.10，并同步更新启动器、环境检查、版本一致性测试与交接文档。

## 0.9.9 - 2026-06-10
- 根据用户 Windows 真机 `windows_fit_check` 截图继续收口 15.6 寸 1080P/125% 缩放布局。
- 修复 PlotCard 在扩展卡片中将固定高度 canvas 垂直居中的问题，B-scan、空间图和测线关联预览改为顶部对齐。
- 优化测线处理页 B-scan 高度、空间成果页主图/DEM/关联图高度，减少大面积空白与图像下沉。
- 新增 `tests/test_windows_fit_plot_layout_v099.py`，防止关键预览画布再次下沉或过小。
- 复核用户回传的 `windows_fit_check_after_patch`，确认窗口按 `1536×816` 逻辑可用区域 / `125%` 缩放捕获，核心页面已无横向溢出。
- 修复现场工作台表格普通单元格在 Windows 浅色表格背景上的文字对比度不足问题，并补充 `tests/test_field_table_contrast_v099.py`。
- 强化测线处理参数面板输入控件和标签的显式深色文字样式，避免不同 Qt/Windows 调色板下出现浅色文字。
- 新增跨平台环境检查脚本 `scripts/check_env.py`，只读检查 Python 版本、运行依赖、项目结构和日志目录权限。
- 新增统一启动入口 `scripts/run_app.py`，集中设置 Qt/OpenGL/Matplotlib/faulthandler 运行环境。
- 更新 Windows 启动器、环境检查和安装脚本版本标识到 v0.9.9，并把 `pyproj` 纳入发布前环境检查。
- 新增 `scripts/check_release_hygiene.py`，用于发布前检查 cache、pyc、runtime_projects、logs 等不应进入发布包的残留。
- 补充 Windows 发布前稳定性审计文档和启动器/环境检查回归测试。
- 发布包打包策略排除缓存、运行时项目、日志和字节码文件。
- 修复发布包卫生问题：移除随包残留的 `runtime_projects/field_demo_project`。
- 修复非 Windows 环境检查/启动入口可能在发布包根目录生成 `logs/` 的问题，日志和 Matplotlib 缓存改写入用户状态目录。
- 同步修正 v0.9.9 文档口径：PDF 已纳入 beta 报告包闭环，正式行业报告模板仍需后续细化。

## 0.9.8 - 2026-06-10
- 成果报告页接入真实报告包生成，不再只是静态预览。
- 新增 core/field_report_export.py，生成可审计的 CSV / JSON / HTML 报告目录。
- 报告包汇总项目元数据、测线清单、数据质检、处理结果、目标标注和空间成果导出。
- 生成 reports/report_<timestamp>/report_manifest.json 与 reports/latest_report_manifest.json，并回写 project.json reports 状态。
- 报告页新增“生成报告包 / 打开报告目录 / PDF 后续接入”操作。
- 暂不引入 PDF 依赖，优先保证 beta 阶段交付文件稳定、可追溯、可人工审计。

## 0.9.7 - 2026-06-10
- 使用真实营山测线数据完成端到端试运行：6 条 CSV 均可导入、投影、质检并生成标准化 GPR 数据和轨迹文件。
- 修复营山文件名 `LineL1origin(36).csv` / `LineX1origin(36).csv` 的测线编号推断，分别生成 `L01_36` / `X1_36`，不再回退为 `L05` / `L06`。
- `CGCS2000 / 3-degree GK` 无显式 Zone 时现在在投影 manifest 中标记 `is_auto=true`，便于区分自动分带与手工指定分带。
- 目标和空间成果保存路径增加 line_id 校验，目标 CSV 中的内嵌非法 line_id 不再覆盖当前测线编号。
- 完成处理结果保存、artifact index、目标来源绑定、空间 XY 导出和无轨迹空坐标导出的真实项目链路验证。

## 0.9.6 - 2026-06-10
- 审计并修复算法入口与处理结果保存链路，处理 manifest 升级到 `mygpr.processing_manifest.v2`。
- 算法执行 manifest 增加输入/输出数据 SHA256、输入/输出数据集元数据、有限值统计和输出警告。
- 处理结果保存改为不可变时间戳参数 sidecar：`<line_id>_params_<timestamp>.json`，保留 `<line_id>_params.json` 作为 latest 指针。
- 处理 artifact manifest 增加 `save_schema`、`params_sha256`、`manifest_sha256`、`output_data_sha256`。
- Artifact 索引改为按数据时间戳匹配参数/manifest，避免旧版“所有历史结果指向最新参数”的追溯错误。
- 测线处理页保存前增加 line_id / manifest 一致性校验，移除项目内保存时的 synthetic fallback。
- 增加 `tests/test_processing_traceability_v096.py` 回归测试。

## 0.9.5 - 2026-06-10
- 继续压缩 field_workbench_window.py，将首页项目总览构建逻辑拆至 ui/field_panels/home_page.py。
- 将通用表格创建/填充逻辑拆至 ui/field_panels/table_utils.py，供项目、首页、处理等页面复用。
- 将 B-scan 与测线轨迹预览辅助逻辑拆至 ui/field_panels/preview_helpers.py。
- 主窗口行数由 1212 行降至 907 行，降低继续开发时的耦合风险。
- 保持项目、导入、处理、质检、空间成果、报告等业务协议不变。

## 0.9.4 - 2026-06-10
- 将测线处理页 UI、算法预览、应用、撤销、保存、参数推荐和参数面板回调从 `ui/field_workbench_window.py` 拆分到 `ui/field_panels/processing_page.py`。
- `FieldWorkbenchWindow` 新增组合 `ProcessingPageMixin`，主窗口从 1665 行压缩到约 1212 行。
- 保留原算法入口、处理参数、manifest 保存和项目刷新逻辑，不改变算法结果含义。
- 增加拆分结构回归测试，防止后续测线处理逻辑重新堆回主窗口。

## 0.9.3 - 2026-06-10
- 配置并验证当前开发环境可导入 PyQt6，恢复 GUI 截图回归能力。
- 新增 `ui/field_panels/project_page.py`，将项目管理页构建、项目操作、导入、质检、备份和测线清单导出回调从主窗口拆出。
- 新增 `ui/field_panels/target_actions.py`，将目标标注来源切换、新建/删除/保存、自动识别辅助和目标 B-scan 点击回调从目标定位页面拆出。
- `FieldWorkbenchWindow` 改为组合 `ProjectPageMixin` / `InterpretationPageMixin` / `SpatialPageMixin` / `DeliveryPageMixin`，降低主窗口继续膨胀风险。
- 更新回归测试，确保项目操作回调和目标标注回调不再堆在主窗口文件中。

## 0.9.2 - 2026-06-10
- 批量导入对话框增加关闭保护：导入线程运行中关闭窗口会先请求取消并阻止销毁窗口。
- 切换测线时清空处理预览缓存，避免上一条测线的处理结果污染当前测线。
- 新建项目默认坐标系统改为 `CGCS2000 / 3-degree GK` 自动分带，不再默认 Zone 39。
- 无真实轨迹时不再为目标标注写入硬编码假 XY 坐标；空间导出保留空坐标，避免污染真实成果。
- B-scan 转置修正始终二次确认，并明确提示会改写标准化数据和保留备份。
- legacy `choose_loose_path()` 不再是 no-op，转入正式导入入口。

## 0.9.1 - 2026-06-10
- 修复 CGCS2000 3-degree Gauss-Kruger Zone EPSG 映射：Zone 39 解析为 EPSG:4527，避免误用 CM 114E CRS。
- 新增统一 line_id 校验，阻止路径穿越、路径分隔符和 Windows 保留设备名进入 raw/<line_id>/。
- 正式导入入口改为事务式：导入失败时恢复 project.json 并清理/恢复 raw 测线目录，避免失败导入污染项目。
- 移除真实项目 load_trajectory() 的 demo fallback；缺失轨迹时显式 FileNotFoundError，由调用层决定空状态。
- B-scan 转置修正 manifest 增加轴重建策略和工程交付警告，便于复核修正后的物理轴含义。
- 新增核心数据安全回归测试，覆盖坐标投影、line_id 校验、导入回滚、轨迹缺失和方向修正 manifest。

## 0.9.0 beta - 2026-06-10
- v0.9.0 beta 准备版：明确当前 beta 功能边界，补齐用户手册、开发交接和按钮回调审计记录。
- 项目备份入口从占位提示改为真实 ZIP 备份，默认写入当前项目 `backups/`，排除 `.venv`、`.git`、`__pycache__` 和已有备份目录。
- 测线清单“导出清单”按钮接入真实 CSV 导出，写入 `reports/line_manifest.csv` 或用户选择路径。
- 新增 beta 回归测试，覆盖项目备份、测线清单导出、占位入口清理和 v0.9.0 文档边界。
- 保留 v0.8.91-v0.8.99 的营山 CSV 导入、坐标投影、后台批量导入、导入诊断、数据质检和 B-scan 方向修正能力。

## 0.8.99 - 2026-06-10
- 新增质检详情对话框，可查看当前测线采样点数、道数、时间窗、长度、振幅范围、NaN/Inf、轨迹点数、方向判断与问题列表。
- 项目管理页增加“查看质检详情”和“修正B-scan方向”入口。
- 新增 B-scan 转置修正服务：修正前自动备份 NPZ，转置后重建轴信息、保存 orientation_fix_manifest.json，并重新运行质检。
- 修正后自动刷新项目树、测线清单、快速预览和测线处理预览。
- 保留 v0.8.98 自动质检与方向风险检测能力；不会自动转置，必须由用户显式确认。

## 0.8.98 - 2026-06-10
- 新增导入后数据质检模块 core/field_data_quality.py，检查矩阵尺寸、振幅范围、NaN/Inf、时间窗、测线长度、轨迹点数和 B-scan 方向风险。
- 测线导入后自动生成 raw/<line_id>/<line_id>_quality_report.json，并把数据质量状态写回测线清单。
- 项目管理页“运行数据质检”接入真实项目质检逻辑，显示通过/警告/失败汇总。
- ProjectStatusSnapshot 增加质检统计，并在任务/关注项中显示未质检、警告和失败状态。
- B-scan 预览纵轴在深度轴可用时显示“深度 (m)”，减少时间轴/深度轴显示混淆。
- 保留 v0.8.97 后台批量导入、结果表格、坏文件诊断和 v0.8.94 坐标投影能力。

## 0.8.97 - 2026-06-10
- 批量导入进度对话框新增结果表格，逐文件显示状态、文件名、测线编号、矩阵尺寸、长度、文件大小、耗时和诊断信息。
- 批量导入服务层结果结构增加 file_size_mb、elapsed_s、raw_dir、manifest_path、diagnosis 字段。
- 增加失败文件诊断规则，针对矩阵校验失败、头信息不完整、数据行不足、不支持格式等情况给出明确建议。
- 导入结果对话框新增“打开 raw 目录 / 查看 manifest / 复制诊断”操作。
- 批量导入完成后不再自动关闭对话框，保留结果表供用户检查。
- 保留 v0.8.96 后台导入、取消后续导入和失败不中断机制。

## 0.8.96 - 2026-06-10
- 批量导入测线改为后台 QThread 任务，避免百万行 CSV 导入时冻结主界面。
- 新增批量导入进度对话框，显示总数、当前进度、逐文件日志、成功/失败汇总。
- 支持用户取消后续文件导入；已成功导入的数据保留，未执行文件标记为取消。
- batch_import_line_data 增加 progress_callback 与 cancel_requested 钩子，保持核心导入逻辑可测试且不依赖 UI。
- 保留 v0.8.94 坐标投影与 v0.8.95 文件名识别/批量导入能力。

## 0.8.95 - 2026-06-10
- 项目管理页新增“批量导入测线”入口，支持一次选择多个 CSV / NPY / NPZ / H5 文件。
- 新增营山常见文件名识别：Line9origin(30).csv、Line3origin.csv、L1origin.csv、X1origin.csv 等可自动生成测线编号和名称。
- 批量导入逐文件执行，单个文件失败不影响其它文件继续导入。
- 批量导入完成后输出成功/失败汇总，并刷新项目树、测线清单、快速预览和空间成果。
- 保留 v0.8.94 坐标投影逻辑，不改变算法入口和数据协议。

## 0.8.94 - 2026-06-10
- 新增 `core/coordinate_projection.py`，支持 CGCS2000 / 3-degree Gauss-Kruger 分带坐标解析和经纬度投影。
- 营山/旧 MyGPR sidecar CSV 导入后，轨迹文件同时保留 `longitude/latitude` 与工程 `x/y` 坐标。
- `import_manifest.json` 新增 projection 记录，保存 EPSG、分带、投影状态与错误信息。
- 空间成果页优先使用真实工程坐标绘制测线轨迹、高程剖面和空间信息。
- 新增 `pyproj` 依赖到 requirements/environment，用于正式坐标转换。
- 新增坐标投影回归测试，覆盖 zone 39、自动分带和导入后投影轨迹写入。

## 0.8.93 - 2026-06-10
- 真实项目联动二次收口：测线处理、目标定位、空间成果和成果报告页继续移除固定 demo 数值。
- 测线处理页指标改为读取 ProjectStatusSnapshot；无真实 GPR 数据时显示空状态，不再渲染合成 B-scan。
- 目标定位页标题、指标和日志跟随当前选中测线；无数据时不再自动生成演示候选目标。
- 空间成果页改为从当前项目轨迹和测线记录绘制平面图、高程剖面、测线汇总与关联视图；无轨迹时显示空状态。
- 成果报告页改为读取项目状态、报告目录、任务行和日志行；报告预览中的项目名称、位置、更新日期跟随当前项目。
- 首页和项目管理页的测线轨迹/位置预览进一步绑定真实轨迹，无轨迹时显示空状态。

## 0.8.92 - 2026-06-10
- 修复新建/打开项目后左侧项目树仍显示 demo 测线的问题，项目树改为随当前项目 manifest 动态重建。
- 顶部“当前项目”改为最近项目切换下拉框，可直接切换已记录项目。
- 项目切换、导入测线、新建项目、打开项目后重建首页和工作区页面，保证项目概览、测线清单、任务区、快速预览和测线轨迹同步当前项目。
- 快速预览 B-scan 不再在无真实数据时显示合成演示图，改为空状态；有真实数据时显示当前选中测线数据。
- 取消项目树/测线选择时自动生成 demo GPR 数据，避免空项目被伪数据污染。
- 新增项目状态绑定回归测试，覆盖空项目、导入测线、项目树和快速预览空状态。

## 0.8.91 - 2026-06-10
- 修复营山实测 CSV 导入失败：支持 `Number of Samples = 501,,,` 等头信息尾随逗号格式。
- 支持 5 列旧 MyGPR 实测 CSV：经度、纬度、高程、幅值、飞行高度，无时间戳列也可导入。
- 导入预检不再把营山 CSV 误判为 5 列普通矩阵；可正确显示 MyGPR 航空 GPR 主数据 CSV、矩阵尺寸和定位信息。
- 新增营山 CSV 导入回归测试，覆盖预检启用确认导入、标准化数据集、轨迹 CSV 和 import_manifest 输出。
- 增加 Qt 对话框/按钮回调静态审计测试，防止 `QDialog.DialogCode` 类未导入等同类崩溃再次出现。

## 0.8.90 - 2026-06-10
- 修复项目操作对话框确认按钮触发路径中的 P0 崩溃：`field_workbench_window.py` 补充 `QDialog` 导入。
- 覆盖“新建项目 -> 直接确定”回归测试，确保默认表单确认后可创建正式项目并刷新工作台。
- 覆盖项目设置和导入预检对话框的 `QDialog.DialogCode.Accepted` 路径，避免同类未导入符号问题再次出现。
- 不改变 CSV sidecar 导入、算法入口、项目数据协议和 GUI 布局。

## 0.8.89 - 2026-06-10
- 修复 Windows 启动器稳定性：启动时启用 Python faulthandler，并在日志中记录真实启动命令与关键运行环境。
- 为 Matplotlib 配置可写 `MPLCONFIGDIR`，避免双击启动时反复重建字体缓存。
- 移除 Matplotlib 字体回退链中的未安装 `Noto Sans`，保留已安装 CJK 字体和 `DejaVu Sans`，减少 `findfont` 日志洪泛。
- Windows 启动器默认设置 `QT_OPENGL=software`，降低显卡/OpenGL 驱动导致的 PyQt 原生崩溃风险。
- 不改变 v0.8.88 的 CSV 导入、项目数据协议和 UI 页面逻辑。

## 0.8.88 - 2026-06-09
- 恢复旧 MyGPR 航空 GPR 主数据 CSV 导入规则：识别 Number of Samples / Time windows / Number of Traces / Trace interval 头信息。
- 旧 CSV 第 4 列 amplitude 自动重塑为 B-scan 矩阵，输出 rows=samples、cols=traces。
- 从旧 CSV 第 1/2/3/5/6 列提取经度、纬度、高程、飞行高度和时间戳，自动生成测线轨迹 CSV。
- 导入后写入标准化 GPRDataSet、元数据、轨迹和 import_manifest.json。
- 导入预检窗口可显示 MyGPR 航空 GPR 主数据 CSV、矩阵尺寸、测线长度、时间窗、列识别和定位信息。
- 保留普通二维矩阵 CSV/TXT/NPY/NPZ/H5 导入能力。

## 0.8.87 - 2026-06-09
- 首页底部“最近项目活动”改为双列紧凑信息流，降低横向空间占用。
- 首页底部三栏比例调整为“活动 / 交付 / 预览 = 4 / 4 / 3”，减少对右侧区域的挤压。
- 新增活动项卡片样式，提升 15.6 英寸 1080P 下的可读性。
- 不改变项目数据协议、算法入口、目标定位、空间成果和报告页业务逻辑。

## 0.8.86 - 2026-06-09
- 首页删除“今日关注”模块，释放右侧空间，避免 15.6 寸 1080P 下首页内容被挤压。
- 首页中部改为“项目流程概览 + 模块快速概览”双栏铺开。
- 首页底部改为“最近项目活动 / 交付成果概览 / 项目位置与 B-scan 预览”三栏布局。
- 不改变项目数据协议、算法入口、目标定位、空间成果和报告页逻辑。

## 0.8.85 - 2026-06-09
- 修复 1080P 首页总览布局：重构底部区块，扩大交付成果概览与右侧预览，消除右下角信息卡严重挤压问题。
- 修复 项目管理页快速预览：提高 B-scan 预览高度与右侧信息区比例，避免剖面图在紧凑布局中出现“倾斜/压扁”的观感。
- 优化 空间成果页：微调右侧信息栏、DEM/关联视图高度与说明文字换行，提升 1920×1080 下的稳定显示。
- 优化 成果报告页：放大报告封面预览与右侧交付信息栏宽度，确保中文标题、目录与字段文本完整显示。

## 0.8.84 - 2026-06-09

- 修复 15.6 英寸 1080P Windows 全屏实机客户区下项目总览、项目管理和目标定位页仍需滚动/内容溢出的问题。
- 将适配基准从仅 1536×864 逻辑视口扩展为 1920×1020 实际截图客户区和 1536×820 高 DPI 紧凑视口。
- 压缩顶部栏、导航、侧栏、指标卡、表格行高、目标定位 B-scan 和右侧标注信息面板高度。
- 本轮不新增业务功能，不改变算法入口和项目数据协议。

## 0.8.83 - 2026-06-09

- 修复 15.6 英寸 1080P 笔记本在 Windows 125% 缩放/1536×864 逻辑视口下，主页面需要滚动才能看全的问题。
- 默认窗口调整为 1600×900，截图脚本仍可按 1920×1080 输出交付截图。
- 压缩顶部栏、导航、侧栏、指标卡、图像预览和表格高度，新增紧凑视口守护测试。
- 首页当前测线预览和项目管理页快速预览改为跟随当前选中测线与真实数据集，不再固定写 L03。
- 保持算法入口、项目数据协议和目标来源绑定逻辑不变。

## 0.8.82 - 2026-06-09

- 新建项目表单补充项目编号、设备型号、坐标系统和垂向基准等工程元数据字段。
- 新增“项目设置”入口，可修改项目名称、编号、测区位置、操作员、设备型号、坐标系统和垂向基准，并同步写入 `project.json`。
- 新增 `update_project_metadata()` 服务函数，避免 GUI 直接修改 manifest。
- 首页和项目管理页继续从真实项目状态读取并显示项目元数据。
- 新增 `tests/test_project_metadata_settings.py`，覆盖项目元数据创建、更新、最近项目记录和对话框字段。

## 0.8.81 - 2026-06-09

- 新建项目入口升级为项目信息表单，可填写项目名称、测区位置、操作员和保存目录。
- 导入测线数据前新增预检窗口，显示文件格式、支持状态、矩阵尺寸、测线长度、时间窗和导入建议。
- 新增 `core/field_import_preview.py`，将导入校验从 GUI 回调中抽出，避免 UI 直接承担数据解析与错误解释。
- 最近项目入口在项目管理页可直接选择并打开，不再只写入 recent projects JSON。
- TXT 矩阵文件纳入正式直接导入入口；DZT / RD3 / DT1 等厂商格式继续只识别并提示转换。
- 新增 `tests/test_project_wizard_import_validation.py`，覆盖导入预检、厂商格式提示、项目创建、最近项目和导入元数据。

## 0.8.80 - 2026-06-09

- 项目管理页和首页的关键指标改为从 `project.json`、`raw/`、`processed/`、`targets/`、`spatial/`、`reports/` 和项目日志计算，不再继续使用固定 demo 数值。
- 新增 `core/field_project_status.py`，集中生成项目状态快照、任务行、检查提示、最近活动和交付文件列表，避免 UI 页面继续自行拼装项目事实。
- 项目树开始从真实测线、处理结果、目标标注和空间成果文件生成。
- 数据质检入口基于项目状态快照给出导入、RTK/IMU 和报告状态提示。
- 新增 `tests/test_project_status_metrics.py`，守护空项目和真实导入项目不再显示固定 demo 指标。

## 0.8.79 - 2026-06-09

- 接入正式项目操作入口：新建项目、打开项目、导入测线数据、导入 RTK/IMU、最近项目记录。
- 新增 `core/field_project_operations.py`，将用户项目操作从 UI 按钮回调中抽出。
- 新建项目现在可创建空正式项目，不自动填充 demo 测线。
- 导入入口支持 CSV / NPY / NPZ / H5 / HDF5 的直接读取与归一化；厂商格式先识别并提示转换。
- 项目管理页数据导入与质检卡片从静态展示变为可点击操作。
- 新增项目操作回归测试。

## 0.8.78 - 2026-06-09

- 目标定位页图像显示现在会随“标注来源”真实切换：原始数据、已保存处理结果、显示与对比 / 坐标轴转换结果分别读取对应矩阵。
- 新增 `core/target_source_data.py`，将目标来源绑定解析为可显示的 B-scan 视图，避免目标定位页继续依赖临时 UI 状态。
- `draw_target_bscan` 支持真实矩阵、距离轴、深度轴和垂直轴标签；`time_to_depth` 来源显示为深度轴。
- 新增 `tests/test_target_source_data.py`，覆盖 raw / processed / time_to_depth 三类来源切换。

## 0.8.77 - Target annotation source binding

- Added target source binding on the target-positioning page so annotations can reference raw data, saved processing artifacts, or display/compare axis-transform artifacts.
- Added `core.target_source_binding` and extended target CSV persistence with source mode, data path, manifest path, method metadata, artifact role, axis transform and input/output shapes.
- Kept `time_to_depth` as a display/compare axis-transform capability and allowed targets to bind to those artifacts with traceable metadata.
- Wired target creation and automatic detection candidates to the selected annotation source.
- Added `tests/test_target_source_binding.py`.

## 0.8.76 - Field project store boundary split

- Split `core.field_project_store` from a multi-responsibility store into a small project-manifest coordinator plus dedicated line, target, spatial, processed-artifact and demo-bootstrap store mixins.
- Added `core.field_project_models` as the shared project schema/model module while keeping the public imports from `core.field_project_store` compatible for existing UI and tests.
- Reduced `core/field_project_store.py` from about 576 lines to about 123 lines so future target, spatial and report work has clearer persistence boundaries.
- Kept the project directory contract unchanged: `project.json`, `raw/`, `processed/`, `targets/`, `spatial/`, `reports/`, `logs/`.
- Kept `time_to_depth` as a supported display/compare axis-transform capability and did not add new business features in this release.

## 0.8.75 - Field workbench risk closure and documentation partitioning

- Moved target interpretation, spatial results and delivery/report page logic out of `ui.field_workbench_window` into dedicated `ui.field_panels.*_page` mixins.
- Moved shared cards and plot helpers into `ui.field_panels.widgets` and `ui.field_panels.plots`, reducing the main workbench file from about 2346 lines to about 1497 lines.
- Actually partitioned documentation into `docs/user`, `docs/developer`, `docs/audit` and `docs/legacy` so current product documentation is separated from historical research material.
- Kept `time_to_depth` as a supported display/compare axis-transform capability.
- Kept preset workflows and spacing-changing acquisition utilities out of the current workbench page.

## 0.8.74 - Engineering hygiene, capture summary fix and artifact indexing

- Kept `time_to_depth` as a supported display/compare axis-transform capability and extended processing manifests with role and axis-transform metadata for future result viewing pages.
- Added `ui.field_panels.capture_service` and `scripts/capture_field_workbench.py` so screenshots write a validated `capture_summary.json` for the actual source root and version.
- Added `core.processing_artifact_index` to index saved `processed/` outputs from project files instead of transient UI state.
- Began low-risk UI boundary cleanup by moving field workbench style constants and processing-panel wording guardrails into `ui/field_panels/`.
- Added engineering hygiene documents covering code-health findings, document index, terminology guardrails and the v0.8.74 cleanup record.
- Added Conda `environment.yml` as an optional reproducible environment entry without changing the existing Windows launcher behavior.

## 0.8.73 - Field algorithm compatibility regression

- Added a v0.8.73 compatibility pass for the field workbench processing bridge, covering dewow, background suppression, frequency filtering, gain, denoising and time-to-depth conversion.
- Added `docs/algorithm_compatibility_v0.8.73.md` and `tests/test_algorithm_compatibility.py` so exposed algorithms are checked for parameter generation, execution, output finiteness and trace-count preservation.
- Extended processing manifests with success status and output-dimension change flags for safer project auditing.
- Improved the measurement-line processing page failure handling: failed algorithms now keep the current preview intact, disable saving, and show the failure reason in the processing information and log area.
- Kept preset workflows and spacing-changing acquisition utilities out of the current workbench page.

## 0.8.72 - Version discipline, launcher regression and algorithm bridge stabilization

- Standardized the release identity to MyGPR v0.8.72 across VERSION, README, launcher output, handoff documents and release package naming.
- Kept the Windows launcher non-invasive: it does not install packages automatically and now documents the expected environment search order before falling back to system Python.
- Added regression checks for the v0.8.71 launcher fix so existing Conda/venv environments are preferred over clean Windows Python Launcher interpreters.
- Stabilized the field processing bridge around the existing methods registry and processing engine, while keeping preset workflows and spacing-changing utilities out of the current workbench page.
- Added CURRENT_STATE, DEV_HANDOFF and TODO handoff files for the new field-workbench iteration line.

## 0.8.71 - Windows launcher environment selection fix

- Fixed the one-click launcher selecting a clean system Python 3.12 before the user's existing MyGPR environment.
- Added `scripts/mygpr_windows_launcher.py` as a read-only environment selector and diagnostic launcher.
- Added `MYGPR_PYTHON` override support for users who want to bind MyGPR to a specific Conda/venv Python.
- Kept dependency installation explicit through `install_mygpr_environment.bat` or manual `python -m pip install -r requirements.txt`; the launcher itself does not install packages.

## 0.8.70

- 修复项目管理页测线预览：现在与测线处理页使用同一套项目数据加载逻辑，正确解析带 RTK/高度/时间戳列的航空/车载 GPR CSV，不再把经纬度等辅助列显示成假竖条。
- 调整大型演示截图流程：采用更贴近现场 B-scan 的多测线模拟数据，截图去重并覆盖“检查、处理、目标、空间、报告、交付文件、日志”的完整流程。
- 优化空间成果/成果报告在大型项目上的生成速度：优先从 RTK/高度计辅助文件读取空间轨迹，不再为了生成空间成果而反复加载完整雷达主数据。
- 成果报告检查表在全部通过时显示明确“成果检查通过”，避免空白表格让用户误以为页面没加载。
- 工作台日志记录项目打开、页面切换、测线预览和交付成果生成事件，并显示实际时间。

## 0.8.69 - Frontend/backend skill-guided product clarity audit

- Applied two skill-style audit passes: a frontend field-operator clarity checklist and a backend delivery/compatibility checklist.
- Replaced remaining user-visible engineering keys in delivery tables and reports with field-facing Chinese labels while keeping stable manifest keys for compatibility.
- Renamed visible legacy/processing wording toward “完整处理/处理结果” so the UI no longer looks like a transitional prototype.
- Mapped QC severity and check codes to readable labels in project, QC, and delivery tables; raw codes remain stored internally for automation.
- Allowed Chinese delivery package names and improved sidecar/status labels shown in the current-information panel.

## 0.8.68 - Field wording and product-readiness polish audit

- Ran a product-language audit across the default Workbench, classic processing window, parameter recommendation page, target-marking page, spatial results page, delivery page, startup guide, and root package layout.
- Reworded operator-facing labels away from research/AI-style terms: AutoTune -> 自动推荐/参数推荐, 证据 -> 交付文件/处理记录, 解释对象 -> 目标标注, ROI -> 关注范围 where visible to field users.
- Replaced vague shell wording such as 工作空间/上下文检查器 with 页面/当前信息, and made gate/status messages explain the operator action required.
- Cleaned the ZIP root by moving historical audit and patch-note documents into `docs/audits/history/`, leaving the root focused on launch scripts, README, requirements, and runtime entry points.
- Rewrote `README.md` as a Chinese field-user guide with the five-step workflow and Windows-first launch instructions.
- Preserved processing algorithms, saved project formats, batch CLI semantics, and developer-only research tools behind the existing explicit environment flags.

## 0.8.67 - Field product UI mode and research-surface hiding

- Repositioned the default Workbench as a field exploration/positioning product with five user-facing workspaces: 项目管理、测线处理、目标定位、空间成果、成果报告.
- Hid research-only surfaces from the normal UI, including the gprMax 仿真验证 workspace, AutoTune 研究验证 entry, and legacy 真值/研究 segmented pages.
- Added an explicit developer override via `MYGPR_ENABLE_RESEARCH_UI=1` or `MYGPR_PRODUCT_MODE=research` to keep prior research tools available without exposing them to field users.
- Renamed user-facing resource groups and delivery text from 解释层/成果包 toward 目标标注/交付成果, and changed the bottom drawer from 证据 to 交付文件.
- Preserved processing algorithms, AutoTune scoring, gprMax campaign tooling, CLI semantics, and project file formats.

## 0.8.66 - Deep line-level audit and remaining coverage closure

- Performed a line-level static audit across auditable source/text files and archived raw audit summaries under `docs/audits/`.
- Added a standard `requirements.txt` runtime dependency file and changed the bundled Windows installer to install runtime dependencies from it; `requirements-dev.txt` now layers test dependencies on top.
- Hardened the gprMax simulation validation page so invalid GPU device text disables command generation/copy instead of silently falling back to a CPU command.
- Implemented `cli_batch.py resume --summary <summary.json>` to re-run failed jobs from a previous batch summary instead of leaving the CLI resume path as a placeholder.
- Reworded unsupported method/family runtime errors from “unfinished” style wording to “unsupported” so guardrails are not mistaken for missing features.
- Preserved processing algorithms, AutoTune scoring, gprMax execution semantics and Evidence schema semantics.

## 0.8.65 - UI coverage audit and launcher alias completion

- Added the Chinese Windows launcher/check aliases promised by the startup README: `启动MyGPR.bat`, `启动MyGPR_调试日志.bat`, and `检查MyGPR环境.bat`.
- Aligned `check_mygpr_environment.bat` with the launcher Python search order across Python 3.13, 3.12, 3.11, and 3.10.
- Hardened gprMax simulation command preview quoting so copied commands with spaces in paths work in Windows CMD/PowerShell-style shells.
- Suppressed command generation for invalid gprMax campaign scenes and blocked direct copy calls even if invoked programmatically.
- When a saved processing version is selected from any workspace, the Workbench now switches to the data-document workspace so the read-only preview is immediately visible.
- Expanded regression coverage for the launcher aliases, environment checker order, simulation command quoting, and processing-version workspace navigation.
- Preserved processing algorithms, AutoTune scoring, gprMax execution and Evidence schema semantics.

## 0.8.64 - Workbench processing-version navigation completion

- Completed the project-first Workbench processing-version resource path: saved results now populate the project tree via `ProjectService.list_processing_results()` and open as read-only B-scan preview documents.
- Added processing-chain summary, result metadata, and inspector context when selecting a saved processing version.
- Hardened global splitter layout restore so the bottom task/QC/evidence drawer remains readable after restart.
- Added regression coverage for opening saved processing versions directly from the Workbench resource tree.
- Preserved processing algorithms, AutoTune scoring, gprMax execution and Evidence schema semantics.

## 0.8.63 - Workbench UI completion guardrail pass

- Tightened Processing Lab AutoTune application so an old recommendation cannot be applied after the user switches to another method.
- Kept the Apply Recommendation button disabled unless the selected method matches the stored recommendation.
- Fixed Interpretation Workbench raw/result source switching so returning to raw data does not accidentally reapply the processing QC gate.
- Added regression coverage for method-specific recommendations and interpretation source switching.
- Preserved processing algorithms, AutoTune scoring, gprMax execution and Evidence schema semantics.

## 0.8.62 - UI completion and environment bootstrap pass

- Added a package-local Windows environment installer that creates `.venv` and installs `requirements-dev.txt` dependencies.
- Aligned dependency checks with the launcher by including PyWavelets/`pywt`.
- Added shared Matplotlib CJK font fallback configuration so direct workbench page imports render Chinese plot labels without missing-glyph warnings when CJK fonts are available.
- Connected delivery package generation back to the workbench Evidence drawer, listing manifest, report, checksums, spatial synthesis and indexed evidence files after export.
- Preserved processing algorithms, AutoTune scoring, gprMax execution and Evidence schema semantics.

## 0.8.61 - Research console primary UI entry completion

- Added a direct visible entry from the main AutoTune recommendation page to the read-only research validation console.
- Integrated the research console as a first-class advanced tab while preserving the legacy compatibility page and existing concise primary tabs.
- Expanded gprMax model draft discovery beyond the initial six hard-coded scenes and replaced the geometry placeholder with parsed target/background directives.
- Hardened research-console file opening across Windows, macOS and Linux without changing algorithms, AutoTune scoring, gprMax execution, or Evidence writing.
- Hardened Workbench close teardown so optional processing-page shutdown hooks cannot block layout persistence or project-lock release.

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


## 0.9.24 - 2026-07-12 UI finalization

- Finalized the five-page visual-reference UI shell, CJK font policy, responsive navigation, status badges, workbench layouts, and high-DPI behavior.
- Added critical-text fit diagnostics to every five-page capture.
- Added an auditable Windows 125%/150% freeze workflow with runtime/font/frame diagnostics, manual reviewer signing, and a combined verification manifest.
- Validated eight logical/DPI profiles with zero layout, visual-comfort, and critical-text clipping issues.
- Final isolated release gate: 913 passed, 0 failed, 0 skipped, 370 deselected across 221 test modules.

## Backend architecture phase 2 - 2026-07-22

- Added UI-independent processing domain models, application ports and legacy infrastructure adapters.
- Completed dependency-injected AutoTune orchestration behind processing and constraint ports.
- Added Qt-free jobs, progress, cancellation, event subscription and event polling.
- Added `MyGPRBackend` API v1 and a backend CLI smoke flow.
- Closed schema catalogue blockers and extended architecture/debt/complexity gates.
- Preserved legacy AutoTune imports and numerical behaviour.
