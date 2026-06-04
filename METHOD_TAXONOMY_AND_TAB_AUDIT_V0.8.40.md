# MyGPR V0.8.40 method taxonomy and tab-content audit

## 1. Algorithm taxonomy audit

### Changes made

| Method | Previous classification | V0.8.40 classification | Reason |
|---|---|---|---|
| `fk_filter` | `filtering`, but AutoTune stage was `background` | `filtering` / AutoTune stage `frequency` | F-K cone filtering is frequency-wavenumber filtering; using it as a background stage overstates background-subtraction semantics. |
| `running_average_2D` | `clutter_suppression` | `denoising` / display name `道向运行平均平滑` | It is a local running-average smoothing operation; it should not be confused with background subtraction. |
| `motion_compensation_vibration` | category `artifact_suppression`, stage `denoise` | category `artifact_suppression`, stage `artifact` | Periodic striping correction is better treated as artifact suppression, not generic denoising. |

### Classifications retained

- `subtracting_average_2D`, `median_background_2D`, `svd_bg`, `ccbs`, `sliding_avg`: background suppression is appropriate.
- `dewow`: drift correction is appropriate.
- `frequency_filter_1d`: frequency filtering is appropriate.
- `hankel_svd`, `svd_subspace`, `wavelet_2d`, `wavelet_svd`, `trace_median_filter`, `trace_savgol_filter`: denoising/smoothing is appropriate.
- `hilbert_envelope`: attribute analysis is appropriate and should not be presented as a normal amplitude-preserving processing step.
- `stolt_migration`, `kirchhoff_migration`: migration/imaging is appropriate.
- `time_to_depth`: depth conversion is appropriate.
- UAV compensation methods: motion compensation is appropriate; `motion_compensation_v2` remains the primary method.

## 2. Tab-content audit

### Current main navigation

| Tab | Current role | Audit result | Recommendation |
|---|---|---|---|
| 处理 | Import, manual processing, workflow chips, current method parameters | Mostly correct | Keep as operational processing page. Do not place trial-table or research claims here. |
| 选参 | AutoTune target/range, recommendation recipe, parameters, candidates, report | Correct after V0.8.40 UI compression | Keep top controls simple. Candidate table and scoring details should stay in sub-tabs/advanced area. |
| 显示 | Display controls, comparison modes, visual-only transformations | Correct if display-only state is explicit | Keep display transformations out of processing history unless explicitly saved as a display preset. |
| 质量 | QC metrics, runtime warnings, processing summary, report status | Correct | Should answer “有没有问题”. Avoid putting full trajectory/3D outputs here. |
| 空间 | trajectory, georeference, terrain/profile/3D outputs | Correct | Should answer “在哪里”. Keep C-scan, route, terrain, spatial interpretation here. |

### AutoTune sub-tabs

| Sub-tab | Current contents | Audit result | Recommendation |
|---|---|---|---|
| 流程 | Recipe steps and method/parameter/source table | Correct | Keep as first sub-tab; users need to see exact execution steps. |
| 参数 | Recommended parameter text and scoring overview | Correct | Later replace text blocks with compact parameter cards. |
| 候选 | Top candidate workflows | Correct | Keep Top-3/Top-N here; full trial table should remain advanced. |
| 范围 | ROI/full-range mode and coordinates | Correct | Good separation from main controls. |
| 说明 | Data mode and result notes | Correct | Avoid terms like claim boundary in this visible page. |
| 报告 | Export-ready summary | Correct | Keep report actions and summary here. |

### Processing lineage strip

- The lineage strip belongs under the B-scan because it describes the current displayed data array.
- Compare actions should not occupy permanent chain space; V0.8.40 moves them into a compact menu.
- Detailed parameters belong in tooltips or a step detail panel, not in long chip labels.

## 3. Remaining UI recommendations

1. Add a dedicated step-detail drawer opened by clicking a lineage chip, so users can see algorithm output/params without crowding the chain.
2. In AutoTune candidate comparison, add a compact “why selected” column sourced from scoring v2 breakdown.
3. In the processing page, keep manual method parameters in the left drawer; keep algorithm output in the bottom chain and main B-scan area.
4. In the display page, label all colormap/stretch/normalization operations as display-only.
5. In the quality page, avoid spatial/3D visualization except thumbnails or links to the space tab.
