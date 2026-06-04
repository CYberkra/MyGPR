# GX-008-PAPER-REPL-001 Parameter Audit

Paper: *Learning to Remove Clutter in Real-World GPR Images Using Hybrid Data*  
Local source checked: `C:/Users/17844/Desktop/Learning_to_Remove_Clutter_in_Real-World_GPR_Images_Using_Hybrid_Data-复制.pdf`

## Parameter Table

| parameter | paper value | source location in paper/code | MyGPR current value | match / mismatch / unknown | action |
|---|---|---|---|---|---|
| gprMax version | gprMax used; version not explicitly stated | Section II-A text + refs [43][44] | 3.1.6 runtime (local) | unknown | keep runtime; mark unresolved |
| simulation domain | 100 x 15 x 40 cm3 | Section II-A | scene_020 uses 1.0 x 0.15 x 0.40 m | match | keep |
| absorbing boundary | enabled | Section II-A | default gprMax PML | match | keep |
| dx/dy/dz | 0.2 cm each | Section II-A | scene_020 uses 0.002 m | match | keep |
| antenna type | built-in GSSI antenna, 1.5 GHz | Section II-A + refs [45][46] | `hertzian_dipole` + ricker 1.5 GHz approx | mismatch | approximation_v1, unresolved exact antenna command |
| antenna size | 17 x 10.8 x 4.5 cm3 | Section II-A | not explicitly modeled | mismatch | unresolved_in_model |
| antenna height above soil | 5 cm | Section II-A | scene_020 source/rx z at 0.10 with soil top z=0.40/near-surface approximation | approximation | keep as inferred |
| scan direction | x-direction | Section II-A | scene_020 x-direction | match | keep |
| scan step | 1 cm | Section II-A | scene_020 step 0.01 m | match | keep |
| A-scans per B-scan | 80 | Section II-A | scene_020 target 61 (gate mode) | mismatch | fast gate first; later try 81 |
| soil types | dry sand, damp sand, dry/wet clay, dry loam, heterogeneous | Section II-A | scene_020 single dry_sand_like_approx | mismatch | acceptable for minimal single-scene |
| soil EM values | listed in Table II | Table II (not machine-readable in current extraction) | inferred dry_sand_like_approx eps=3 sigma=0.001 | unknown_in_paper_extraction | unresolved, needs manual table transcription |
| object types | cylinders, PEC/PVC | Section II-A | scene_020 uses single PEC cylinder | match (subset) | keep minimal |
| object radius | 1–5 cm | Section II-A | scene_020 radius 3 cm | match | keep |
| one-object depth range | 1–10 cm | Section II-A | scene_020 approx 12 cm center depth | mismatch | tune depth toward 1–10 cm in next iteration |
| one-object x position | 50 cm | Section II-A | scene_020 center x=50 cm | match | keep |
| object orientation | along y-direction | Section II-A | scene_020 cylinder along y | match | keep |
| raw/background/clutter-free construction | clutter-free = raw - background; background: same soil/surface without objects | Section II-A / Fig.1(b) | MyGPR paired contract identical | match | keep |
| synthetic counts | 1920 raw B-scans (4 surfaces x 6 soils x 80 arrangements) | Section II-A | one-scene only | mismatch | intentional minimal scope |
| rough surface fluctuations | up to 4 cm | Section II-A | scene_020 flat-only approx | mismatch | defer rough-surface replication |
| grass/surface water settings | grass roots/blades and 1 cm water scenario | Section II-A | not modeled | mismatch | defer |
| synthetic B-scan image size used in training | 256 x 256 after augmentation/resizing | Section IV-A | MyGPR native shape; no resize in this task | mismatch | out of scope for this stage |
| evaluation target | clutter removal + target restoration | Sections IV/V | only forward-model shape alignment in this task | partial | keep limited claim |

## Parameters not found / unresolved
- Exact numeric values from Table II for all listed soil categories were not fully recoverable via current machine text extraction (manual table transcription needed).
- Exact gprMax command/config for “built-in GSSI antenna” not explicitly listed in extracted text.
- Explicit simulation `time_window` value not found in available extracted text.

## Current status
- replication_type for scene_020: `paper_aligned_approximation_v1`
- exact_replication status: **not achieved** (insufficient explicit parameters from available sources + fast diagnostic scope).
