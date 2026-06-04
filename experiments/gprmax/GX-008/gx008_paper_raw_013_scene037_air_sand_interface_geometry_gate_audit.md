# GX-008-PAPER-RAW-013-SCENE037-AIR-SAND-INTERFACE-SETUP-GEOMETRY-GATE Audit

## 1. Task ID
- `GX-008-PAPER-RAW-013-SCENE037-AIR-SAND-INTERFACE-SETUP-GEOMETRY-GATE`

## 2. Branch / Base / New Commit
- Branch: `main`
- Base commit: `8a0cd2aa9c6a04a9ce5ae6362cd2543ff0c8e009`
- New commit: pending

## 3. Why scene_037 is needed
- `scene_036` runs were successful, but its geometry is homogeneous dry sand (`#box ... 0.40 dry_sand_tableii`), so antenna/target are embedded in sand.
- `scene_037` introduces an explicit air/sand interface to better match realistic GPR geometry assumptions while preserving the validated GX-008 chain.

## 4. Difference from scene_036 homogeneous dry-sand gate
- Keep: domain/grid/time-window, dry-sand material constants, GSSI-like antenna model, Ey component, target radius03, depth05 cover concept, n80 safe aperture.
- Change:
  - dry sand limited to `z=0.000..0.260` (`#box ... 0.260 dry_sand_tableii`)
  - air/free-space becomes `z=0.260..0.400` (implicit default free_space)
  - antenna z set to `0.310` so skid bottom is 5 cm above soil surface
  - target z redefined for interface convention: center `z=0.180`, top `z=0.210`, bottom `z=0.150`

## 5. Coordinate convention
- Domain z: `0.000..0.400 m`
- Soil surface z: `0.260 m`
- Dry sand: `0.000..0.260 m`
- Air/free-space: `0.260..0.400 m`
- Antenna skid bottom z: `0.310 m`
- Antenna standoff to surface: `0.050 m`
- Target cover depth from surface: `0.050 m` (`0.260 - 0.210`)

## 6. Geometry table

| Item | Value |
|---|---|
| Scene ID | `scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate` |
| Domain | `1.0 x 0.15 x 0.40 m` |
| Grid | `dx=dy=dz=0.002 m` |
| Time window | `14e-9 s` |
| Scan step | `0.01 m` |
| Expected runs | `80` |
| Antenna input x range | `0.10..0.89 m` |
| Target center trace (1-based) | `41` |
| Soil box | `z=0.000..0.260 m` |
| Air layer | `z=0.260..0.400 m` |
| Antenna z | `0.310 m` |
| Target type | PEC cylinder along y |
| Target center | `(x=0.50, z=0.180) m` |
| Target radius | `0.03 m` |
| Target top / bottom z | `0.210 / 0.150 m` |

## 7. Campaign update summary
- Added new scene entry to `experiments/gprmax/GX-008/campaign_draft.yaml`.
- Added model directory:
  - `raw_with_target.in`
  - `background_only.in`
  - `materials.txt`
  - `scene_manifest_draft.json`
  - `roi_draft.json`

## 8. Dry-run result
- Command: `python scripts/gprmax_campaign_runner.py --campaign experiments/gprmax/GX-008/campaign_draft.yaml --dry-run`
- Result:
  - `campaign_status: ready`
  - `total_scenes: 21`
  - `scene_037...: ready`
  - `invalid_count: 0`

## 9. Geometry-only command outputs summary
- Raw geometry-only command: success
  - `python -m gprMax raw_with_target.in --geometry-only`
  - 3D mode confirmed
  - Domain/grid/time-window matched
  - `Ey` receiver and `y`-polarity voltage source created
  - `myGaussian` waveform at `1.71e9 Hz` created
  - Dry-sand box (`z..0.260`) and PEC cylinder both created
  - No geometry boundary error
- Background geometry-only command: success
  - `python -m gprMax background_only.in --geometry-only`
  - Same antenna/material/box setup
  - No target cylinder command present (as expected)
  - No geometry boundary error

## 10. ParaView files generated
- `D:\CDUT-UavGPR-Controller\MyGPR\experiments\gprmax\GX-008\models\scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate\gx008_scene037_raw.vti`
- `D:\CDUT-UavGPR-Controller\MyGPR\experiments\gprmax\GX-008\models\scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate\gx008_scene037_bg.vti`

## 11. What to inspect in ParaView
- Confirm dry sand occupies only lower region.
- Confirm air/free-space region exists above `z=0.260`.
- Confirm antenna geometry sits above soil surface (not embedded in dry sand).
- Confirm raw has target cylinder fully inside dry-sand region.
- Confirm background has no target cylinder.
- Confirm antenna geometry remains inside domain away from right boundary overflow.

## 12. Preflight result
- `python scripts/preflight_check.py` -> `[OK] Preflight passed`

## 13. Any warnings
- No blocking geometry warnings/errors in raw/background geometry-only logs.
- Runtime wrapper still prints known environment helper output; no failure impact.

## 14. Claim boundary
- This task only creates and geometry-validates an air/sand-interface synthetic GSSI/Ey depth05 radius03 n80 geometry gate.
- No full n80 EM simulation was run.
- No target_response produced.
- Not exact CLT-GPR replication.
- Not a paper benchmark.
- Not field validation.
- Not AutoTune evaluation.

## 15. Recommended next task
- Run a segmented n80 raw/background pair on scene_037 (same strategy as scene_036), then Ey conversion and paired visual/metrics comparison against scene_036 to quantify air/sand-interface impact.
