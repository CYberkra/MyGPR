# GX-008-HYPERBOLA-FAST-DIAG-001 Audit

## Date
- 2026-05-25

## Branch
- main

## Base commit
- 09d643917cb5bdf071d08409f6e92b31a0651c0d

## Remote verification
- `git rev-parse HEAD`: `09d643917cb5bdf071d08409f6e92b31a0651c0d`
- `git rev-parse origin/main`: `09d643917cb5bdf071d08409f6e92b31a0651c0d`
- `git ls-remote origin main`: `09d643917cb5bdf071d08409f6e92b31a0651c0d`

## Why full 201-trace runs are paused
- Previous scene_007 full paired run consumed ~40 minutes (raw + background) and still did not show clear typical hyperbola in target_response preview.
- Fast diagnosis is needed before another full paired run.

## scene_007 runtime summary
- raw: success, `num-runs=201`, runtime ~1102s
- background: success, `num-runs=201`, runtime ~1138s
- total ~2240s

## scene_007 visual result
- Current target_response preview still has no clear typical hyperbola.

## No-run diagnostic result (scene_007 existing outputs)
1. File presence check at evidence-root top level:
   - `...scene_007.../raw_with_target1.out`, `raw_with_target101.out`, `raw_with_target201.out`: not present there.
   - Numbered `.out` files are under source model folder:  
     `experiments/gprmax/GX-008/models/scene_007_flat_dry_sand_pec_sphere_shallow/`.
2. Trace metadata check (`raw_with_target1/101/201.out`):
   - rx positions all identical: `[0.55, 0.50, 0.10]`
   - src positions all identical: `[0.50, 0.50, 0.10]`
3. Converted column variability:
   - raw_bscan col1/101/201 exactly identical (L2=0)
   - target_response col1/101/201 exactly identical (L2=0)
4. Natural sort check:
   - conversion summary source series is numerically expanded (`...1.out` to `...201.out`), no lexicographic ordering issue.
5. Preview scaling note:
   - full-window scaling can hide weak responses, but here root issue is upstream: columns are mathematically identical.

### Root-cause hypothesis from no-run diagnosis
- `dx=0.01 m` while `src_steps=rx_steps=0.005 m` for scene_007.
- In FDTD grid indexing, sub-cell movement below grid spacing can collapse to same discrete cell index.
- This explains constant src/rx metadata and identical traces across all runs.

## scene_008 design
- Scene: `scene_008_debug_three_trace_step_probe`
- Purpose: verify stepping/conversion chain with minimal run cost.
- Key settings:
  - domain `1.2 x 0.6 x 0.4`
  - `dx=dy=dz=0.01`
  - `time_window=12e-9`
  - src start x=0.30, rx start x=0.35
  - `src_steps=rx_steps=0.25`
  - raw-only `num-runs=3`
  - local `pec` sphere target.

## scene_008 raw n=3 status
- Run: success (`return_code=0`, runtime ~44.675s).

## scene_008 stepping check result
- Positions vary as expected:
  - n1: rx 0.35, src 0.30
  - n2: rx 0.60, src 0.55
  - n3: rx 0.85, src 0.80
- Ez traces differ strongly:
  - L2(1,2)=94.146
  - L2(2,3)=94.640
  - L2(1,3)=24.106
- Conclusion: gprMax stepping + reading chain works when step size is sufficiently larger than grid spacing.

## scene_009 design
- Scene: `scene_009_micro_hyperbola_dry_sand_pec_sphere`
- Purpose: micro hyperbola trend diagnosis before any full run.
- Key settings:
  - domain `1.2 x 0.6 x 0.4`
  - `dx=dy=dz=0.01`
  - `time_window=12e-9`
  - src start x=0.35, rx start x=0.40
  - `src_steps=rx_steps=0.005`
  - expected runs 61
  - target center x=0.55, local `pec` sphere.

## scene_009 raw n=61 status
- Run: success (`return_code=0`, runtime ~314.399s).

## scene_009 raw diagnostic result
- Metadata for n1/n31/n61:
  - rx all `[0.40, 0.30, 0.06]`
  - src all `[0.35, 0.30, 0.06]`
- Ez traces for n1/n31/n61 are identical:
  - L2(1,31)=0
  - L2(31,61)=0
  - L2(1,61)=0
- Same stepping-collapse behavior as scene_007.

## scene_009 background run decision
- **Not run**.
- Decision rule: raw-only stage must first show meaningful trace variability/trend; this condition failed.

## Conversion status
- scene_007 converted outputs were reused for no-run diagnosis.
- scene_008/scene_009 conversion for paired workflow: intentionally skipped at this stage (raw-only gate failed for scene_009).

## Pairing status
- scene_008/scene_009: not executed (by design gate).

## Preview status
- scene_008/scene_009 paired previews: not executed (background skipped).
- Note: default full-window preview can hide weak response, but current blocker is identical traces due stepping collapse.

## Visual hyperbola check
- scene_007 remains no clear typical hyperbola.
- scene_009 raw-only stage showed no trace-to-trace variability, so no basis to claim hyperbola trend.

## Runtime comparison vs scene_007
- scene_007 full paired 201+201: ~2240s total.
- scene_008 raw-only n=3: ~44.7s.
- scene_009 raw-only n=61: ~314.4s.
- Fast diagnostic path reduced cost while isolating stepping issue.

## Generated local artifacts
- `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_008_debug_three_trace_step_probe\raw_with_target\`
- `D:\CDUT-UavGPR-Controller\MyGPR-Evidence\gprmax\GX-008\scene_009_micro_hyperbola_dry_sand_pec_sphere\raw_with_target\`
- Source model `.out` files under:
  - `experiments/gprmax/GX-008/models/scene_008_debug_three_trace_step_probe/`
  - `experiments/gprmax/GX-008/models/scene_009_micro_hyperbola_dry_sand_pec_sphere/`

## Files deliberately excluded
- Not committed: `.out/.h5/.vti/.vtk/.vtu/.csv/.npy/.png`
- No MyGPR-Evidence git operations.

## Claim boundary
- Fast synthetic diagnostics only.
- scene_008 is stepping probe debug scene, not benchmark.
- scene_009 is micro hyperbola diagnostic candidate, not benchmark.
- Not Evidence artifact, not AutoTune evaluation, not field validation, not paper-candidate benchmark.

## Recommended next task
- `GX-008-HYPERBOLA-FAST-DIAG-002`:
  1. enforce `src_steps/rx_steps >= dx` (e.g., 0.01 or 0.02) in hyperbola-oriented scenes;
  2. rerun scene_009 raw-only with corrected step size first;
  3. only if raw shows variability, proceed to paired background/pairing/preview.
