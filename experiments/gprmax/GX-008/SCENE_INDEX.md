# GX-008 Scene Index

## Purpose
GX-008 contains iterative gprMax scene development. Older scenes are retained for reproducibility and audit history, while `scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate` is the current primary paper-facing candidate.

## Current Active Scene

- Scene: `scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate`
- Status: `PRIMARY_ACTIVE`
- Role: air/sand-interface synthetic paired diagnostic candidate
- Use:
  - report package
  - thesis/course report draft
  - paper-facing synthetic diagnostic figure candidate
- Claim boundary:
  - synthetic paired diagnostic only
  - not exact CLT-GPR replication
  - not final benchmark
  - not field validation
  - not AutoTune evaluation

## Retained Baseline Scene

- Scene: `scene_036_gssi_ey_depth05_radius03_safe_n80_pair_gate`
- Status: `RETAINED_BASELINE`
- Role: homogeneous dry-sand baseline / ablation reference
- Use:
  - optional internal comparison
  - explains why scene_037 air/sand interface was needed

Do not use as primary report figure unless explicitly justified.

## Archived/Debug Scenes

All earlier development scenes are retained as `ARCHIVED_DEBUG_HISTORY`.

| Scene | Status | Note |
|---|---|---|
| scene_033_gssi_ey_depth03_radius05_centered_n31_raw_gate | ARCHIVED_DEBUG_HISTORY | radius/depth gate history |
| scene_034_gssi_ey_depth03_radius05_centered_n80_pair_gate | ARCHIVED_DEBUG_HISTORY | n80 segmented resume history |
| scene_035 (if present in branch history) | ARCHIVED_DEBUG_HISTORY | transitional debug history |
| scene_001 to scene_032 (excluding retained/active) | ARCHIVED_DEBUG_HISTORY | early parameter, stepping, component, and aperture diagnostics |

These scenes are retained for audit trail, parameter exploration, aperture/boundary debugging, and failed/incomplete gate documentation. They should not be used as primary report results.

## Why Not Delete Old Scenes

- They are part of the audit trail.
- Historical audits reference them.
- They document decisions that led to scene_037.
- Deleting them would harm reproducibility.
- Their status is now explicitly downgraded.

## Recommended Current Workflow

- Use `scene_037` model files for current paper-facing synthetic diagnostic work.
- Use `scene_037` Evidence report package for figures/text.
- Use `scene037_geometry_claim_audit.md` for model explanation.
- Do not create new scenes unless supervisor feedback identifies a specific geometry issue.
