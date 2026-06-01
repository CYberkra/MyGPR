# GX-008 Scene Index

## Current Primary Scene

- Scene: `scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate`
- Status: `PRIMARY_ACTIVE`
- Role: air/sand-interface synthetic paired diagnostic candidate.
- Component: Ey / `rxs/rx1/Ey`
- Expected trace count: n80

## Supplementary Scene

- Scene: `scene_038_gssi_ey_depth07_radius03_air_sand_interface_n80_pair_gate`
- Status: `SUPPLEMENTARY_DEPTH_SENSITIVITY`
- Role: single-variable depth07 supplement based on scene_037.
- Component: Ey / `rxs/rx1/Ey`
- Expected trace count: n80

## Removed From Source Tree

Older GX-008 development scenes and generated gprMax outputs were removed from
the source tree cleanup pass. The retained source-side model set is now limited
to scene_037 and scene_038.

## Claim Boundary

- synthetic paired diagnostic only
- not exact CLT-GPR replication
- not final benchmark
- not field validation
- not AutoTune evaluation
