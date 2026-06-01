# GX-008 Working Guide

GX-008 has been slimmed to the two currently useful air/sand-interface scenes:

- `scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate`
  - Role: primary active synthetic paired diagnostic scene.
  - Cover depth: 0.050 m.
  - Component: Ey / `rxs/rx1/Ey`.
- `scene_038_gssi_ey_depth07_radius03_air_sand_interface_n80_pair_gate`
  - Role: scene_037 single-variable depth07 sensitivity supplement.
  - Cover depth: 0.070 m.
  - Component: Ey / `rxs/rx1/Ey`.

Generated gprMax outputs are intentionally not retained in this source tree.
Only model inputs and small metadata files remain here.

Claim boundary:

- synthetic paired diagnostic scenes only
- not exact CLT-GPR replication
- not finalized paper benchmark
- not field validation
- not AutoTune evaluation
