# GX-009 GSSI400 Deep-Target Feasibility

GX-009 is a new feasibility line for GSSI-like 400 MHz deep-target experiments.
It is separate from GX-008.

GX-008 `scene_037_gssi_ey_depth05_radius03_air_sand_interface_n80_geometry_gate`
remains the primary active paper-facing synthetic diagnostic model.

GX-009 `scene_001_gssi400_deep20_radius03_air_sand_feasibility_gate` must pass:

1. geometry-only validation;
2. component audit on small raw runs;
3. raw-only `n31` gate.

Only after those gates pass should a full paired run be considered.

Expected behavior note:
400 MHz typically provides deeper penetration but lower spatial resolution than
1.5 GHz. Target response may be broader and less sharp.
