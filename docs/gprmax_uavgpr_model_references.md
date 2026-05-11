# gprMax UAV-GPR Model References

This note records which public gprMax model ideas are currently useful for the
MyGPR UAV-GPR validation suite.

## Adopted now

- HeliMax (`https://github.com/bart-inho/HeliMax`): useful for the airborne
  radar modelling pattern: Python-generated gprMax inputs, explicit free-space
  layer, Tx/Rx separation, repeated B-scan runs, and GPU/MPI execution. MyGPR
  only adopts the organisation idea, not the glacier scale or helicopter
  geometry.
- gprMax `cylinder_Bscan_2D.in`: useful as the clean hyperbola reference. The
  MyGPR `airborne_hyperbola_demo_v1` keeps UAV-style air launch geometry but
  uses a small PEC cylinder to make the diffraction hyperbola easy to inspect.
- gprMax `heterogeneous_soil.in`: useful as the complex-background reference.
  MyGPR does not copy `#add_surface_roughness` directly because our fast 2D TMz
  models use `y` as the vertical axis, while the official example expresses
  roughness in the `z` direction. Instead, `airborne_rough_soil_hyperbola_v1`
  uses deterministic segmented ground boxes plus weak clutter bodies.

## Deferred

- gprMax `cylinder_Bscan_GSSI_1500.in` and `user_libs/antennas/GSSI.py`: useful
  for a later realistic-antenna v2. It should not be mixed into the current
  benchmark until the simplified Hertzian source scenes are stable, because it
  adds a much heavier antenna model and changes the wavelet independently of
  geometry.
- Large synthetic datasets such as TunGPR are useful as evidence-generation
  architecture references, but their tunnel/bridge geometries are not UAV-GPR
  geometry and should not be treated as direct physical models for MyGPR.

## Current scene added from this review

`airborne_rough_soil_hyperbola_v1` is a UAV-GPR benchmark with:

- air-launched Tx/Rx geometry from the MyGPR airborne scene family;
- deterministic segmented rough air-ground interface;
- weak dielectric and air-like clutter bodies that are not labelled as target
  truth;
- one labelled central PEC cylinder to preserve a measurable hyperbola ROI.

The scene is meant to test whether the standard processing chain can preserve a
target hyperbola when the ground reflection and background are less ideal than
the clean demo case.
