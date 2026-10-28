# Field-processing maintenance verification

Scope: the current MyGPR workbench source changes, including full-resolution
B-scans, physical-axis provenance, trace snapshots, line properties and linked
views, selection-driven processing, A-scan tools, and saved-result roundtrips.

## Evidence

- The latest focused processing/coordinate/package suite passed 213 tests with
  one environment-dependent skip. An earlier overlapping maintenance suite
  passed 280 tests with one skip; these counts must not be added together.
- The previous CI failure paths (chain-editor presentation, optional Qt tests,
  package verification) and preview lifecycle checks passed 38 focused tests.
- The first updated CI exposed time-boundary and legacy test-adapter failures.
  The two affected time-axis files passed 62 tests after correcting source-axis
  quantization and inconsistent endpoint/relative-time fixtures; the related
  axis suite passed 46 tests with one skip, and the velocity API suite passed 5.
  Off-grid requests retain the absolute half-open interval contract.
- Windows native Qt verification used a local CDUT streaming H5 with 501 samples
  and 4,751 traces. The retained first 2,000 traces preserved full display
  resolution, and the original file SHA-256 was unchanged.
- The terrain-coordinate replay used real waveforms with an explicitly
  synthetic terrain profile and test-grid length. It verified descending
  elevation ticks, no second terrain warp, continued processing, and exact
  ZIP/H5 export and reimport. It is software-coordinate evidence, not measured
  RTK or geological validation.
- Independent code reviews checked axis provenance, Windows cache cleanup,
  optional-GUI imports, persistence, and terrain coordinate handling.

## Compatibility and acceptance limits

- Historical results without a trace snapshot do not borrow current raw
  sensors or IDs: raw replacement or changed trace mappings cannot be excluded.
  Operations requiring that metadata may need reprocessing from original input.
- Kirchhoff post-terrain correction (`topo_cor=1`) requires complete, finite
  ground elevations. Pre-terrain correction (`topo_cor=2`) rejects before
  execution because its absolute elevation reference is not verified.
- Resampled Kirchhoff results do not claim original trace IDs/sensors remain
  aligned with output columns; the original input retains its snapshots.
- Unknown migrated time axes reject time-domain operations and SEG-Y export.
- GPU hardware, new EXE builds/installation, remote transmission and hardware
  acquisition are outside this verification. No full local pytest run was
  restarted; GitHub CI is the separate merge check.
