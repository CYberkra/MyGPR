# cylinder_single_v1

This is the first clean MyGPR GPRMAX benchmark package.

- Scenario: single PEC cylinder in a dielectric half-space.
- Source kind: synthetic_reference
- B-scan shape: 220 samples x 81 traces.
- Ground truth: `ground_truth.json`.
- MyGPR input CSV: `mygpr_bscan.csv`.

The bundled fallback B-scan is deterministic and contract-oriented. It allows
MyGPR auto-tune scoring and export paths to be tested without running gprMax.
A later optional smoke can replace `mygpr_bscan.csv` with data converted from
real gprMax `.out` files while preserving the same scenario and ground-truth
schema.
