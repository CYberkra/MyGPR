#!/usr/bin/env markdown
# Native gprMax Benchmark Package Contract

GX-002 defines the minimum provenance chain for using gprMax data as MyGPR
research evidence:

```text
model.in / native .out or _merged.out
-> selected /rxs/<receiver>/<component>
-> MyGPR-compatible CSV
-> scenario + manifest + ground truth
-> audit
-> Evidence repository
```

CSV is allowed as a MyGPR-compatible representation only when the package
preserves traceability to the native `.out` and exact `.in`. A CSV without that
chain is a smoke or synthetic fixture, not native gprMax evidence.

## Required Files

A native package should contain or reference:

- `model.in`: exact gprMax input used to generate the run.
- native `.out` or `_merged.out`: HDF5 gprMax output. Large raw files may stay
  outside Git, but their absolute path, hash, shape, and generation/conversion
  command must be recorded.
- `mygpr_bscan.csv`: converted samples x traces matrix for MyGPR.
- `scenario.json`: source kind, geometry, timing, selected receiver/component,
  raw hash, CSV hash, and conversion command.
- `native_gprmax_package_manifest.json`: lightweight package index.
- `ground_truth.yaml` or converted `ground_truth.json`: target/background ROIs
  with documented interval semantics.
- `preview.png`: quick B-scan preview for review.
- `gprmax_package_audit.json`: machine-readable audit result.

## Required Metadata

The package must record:

- `dt` from the native `.out` HDF5 attributes;
- iteration/sample count and converted CSV shape;
- `#domain`, `#dx_dy_dz`, and `#time_window`;
- source and receiver positions;
- `#src_steps` and `#rx_steps` for constant-step B-scans;
- selected receiver and component, for example `/rxs/rx1/Ez`;
- raw `.out` SHA-256 and converted CSV SHA-256;
- conversion command and source MyGPR commit when written to Evidence.

## PML / Domain Requirements

For paper-candidate data:

- Tx/Rx start and end positions must stay outside the effective PML margin.
- Targets should remain away from PML and lower/side boundaries.
- A 2D thin dimension such as `z == dx` may be used for fast TMz-style models,
  but it must be explicitly documented as a thin/2D dimension.
- Scan length must be sufficient to show the target response without clipping
  the hyperbola arms.
- Time window must cover the direct wave, surface/interface response, and target
  response with enough post-target samples for noise/background metrics.

The audit tool assumes a conservative default 10-cell PML margin unless the
package records a different value.

## Package Types

- `smoke`: tiny deterministic fixture or short native run for import and report
  plumbing.
- `normal`: representative native run with acceptable geometry and ROI audit,
  usable for method development.
- `stress`: deliberately exaggerated noise, motion, or geometry perturbation for
  robustness and visualization; not a field baseline.
- `paper-candidate`: native `.out` provenance, safe PML/domain geometry, reviewed
  ground truth, sufficient scan length, and complete Evidence.

## Paper-Safe Claims

Allowed when audit passes:

- MyGPR converted a native gprMax receiver/component into a reproducible CSV.
- The Evidence report is traceable to a specific `.in`, `.out`, receiver,
  component, hashes, and source commit.
- Ground-truth metrics are computed after processing and do not guide the same
  AutoTune search unless explicitly stated in a separate method.

Forbidden without stronger evidence:

- Claiming a synthetic CSV is native gprMax output.
- Claiming paper-grade AutoTune performance from a smoke fixture.
- Claiming field generalization from a single gprMax scene.
- Hiding PML/domain/ROI warnings when using the figure in a report.
- Using ground truth as selection input while describing the method as ordinary
  heuristic AutoTune.

## Tooling

Prepare a package:

```bash
python scripts/gprmax_benchmark/prepare_native_gprmax_package.py ^
  --model-in output/gprmax_datasets/pipe_demo_longline_v1/pipe_demo_longline_v1.in ^
  --out output/gprmax_datasets/pipe_demo_longline_v1/pipe_demo_longline_v1_merged.out ^
  --output-dir output/gprmax_native_csv_packages/pipe_demo_longline_v1 ^
  --receiver rx1 ^
  --component Ez ^
  --scenario-id pipe_demo_longline_v1 ^
  --ground-truth output/gprmax_datasets/pipe_demo_longline_v1/ground_truth.yaml
```

Audit a package:

```bash
python scripts/gprmax_benchmark/audit_gprmax_package.py ^
  --package output/gprmax_native_csv_packages/pipe_demo_longline_v1 ^
  --output-json output/gprmax_native_csv_packages/pipe_demo_longline_v1/gprmax_package_audit.json
```

If the `.out` is missing or `h5py` is unavailable, the preparation script writes
a `pending_native_out` manifest instead of faking native data.
