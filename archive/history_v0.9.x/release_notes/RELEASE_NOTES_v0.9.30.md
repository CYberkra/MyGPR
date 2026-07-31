# MyGPR 0.9.30 — Studio Industrial Hardening

## Release status

Integration Preview / Alpha 2. The source baseline has closed the Phase-13 P0
interpretation, coordinate-mapping and depth-model defects. Commercial release
still requires Windows/PyQt6 runtime, CUDA and hardware-in-loop evidence.

## Safety and correctness

- Added formal backend interpretation and spatial services; Studio no longer
  reaches project-store private members.
- Enforced read-only state, managed project paths, validated line and borehole
  identifiers, and existing-line checks for interpretation persistence.
- Preserved full-resolution trace/sample coordinates across bounded B-scan
  previews and added headless coordinate-contract tests.
- Replaced the fixed borehole propagation velocity with the persisted depth
  axis or dataset dielectric model.
- Added formal processed-artifact window/read APIs and parent-artifact lineage
  for downstream processing.
- Made report scope authoritative, persisted report history and required the
  integrity manifest.
- Moved spatial generation and project restore into cancellable backend jobs.

## Runtime packaging

- The production desktop distribution now launches MyGPR Studio by default and
  declares PyQt6 as a required runtime dependency.
- Legacy Qt UI and compatibility packages remain in source for regression only
  and are excluded from the production desktop wheel.
- A separate headless backend wheel is produced for compute and CI nodes.
