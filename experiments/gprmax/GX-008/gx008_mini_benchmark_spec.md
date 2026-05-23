#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GX-008 Mini Benchmark Specification

## 1. Purpose

GX-008 is a paper-inspired paired synthetic mini benchmark for MyGPR.

- Primary objective: support the MyGPR evidence chain and fixed-workflow AutoTune parameter recommendation validation.
- This is not a full CLT-GPR replication.
- This is not a deep learning dataset.
- This is not field validation.

## 2. Relationship to GX-007

- GX-007: single-scene complete 2D diagnostic artifact.
- GX-008: small multi-scene benchmark specification.
- GX-007 validated backend run/pairing/preview/report/evidence path.
- GX-008 targets cross-scene stability and parameter recommendation evaluation.

## 3. Scene Matrix

Planned scene scale:
- Core scenes: 12
- Optional stress scenes: 4-6

Core factors:
- Soil: `dry_sand_like`, `damp_sand_like`, `dry_loam_or_clay_like`
- Surface: `flat`, `rough_light`
- Target: `PEC cylinder`, `PVC cylinder`
- Depth: `shallow`, `medium`

Selected 12-scene controllable subset (not full-factorial explosion):

| Scene ID | Soil | Surface | Target | Depth | Purpose |
| --- | --- | --- | --- | --- | --- |
| gx008_s01 | dry_sand_like | flat | PEC | shallow | clean baseline |
| gx008_s02 | dry_sand_like | flat | PEC | medium | depth sensitivity |
| gx008_s03 | dry_sand_like | flat | PVC | shallow | low-contrast baseline |
| gx008_s04 | dry_sand_like | rough_light | PEC | shallow | mild clutter robustness |
| gx008_s05 | damp_sand_like | flat | PEC | shallow | conductivity impact |
| gx008_s06 | damp_sand_like | flat | PEC | medium | deeper + conductive medium |
| gx008_s07 | damp_sand_like | flat | PVC | medium | weak target under damping |
| gx008_s08 | damp_sand_like | rough_light | PEC | shallow | clutter + conductivity |
| gx008_s09 | dry_loam_or_clay_like | flat | PEC | shallow | high permittivity contrast |
| gx008_s10 | dry_loam_or_clay_like | flat | PVC | shallow | low-contrast in difficult soil |
| gx008_s11 | dry_loam_or_clay_like | rough_light | PEC | medium | difficult but tractable case |
| gx008_s12 | dry_loam_or_clay_like | rough_light | PVC | medium | hardest core case |

Optional stress scenes (example range):
- multi-target crossing hyperbolas
- thin-layer interface clutter
- shallow void stress
- stronger roughness stress
- low-SNR stress with constrained time_window
- small geometry perturbation sensitivity

## 4. Material Parameters

Recommended ranges (to be finalized with explicit references in implementation phase):

- dry sand-like: `eps_r ~ 3`, `sigma ~ 0.001 S/m`
- damp sand-like: `eps_r ~ 8`, `sigma ~ 0.01 S/m`
- dry loam/clay-like: `eps_r ~ 10-12`, `sigma ~ 0.001-0.01 S/m`
- PVC target: `eps_r ~ 3.5`, `sigma = 0`
- PEC target: ideal metal (`PEC`)

Note: implementation tasks must attach paper/manual references for final parameter locks.

## 5. Geometry and Scan Design

Recommended modeling baseline:
- Domain: keep compact but enough for target hyperbola development; e.g. x-length supporting 21-41 traces first.
- Grid step `dx/dy/dz`: choose stable values that avoid non-physical cell/wavelength issues; keep consistent per pair.
- `time_window`: long enough to include direct wave and target reflection zone without unnecessary overhead.
- Source/receiver: begin with simplified Ricker + Hertzian dipole for controllable baseline.
- Commercial antenna models: consider only in later stress/realism phases after baseline reproducibility is stable.
- Scan step: fixed small step per scene family.
- Trace count and scan length: define together; avoid mixing ad hoc `-n` with inconsistent `#src_steps/#rx_steps`.

`#src_steps/#rx_steps` and `-n` / `--num-runs` policy:
- Model file defines per-run stepping geometry.
- `-n` (or MyGPR `--num-runs`) controls repeated model runs / trace count series.
- For paired runs, raw/background must use the same `-n`, same steps, and same geometry so shape remains compatible.

## 6. Target Design

Target design rules:
- Cylinder orientation should be selected to produce interpretable hyperbola in B-scan.
- For typical hyperbola behavior: ensure lateral scan crosses target center offset range.
- Radius/size: pick small-to-moderate values to cover detectable and weak cases.
- Depth bands:
  - shallow: strong/cleaner response
  - medium: attenuation and clutter sensitivity
- PEC vs PVC:
  - PEC: high-contrast idealized reflector
  - PVC: lower-contrast realistic stress target
- Optional stress variants: void/layer/rough-surface perturbations.

## 7. Paired Design Contract

Per scene contract:
- Must provide `raw_with_target` and `background_only`.
- Raw/background may differ only by target object definitions.
- Must keep identical:
  - soil and surface
  - antenna/source/receiver setup
  - scan geometry and stepping
  - `time_window`
  - `dx/dy/dz`
- `target_response = raw - background`.

Failure policy:
- Shape mismatch: hard failure.
- Partial output (timeout/crash): mark invalid, no synthetic claim upgrade.
- Timeout policy: record status, runtime, and partial-ness in manifests; do not silently coerce as success.
- Expected shape policy: each pair must have same 2D shape `[samples, traces]`.

## 8. Artifact Contract

Per scene curated outputs:
- `raw_bscan.csv`
- `background_bscan.csv`
- `target_response.csv`
- `paired_metrics.json`
- `paired_validation_summary.json`
- preview PNG set
- lightweight report
- `evidence_manifest.json`

Must not be committed to MyGPR source repo:
- `.out`
- `.h5`
- `.vti`
- `.vtk`
- `.vtu`
- generated arrays/figures from real runs

These belong to MyGPR-Evidence curation path.

## 9. Metrics Plan

For `METRIC-SYN-001`:
- MAE
- MSE
- PSNR
- MS-SSIM
- energy ratio
- target/background contrast
- clutter suppression ratio
- ROI preservation
- target distortion penalty
- runtime

## 10. AutoTune Validation Plan

GX-008 support scope for AutoTune:
- fixed workflow only
- baseline branch
- manual/default branch
- auto recommendation branch

Boundaries:
- no full workflow selection optimality claim
- in synthetic paired scenes, compare processed output against `target_response` references
- for real no-prior field data, only proxy metrics + manual review warnings are allowed

## 11. Claim Boundary

GX-008 boundaries:
- can support synthetic fixed-workflow parameter recommendation evaluation
- cannot prove field performance
- cannot prove AutoTune superiority over experts in all data
- cannot prove real no-prior underground truth correctness
- is not a CR-Net dataset replication

## 12. Implementation Roadmap

- `GX-008-MODEL-001`: build initial scene model files and ROI drafts.
- `GX-008-DRYRUN-001`: campaign dry-run audit and contract checks.
- `GX-008-RUN-001`: first controlled subset execution.
- `GX-008-CONVERT-001`: convert native outputs to pairing-ready arrays.
- `GX-008-EVIDENCE-001`: curate benchmark evidence package.
- `METRIC-SYN-001`: add synthetic paired quantitative metrics.
- `AT-SYN-001`: evaluate fixed-workflow AutoTune recommendation quality on GX-008.
