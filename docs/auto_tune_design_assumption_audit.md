#!/usr/bin/env markdown
# AutoTune Design Assumption Audit (AT-012)

## 1) Executive summary

This audit freezes current claims and documents design assumptions before adding
new native gprMax scenes or finalizing any preset.

Current status:

- Fixed absolute parameters from single-scene runs are **not** universal defaults.
- GX-003 indicates a tendency toward large / near-full-line background windows,
  but this is scene-specific evidence.
- Ground-truth metrics and heuristic visual QC must remain separated.
- No overall AutoTune superiority claim is allowed at this stage.

The main risk is assumption drift: once single-scene settings are mistaken for
global defaults, future experiments can look stable while being invalid for
other trace counts, spacing, and target/background structures.

## 2) Parameter scale risk table

| Parameter family | Current usage pattern | Scale risk | Severity | Guardrail |
|---|---|---|---|---|
| `subtracting_average_2D.ntraces` | Previously tuned as absolute values (`73`, `97`, `89-121`) | Absolute values can overfit one trace count and become invalid when line length changes | P0 | Use ratio-based candidates with trace-count-aware clamping and policy labels |
| `subtracting_average_2D.window_length_m` (implicit) | Usually not reported unless computed | Physical meaning lost across different trace spacing | P1 | Always report `window_length_m` when spacing exists |
| `dewow.window` | History contains small and large windows | Small windows can collapse target-band energy on some native scenes | P1 | Keep dewow optional/diagnostic in reduced lane until multi-scene domain is validated |
| `set_zero_time.new_zero_time` | Legacy defaults existed; validation policy now guards fixed-zero | Implicit shift can silently destroy global/ROI energy | P0 | Keep validation policy `explicit_only_fixed_zero` or `excluded` for native benchmark lanes |
| Gain windows / strengths (`AGC`, `energy_decay_gain`, `sec_gain`) | Mixed display and interpretable gain variants | Display-driven gain can inflate visual contrast while harming interpretability | P1 | Separate display-oriented and conservative/interpretable gain policies |
| Optional denoise strengths | Often dataset-dependent | Over-smoothing can erase weak targets | P2 | Keep denoise out of preset finalization unless scene diversity is adequate |

## 3) Metric / scoring risk table

| Metric class | Example | Risk | Severity | Control |
|---|---|---|---|---|
| Ground-truth metrics | target preservation, target/background contrast, false-positive proxy within benchmark definition | ROI definition bias can make metrics look strong on one geometry | P1 | Pair metric outputs with ROI overlay review and scene diversity checks |
| Heuristic QC metrics | clipping ratio, saliency, edge preservation, deep-zone visibility | Can reward visually cleaner but geologically misleading outputs | P1 | Never use heuristic-only runs for ground-truth claims |
| Composite candidate score | local weighted score used inside runner | Weights can hide tradeoffs across metrics | P2 | Record metric table and risk flags; do not treat composite score alone as proof |
| Best-at-edge signal | best params at max candidate | Indicates search domain truncation | P0 | Mark as `not_ready_edge_limited`; extend domain or add scenes |
| No-target control signal | non-target energy / false-positive behavior | False positive growth may be missed if no explicit no-target lane | P0 | Require no-target scene in preset-finalization gate |

## 4) Pipeline node classification

| Node | Class | Status in reduced validation lane | Notes |
|---|---|---|---|
| `set_zero_time` | Alignment-correction (high-risk default) | excluded or fixed-zero by policy | Explicit nonzero use requires dedicated justification |
| `dewow` | Optional preprocessing | excluded in current primary reduced lane | Diagnostic side lane only until multi-scene validation |
| `frequency_filter_1d` | Optional preprocessing | controlled/diagnostic | Must respect data context and acquisition band |
| `subtracting_average_2D` | Core structural background suppression | primary lane | Candidate policy must be ratio-based |
| `energy_decay_gain` | Conservative/interpretable gain | primary lane | Preferred for benchmark interpretability currently |
| `sec_gain` / `time_power_gain` | Secondary gain variants | comparative lanes | Useful for ablation, not default by assumption |
| `agcGain` | Display-oriented gain | comparative/visual lane only | Non-amplitude-preserving by policy |

## 5) ROI-aware / expert ROI / no-prior mode distinction

| Mode | Input prior | Allowed claim level | Required output |
|---|---|---|---|
| ROI-aware benchmark mode | Structured ground truth ROI from native benchmark sidecar | Ground-truth bounded claims only | ROI overlays, per-step ROI metrics, false-positive proxy |
| Expert ROI mode | Human-drawn ROI(s) without benchmark truth | Expert-guided comparative claims only | ROI metadata, reviewer note, heuristic + ROI metrics separated |
| No-prior global QC mode | No ROI available | Heuristic QC only | Global QC panel, no-target risk indicators, explicit "no ground truth" banner |

No-prior global QC mode is mandatory for users who cannot provide ROI. It must
never be promoted as ground-truth validation.

## 6) Data-context dependency table

| Data context | Typical source | Dependency risk | Required handling |
|---|---|---|---|
| `native_gprmax_converted` | gprMax `.out` converted package with scenario metadata | Over-trusting single scene and ROI | Keep policy guard on zero-time; track trace count/spacing and ROI validity |
| `gprmax_impulse` synthetic/reference fixtures | Small smoke fixtures | Under-representative scene complexity | Use for smoke/contract only; not preset finalization |
| `uav_gpr_sfcw_field` | Real UAV SFCW CSV | No ground truth, variable noise, acquisition variance | Use no-prior global QC mode or expert ROI mode; no truth claims |
| Mixed / unknown metadata completeness | Missing spacing/time window/sensor sidecars | Silent fallback can hide assumptions | Emit runtime warnings and downgrade claim level |

## 7) GX-004 / GX-005 / GX-006 scene requirements

Before preset finalization, three additional native scenes are required:

| Scene | Purpose | Minimum requirement |
|---|---|---|
| GX-004 | No-target false-positive control | Native conversion chain + explicit no-target ground truth region and false-positive risk readout |
| GX-005 | Multi-target / varying depth | At least two targets with different depth or lateral spacing; ROI and background zones validated |
| GX-006 | Layered/complex background | Layered interfaces or cluttered background to test background suppression robustness |

Per-scene contract requirements:

- Native provenance chain (`model.in` + `.out/.merged.out` + conversion manifest).
- Valid ROI or explicit no-target schema coverage.
- Stable trace spacing metadata for physical window-length reporting.
- Reproducible runner command and source commit binding.

## 8) Preset finalization gate

Preset candidate is allowed only if **all** gates pass:

1. Multi-scene pass:
   GX-003 + GX-004 + GX-005 + GX-006 all evaluated with the same candidate
   policy family.
2. Edge safety:
   no `best_params_at_edge` on final selected candidate domain.
3. Stability:
   selected label/range remains stable across scenes (not oscillating between
   local and full-line extremes).
4. False-positive safety:
   no-target scene does not show unacceptable false-positive growth.
5. Claim boundary compliance:
   no heuristic-only run is used as truth evidence.
6. Reviewer visibility:
   per-scene report must include ROI overlay (or explicit no-target map) and
   candidate risk flags.

If any gate fails, status remains:

- `provisional_single_scene_preset` or
- `not_ready_edge_limited` or
- `not_ready_metric_conflict`.

## 9) Current allowed claims

- Current AutoTune validation chain is reproducible and auditable.
- Relative, trace-count-aware candidate policy is a safer direction than fixed
  absolute windows.
- GX-003 currently favors near-full-line background windows under the reduced
  lane and current metric policy.
- Zero-time validation guard prevents implicit destructive shifts in native
  benchmark lanes.

## 10) Current forbidden claims

- `ntraces=97` or `89-121` is a universal preset.
- AutoTune is globally superior to manual processing.
- Heuristic visual quality alone proves geological correctness.
- Single-scene benchmark behavior generalizes to all UAV-GPR field data.
- Field-data runs without truth ROI can be presented as truth-validated.

## 11) Recommended next task

Run a multi-scene policy validation package:

1. Build GX-004/GX-005/GX-006 native benchmark packages with valid contracts.
2. Replay AT-011 relative candidate policy unchanged across all scenes.
3. Publish a gate checklist report against Section 8.
4. Decide preset status only from cross-scene gate results, not single-scene
   best score.

