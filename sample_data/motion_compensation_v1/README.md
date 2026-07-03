# motion_compensation_v1 benchmark evidence

## Purpose

`motion_compensation_v1` is the deterministic synthetic UAV-GPR benchmark used to prove the full V1 motion-compensation chain end to end. It covers the five V1 error families only:

1. height / time-shift distortion
2. uneven trace spacing / speed error
3. lateral trajectory drift
4. attitude / footprint offset
5. periodic vibration-style banding background

## Verified V1 semantics

- `trajectory_smoothing` is a geometry / metadata stage. It smooths trajectory coordinates and returns updated `trace_metadata`, but does **not** rewrite the B-scan amplitude matrix.
- `motion_compensation_speed` is a data-rewriting stage. It resamples the B-scan to an equal-distance trace axis and also outputs resampled metadata.
- `motion_compensation_attitude` is a geometry / footprint stage in V1. It updates local geometry plus footprint-related metadata, but does **not** rewrite the B-scan amplitude matrix.
- `motion_compensation_height` is a data-rewriting stage. It applies amplitude normalization plus per-trace time-shift interpolation.
- `motion_compensation_vibration` is a conservative signal-domain suppression stage. It attenuates periodic lateral banding while protecting salient target rows; it is explicitly **not** RPM notch filtering.

Height-specific caveat:

- The current V1 implementation and benchmark keep `wave_speed_m_per_ns = 0.1` as an implementation constant for deterministic alignment.
- Formal physics write-ups should distinguish this implementation constant from the free-space / air-path propagation speed (`~0.3 m/ns`) instead of treating them as automatically identical.

## Exact GUI path

1. Run `python app_qt.py`
2. Import the benchmark-compatible line data you want to inspect in the main workbench
3. In the preset selector, apply preset key `motion_compensation_v1` (`运动补偿 V1`)
4. Run the chain in this exact order:
   - `trajectory_smoothing`
   - `motion_compensation_speed`
   - `motion_compensation_attitude`
   - `motion_compensation_height`
   - `motion_compensation_vibration`

This is the **current V1 implementation order**. A future physics-first workflow could choose a different ordering, but the benchmark evidence in this repo is pinned to the order above.

GUI use is for visual confirmation only. The deterministic benchmark evidence below is produced by the CLI/export path.

## CLI commands

Validate the benchmark config:

```bash
python cli_batch.py validate --config config/motion_compensation_v1_benchmark.json
```

Run the deterministic benchmark export:

```bash
python cli_batch.py run --config config/motion_compensation_v1_benchmark.json
```

The output directory is repo-relative and Windows-safe:

```text
output/motion_compensation_v1_benchmark/
```

## Expected artifacts

Running the config above must emit these machine-checkable files:

- `output/motion_compensation_v1_benchmark/before.png`
- `output/motion_compensation_v1_benchmark/after.png`
- `output/motion_compensation_v1_benchmark/difference.png`
- `output/motion_compensation_v1_benchmark/motion_metrics.json`
- `output/motion_compensation_v1_benchmark/corrected_trace_metadata.csv`
- `output/motion_compensation_v1_benchmark/motion_compensation_v1-summary.json`

## Objective metrics

`motion_metrics.json` records raw vs corrected metrics plus boolean objective checks. The benchmark is considered successful when the corrected result improves these five motion-error families relative to the raw benchmark input:

- `raw_ridge_rmse_samples` decreases
- `trace_spacing_std_m` decreases
- `path_rmse_m` decreases
- `footprint_rmse_m` decreases
- `periodic_banding_ratio` decreases

Additional guardrail:

- `target_preservation_ratio` must stay at least as high as the raw benchmark value

## Explicit non-V1 exclusions

The V1 benchmark and preset intentionally do **not** include:

- autofocus
- DEM fusion / terrain coupling
- antenna-pattern inversion
- RPM notch filtering
- manual-inspection-only acceptance in place of objective metrics
