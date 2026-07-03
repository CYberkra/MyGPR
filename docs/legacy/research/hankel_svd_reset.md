# Hankel-SVD Denoiser Reset (Xue 2019)

## Reference

- **Paper**: Xue, Y.; Luo, Y.; Yang, F.; Huang, J. "Denoising of GPR Data Based on SVD of Hankel Matrix." *Sensors* 2019, 19, 3807.
- **DOI**: `10.3390/s19173807`
- **Comparison reference**: Golyandina, N.; Korobeynikov, A. "Basic Singular Spectrum Analysis and Forecasting with R." `https://arxiv.org/abs/1309.5050` -- cited here for the classical anti-diagonal averaging / SSA reconstruction pattern used only as an internal comparison path.

This document describes the literature-grounded reset of the `hankel_svd` processing method in GPR GUI.
The old implementation was replaced in-place; the public registry key `hankel_svd` and entry point `method_hankel_svd` are preserved.

---

## Algorithm Overview

The reset follows the three-stage pipeline described in Xue et al. 2019:

1. **Hankel embedding** -- each 1-D trace is turned into a Hankel matrix via sliding window.
2. **Low-rank truncation** -- full SVD is computed, singular values are sorted descending, and the matrix is reconstructed with the top-r singular components.
3. **Trace recovery** -- the denoised trace is extracted from the reconstructed Hankel matrix.

Two recovery modes exist; only the first is the production default:

- **`paper`** (default): first row + last column (excluding the duplicated first element). This is the exact method used in Xue 2019 Section 3.3.
- **`anti_diagonal_average`** (internal benchmark only): averages elements along each anti-diagonal. This is the classical SSA recovery path and is kept for controlled comparison, but it is **not** exposed as a public UI option in v1.

---

## Auto-Selection Details

### FRFCM Window Optimization

When `window_length=0` or `None`, the method auto-selects the Hankel window length.

- **Candidate generation**: up to `max_candidates` windows are drawn from a bounded interval `[ceil(n/8), floor(n/2)]`. The lower bound avoids near-degenerate Hankel matrices; the upper bound keeps the embedding square-ish and limits SVD cost.
- **Scoring**: for each candidate window, a small representative trace subset (calibration traces) is Hankel-embedded and decomposed. The **FRFCM** (Fourth Root of Fourth Central Moment) of the singular-value vector is computed:

  `FRFCM = (mean((sv - mean(sv))^4))^(1/4)`

  A higher FRFCM indicates a clearer separation between signal-related and noise-related singular values, so the window with the highest mean FRFCM score is chosen.
- **Fallback**: if every candidate produces an empty or invalid Hankel, a deterministic fallback `max(2, min(n//4, n-1))` is used and a warning is recorded in metadata.

### Difference-Spectrum Rank Selection

When `rank=0` or `None`, the method auto-selects the truncation rank.

- **Difference spectrum**: `diff[i] = max(sv[i] - sv[i+1], 0)` for sorted singular values.
- **Threshold**: `threshold = rho * mean(diff)`. Only differences at least this large are considered significant.
- **Selection**: among the significant differences, the index with the largest local difference is chosen; the rank is `index + 1`.
- **Fallback**: if no difference exceeds the threshold (or the spectrum is degenerate), a deterministic fallback cap is applied: `min(5, feasible_rank)` in quality mode, `min(3, feasible_rank)` in preview mode.

The scalar `rho` defaults to `0.5` (more conservative than the paper's original `1.0`). It is exposed as an internal kwarg and recorded in metadata. The GUI exposes `aggressiveness` as a user-facing alias for `rho` (range 0.1–2.0, default 0.5); when `aggressiveness` is set it overrides `rho`.

---

## Defaults and Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `window_length` | `0` | `0` or `None` triggers FRFCM auto-selection. Any positive integer uses fixed window length (clamped to `[1, n_samples-1]`). |
| `rank` | `0` | `0` or `None` triggers difference-spectrum auto-selection. Any positive integer uses fixed rank (clamped to feasible maximum). |
| `rho` | `0.5` | Multiplier for the difference-spectrum mean threshold. |
| `aggressiveness` | `0.5` | Alias for `rho` exposed in the GUI (0.1=conservative, 2.0=aggressive). When set, it overrides `rho`. |
| `post_recovery_gain` | `None` | When `None` or absent, applies the legacy heuristic `rank/(rank+1)`. When set to a positive float, that value is used directly as the post-recovery amplitude gain. Set to `1.0` to disable gain suppression entirely. |
| `recovery_mode` | `"paper"` | Internal default; `"anti_diagonal_average"` exists only for benchmark comparison. |
| `preview` | `False` | When `True`, caps candidate windows to 6, calibration traces to 8, and fallback rank to 3. |

---

## Fallback Policies

| Scenario | Behavior |
|----------|----------|
| FRFCM yields no valid score | Deterministic fallback window + metadata warning |
| Difference spectrum yields no significant gap | Deterministic fallback rank + metadata warning |
| Requested window > `n_samples - 1` | Silently clamped + metadata warning |
| Requested rank > feasible max | Silently clamped + metadata warning |
| Flat trace detected | Skip conservative post-recovery gain (gain = 1.0) and record warning |
| NaN / Inf in input | Replaced by interpolation or constant fill; metadata warning |
| Empty input matrix | Zero-filled output + metadata warning |

---

## Paper Recovery vs Anti-Diagonal Comparison

- **Paper-exact recovery** concatenates the first row and the last column of the reconstructed Hankel matrix. It is computationally cheap and reproduces the Xue 2019 figures exactly. It is the default because it matches the reference implementation and avoids the smoothing bias that anti-diagonal averaging can introduce on short traces.
- **Anti-diagonal averaging** is the conventional SSA recovery. It is not universally superior; on GPR traces with strong lateral variability it can blur sharp arrivals. It is retained as an internal comparison mode only.

Do not treat either recovery mode as universally better; the choice depends on trace structure and noise character.

The production `paper` path controls sample recovery only. After recovery, v1 defaults to **no post-recovery amplitude suppression** (`post_recovery_gain = 1.0`). A legacy heuristic `rank / (rank + 1)` is still available by passing `post_recovery_gain=None`, but the default is now `1.0` so weak reflectors are not silently attenuated. To avoid erasing compact high-amplitude reflectors, v1 applies explicit saliency preservation on the strongest sanitized trace samples (threshold lowered to `max(p85(abs(trace)), 0.35 * max_abs)` in the 2026-04-28 tuning) and records `saliency_preservation_fraction_min` / `saliency_preservation_fraction_max`. These are denoising guardrails for the project fixtures, not part of the Xue 2019 recovery formula; any future amplitude normalization or target-preservation change must remain explicit in parameters, metadata, tests, and documentation.

---

## Preview Mode

Preview mode exists to give fast bounded feedback in the GUI quick-look path.

- `max_candidates` drops from 12 to 6
- `max_calibration_traces` drops from 16 to 8
- Window is capped to `max(2, min(n//4, n-1))`
- Auto-rank fallback cap drops from 5 to 3

Preview mode is recorded in metadata (`preview=True`) so that evidence scripts can separate quick-look from quality-run results.

---

## Limitations

- **Per-trace processing**: each column is embedded and decomposed independently. Lateral coherence is not exploited; this is intentional for v1 simplicity.
- **Full SVD only**: the implementation uses `scipy.linalg.svd` with `full_matrices=False`. No truncated/randomized SVD is used, so very large windows can be slow.
- **Single recovery per invocation**: one window and one rank are chosen from the calibration subset and then applied to every trace. Per-trace adaptive selection is not implemented.
- **Bounded but not infallible**: FRFCM and difference spectrum are heuristics. Synthetic evidence shows improvement on average, but pathological inputs can still produce suboptimal parameters.

---

## v1 Non-Goals

The following are explicitly out of scope for the first reset release:

- **2-D / 3-D Hankel embedding** (block Hankel, multi-trace matrices)
- **MSSA** (Multichannel Singular Spectrum Analysis)
- **GPU / numba / CUDA acceleration**
- **Public `recovery_mode` UI enum** in the registry or CLI schema
- **Per-trace adaptive window/rank selection** (the current fixed-per-invocation design is deliberate)

---

## Verification Commands (Definition of Done)

Run these commands from the repo root to verify the reset:

```bash
# 1. Quality regression tests
pytest tests/test_hankel_svd_quality.py -v

# 2. Processing kernel contract tests
pytest tests/test_round2_processing_kernels.py -k hankel_svd -v

# 3. CLI batch profile tests
pytest tests/test_cli_batch_profiles.py -k hankel -v

# 4. GUI preset tests
pytest tests/test_gui_presets.py -k hankel -v

# 5. Auto-tune integration tests
pytest tests/test_auto_tune.py -k hankel -v

# 6. Script-style batch smoke
python tests/test_hankel_batch.py

# 7. Benchmark evidence generation
python scripts/benchmark_hankel_svd_reset.py --mode compare --output .sisyphus/evidence/hankel-svd-reset

# 8. Preflight check
python scripts/preflight_check.py
```

All commands are expected to exit 0 on a clean working tree.

---

## File Locations

- Kernel implementation: `PythonModule/hankel_svd.py`
- Quality tests: `tests/test_hankel_svd_quality.py`
- Contract tests: `tests/test_hankel_svd_contract.py`
- Benchmark / evidence script: `scripts/benchmark_hankel_svd_reset.py`
- Registry / presets: `core/methods_registry.py`, `core/preset_profiles.py`
