"""Native global spectral and low-rank processing algorithms.

The functions in this module are independent from the historical ``core`` and
``PythonModule`` registries.  Exact solvers are retained for small matrices,
while deterministic randomized low-rank factorization is used for large
matrices so project execution can remain bounded and cancellable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import fft as sp_fft
from scipy import ndimage
from scipy.linalg import svd as scipy_svd

from mygpr.infrastructure.processing.algorithms.common import (
    as_float,
    as_int,
    ensure_matrix,
    normalize_output,
    warning,
)

CancelCheck = Callable[[], bool]


class GlobalProcessingCancelled(RuntimeError):
    """Raised when a global transform observes cooperative cancellation."""


def _cancel_check(params: dict[str, Any]) -> CancelCheck:
    checker = params.get("cancel_checker")
    if callable(checker):
        return checker
    context = params.get("_execution_context")
    if context is not None and hasattr(context, "is_cancelled"):
        return context.is_cancelled
    return lambda: False


def _poll(checker: CancelCheck, operation: str) -> None:
    if checker():
        raise GlobalProcessingCancelled(f"processing cancelled during {operation}")


@dataclass(frozen=True, slots=True)
class LowRankFactors:
    left: np.ndarray
    singular_values: np.ndarray
    right: np.ndarray
    solver: str
    requested_rank: int
    effective_rank: int


def _randomized_svd(
    matrix: np.ndarray,
    rank: int,
    *,
    oversampling: int,
    power_iterations: int,
    seed: int,
    checker: CancelCheck,
) -> LowRankFactors:
    """Compute leading singular triplets without materialising a float64 copy."""
    rows, cols = matrix.shape
    max_rank = min(rows, cols)
    requested = max(1, min(int(rank), max_rank))
    sketch_rank = min(max_rank, requested + max(2, int(oversampling)))
    rng = np.random.default_rng(int(seed))
    omega = rng.standard_normal((cols, sketch_rank), dtype=np.float32)
    _poll(checker, "randomized SVD range initialization")
    sample = np.asarray(matrix @ omega, dtype=np.float64)
    q, _ = np.linalg.qr(sample, mode="reduced")
    for iteration in range(max(0, int(power_iterations))):
        _poll(checker, f"randomized SVD power iteration {iteration + 1}")
        z = np.asarray(matrix.T @ q, dtype=np.float64)
        qz, _ = np.linalg.qr(z, mode="reduced")
        sample = np.asarray(matrix @ qz, dtype=np.float64)
        q, _ = np.linalg.qr(sample, mode="reduced")
    _poll(checker, "randomized SVD projection")
    projected = np.asarray(q.T @ matrix, dtype=np.float64)
    ub, singular_values, right = scipy_svd(
        projected,
        full_matrices=False,
        check_finite=False,
        overwrite_a=True,
    )
    left = q @ ub
    effective = min(requested, singular_values.size)
    return LowRankFactors(
        left=np.asarray(left[:, :effective], dtype=np.float64),
        singular_values=np.asarray(singular_values[:effective], dtype=np.float64),
        right=np.asarray(right[:effective, :], dtype=np.float64),
        solver="randomized",
        requested_rank=requested,
        effective_rank=effective,
    )


def leading_svd(
    matrix: np.ndarray,
    rank: int,
    params: dict[str, Any],
) -> LowRankFactors:
    """Resolve exact versus deterministic randomized SVD."""
    rows, cols = matrix.shape
    max_rank = min(rows, cols)
    requested = max(1, min(int(rank), max_rank))
    exact_max_elements = max(1, as_int(params.get("exact_max_elements"), 2_000_000))
    exact_rank_ratio = float(np.clip(as_float(params.get("exact_rank_ratio"), 0.35), 0.05, 1.0))
    checker = _cancel_check(params)
    force_solver = str(params.get("solver") or "auto").strip().lower()
    use_exact = force_solver == "exact" or (
        force_solver == "auto"
        and matrix.size <= exact_max_elements
        and requested >= max(1, int(max_rank * exact_rank_ratio))
    )
    # For small matrices exact SVD remains the compatibility baseline even for
    # low requested ranks. This keeps historical numerical regression stable.
    if force_solver == "auto" and matrix.size <= exact_max_elements:
        use_exact = True
    if use_exact:
        _poll(checker, "exact SVD")
        work = np.asarray(matrix, dtype=np.float64)
        left, singular_values, right = scipy_svd(
            work,
            full_matrices=False,
            check_finite=False,
            overwrite_a=False,
        )
        effective = min(requested, singular_values.size)
        return LowRankFactors(
            left=left[:, :effective],
            singular_values=singular_values[:effective],
            right=right[:effective, :],
            solver="exact",
            requested_rank=requested,
            effective_rank=effective,
        )
    return _randomized_svd(
        matrix,
        requested,
        oversampling=max(2, as_int(params.get("oversampling"), 8)),
        power_iterations=max(0, as_int(params.get("power_iterations"), 1)),
        seed=as_int(params.get("random_seed"), 0),
        checker=checker,
    )


def _factor_metadata(factors: LowRankFactors) -> dict[str, Any]:
    return {
        "solver": factors.solver,
        "requested_rank": factors.requested_rank,
        "effective_rank": factors.effective_rank,
        "singular_values": factors.singular_values.astype(float).tolist(),
    }


def method_svd_background_native(
    data: Any,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    rank = max(1, as_int(params.get("rank"), 1))
    factors = leading_svd(arr, rank, params)
    background = (factors.left * factors.singular_values) @ factors.right
    result = arr - background
    metadata = {
        "method": "svd_bg",
        "removed_rank": factors.effective_rank,
        **_factor_metadata(factors),
    }
    if factors.solver == "randomized":
        warnings.append(
            warning(
                "randomized_svd_used",
                "矩阵超过精确 SVD 阈值，已使用确定性随机低秩分解。",
                "svd_bg",
                effective_rank=factors.effective_rank,
            )
        )
    return normalize_output("svd_bg", result, metadata, warnings)


def method_svd_subspace_native(
    data: Any,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    max_rank = min(arr.shape)
    rank_start = max(1, as_int(params.get("rank_start"), 1))
    rank_end = min(max_rank, max(rank_start, as_int(params.get("rank_end"), 2)))
    factors = leading_svd(arr, rank_end, params)
    start = min(rank_start - 1, factors.effective_rank)
    end = min(rank_end, factors.effective_rank)
    if start >= end:
        result = np.zeros_like(arr)
    else:
        result = (
            factors.left[:, start:end]
            * factors.singular_values[start:end]
        ) @ factors.right[start:end, :]
    metadata = {
        "method": "svd_subspace",
        "rank_start": rank_start,
        "rank_end": rank_end,
        "rank_count": max(0, end - start),
        **_factor_metadata(factors),
    }
    if factors.solver == "randomized":
        warnings.append(
            warning(
                "randomized_svd_used",
                "矩阵超过精确 SVD 阈值，已使用确定性随机子空间分解。",
                "svd_subspace",
                effective_rank=factors.effective_rank,
            )
        )
    return normalize_output("svd_subspace", result, metadata, warnings)


def method_fk_filter_native(
    data: Any,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    angle_low = float(np.clip(as_float(params.get("angle_low"), 10.0), 0.0, 90.0))
    angle_high = float(np.clip(as_float(params.get("angle_high"), 65.0), 0.0, 90.0))
    if angle_low > angle_high:
        angle_low, angle_high = angle_high, angle_low
        warnings.append(
            warning(
                "fk_angle_range_reordered",
                "F-K 角度下限高于上限，已自动交换。",
                "fk_filter",
            )
        )
    taper_width = max(0.0, as_float(params.get("taper_width"), 5.0))
    stop_gain = float(np.clip(as_float(params.get("stop_gain"), 0.05), 0.0, 1.0))
    checker = _cancel_check(params)
    _poll(checker, "F-K forward transform")
    spectrum = sp_fft.fftshift(sp_fft.fft2(arr))
    rows, cols = spectrum.shape
    ky = sp_fft.fftshift(sp_fft.fftfreq(rows))
    kx = sp_fft.fftshift(sp_fft.fftfreq(cols))
    angle = np.degrees(np.arctan2(np.abs(ky[:, None]), np.abs(kx[None, :])))
    band = (angle >= angle_low) & (angle <= angle_high)
    mask = np.ones_like(angle, dtype=np.float32)
    if taper_width > 0.0:
        distance = np.minimum(np.abs(angle - angle_low), np.abs(angle - angle_high))
        mask[band] = stop_gain
        taper = band & (distance < taper_width)
        if np.any(taper):
            mask[taper] = 1.0 - np.exp(
                -(distance[taper] ** 2) / (2.0 * taper_width**2)
            )
    else:
        mask[band] = stop_gain
    _poll(checker, "F-K inverse transform")
    result = np.real(sp_fft.ifft2(sp_fft.ifftshift(spectrum * mask)))
    metadata = {
        "method": "fk_filter",
        "angle_low": angle_low,
        "angle_high": angle_high,
        "taper_width": taper_width,
        "stop_gain": stop_gain,
        "suppressed_fraction": float(np.mean(mask < 0.999)),
        "solver": "scipy-pocketfft-cpu",
    }
    return normalize_output("fk_filter", result, metadata, warnings)


_STOLT_AXIS_CACHE: dict[tuple[int, int, float, float], tuple[np.ndarray, np.ndarray]] = {}


def _stolt_axes(nt_pad: int, nx_pad: int, dt: float, dx: float) -> tuple[np.ndarray, np.ndarray]:
    key = (int(nt_pad), int(nx_pad), float(dt), float(dx))
    cached = _STOLT_AXIS_CACHE.get(key)
    if cached is not None:
        return cached
    omega = (2.0 * np.pi * sp_fft.rfftfreq(nt_pad, d=dt)).astype(np.float32)
    kx = (2.0 * np.pi * sp_fft.fftshift(sp_fft.fftfreq(nx_pad, d=dx))).astype(np.float32)
    if len(_STOLT_AXIS_CACHE) > 16:
        _STOLT_AXIS_CACHE.clear()
    _STOLT_AXIS_CACHE[key] = (omega, kx)
    return omega, kx


def _stolt_metadata(
    *, dx: float, dt: float, velocity: float, pad_x: int, pad_t: int,
    nt_pad: int, nx_pad: int, jacobian_power: float, obliquity_power: float,
    mask_softness: float, kz_smooth: int, depth_gain: float,
    depth_gain_power: float, clip_percentile: float, coverage: int, total: int,
) -> dict[str, Any]:
    return {
        "method": "stolt_migration",
        "mapped_params": {
            "dx": dx, "dt": dt, "v": velocity, "pad_x": pad_x, "pad_t": pad_t,
            "nt_pad": nt_pad, "nx_pad": nx_pad,
            "jacobian_power": jacobian_power, "obliquity_power": obliquity_power,
            "mask_softness": mask_softness, "kz_smooth": kz_smooth,
            "depth_gain": depth_gain, "depth_gain_power": depth_gain_power,
            "clip_percentile": clip_percentile,
            "mapped_coverage": float(coverage / max(total, 1)),
        },
        "solver": "scipy-pocketfft-cpu",
    }


def method_stolt_migration_native(
    data: Any,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    arr, warnings = ensure_matrix(data)
    arr32 = np.asarray(arr, dtype=np.float32)
    nt, nx = arr32.shape
    if nt < 2 or nx < 2:
        raise ValueError("Stolt migration requires at least 2x2 samples")
    dx = as_float(params.get("dx"), 0.05)
    dt = as_float(params.get("dt"), 0.1)
    velocity = as_float(params.get("v"), 0.10)
    if dx <= 0.0 or dt <= 0.0 or velocity <= 0.0:
        raise ValueError("dx, dt and v must be positive")
    pad_x = max(0, as_int(params.get("pad_x"), 1))
    pad_t = max(0, as_int(params.get("pad_t"), 1))
    # 教科书 Stolt 倾角/斜射因子 kz/kmag（Margrave/Yilmaz）：jacobian 幂次为 0 时
    # obliquity=1.0 才是完整幅度校正；jacobian 与 obliquity 同形（kz/omega ∝ obliquity），
    # 二者同时开启会重复施加同一倾角滤波，故 jacobian 默认保持关闭。
    jacobian_power = as_float(params.get("stolt_jacobian_power"), 0.0)
    obliquity_power = as_float(params.get("stolt_obliquity_power"), 1.0)
    mask_softness = max(1.0e-4, as_float(params.get("stolt_mask_softness"), 0.03))
    kz_smooth = max(1, as_int(params.get("stolt_kz_smooth"), 3))
    depth_gain = max(0.0, as_float(params.get("stolt_depth_gain"), 0.0))
    depth_gain_power = max(0.1, as_float(params.get("stolt_depth_gain_power"), 1.1))
    clip_percentile = float(np.clip(as_float(params.get("stolt_clip_percentile"), 100.0), 95.0, 100.0))
    checker = _cancel_check(params)

    nt_pad = max(nt, int(2 ** np.ceil(np.log2(nt * (1 + pad_t)))))
    nx_pad = max(nx, int(2 ** np.ceil(np.log2(nx * (1 + pad_x)))))
    work = np.zeros((nt_pad, nx_pad), dtype=np.float32)
    work[:nt, :nx] = arr32
    _poll(checker, "Stolt forward transform")
    spec_tx = sp_fft.rfft(work, axis=0)
    spec_wkx = sp_fft.fftshift(sp_fft.fft(spec_tx, axis=1), axes=1)
    omega, kx = _stolt_axes(nt_pad, nx_pad, dt, dx)
    omega_max = float(omega[-1])
    kz = np.linspace(0.0, 2.0 * omega_max / velocity, omega.size, dtype=np.float32)
    mapped = np.zeros_like(spec_wkx, dtype=np.complex64)
    coverage = 0
    total = mapped.size
    eps = 1.0e-6
    for index, kx_value in enumerate(kx):
        if index % 8 == 0:
            _poll(checker, "Stolt interpolation")
        kmag = np.sqrt(float(kx_value) ** 2 + kz * kz, dtype=np.float32)
        omega_src = (0.5 * velocity * kmag).astype(np.float32)
        valid = omega_src <= omega_max
        if not np.any(valid):
            continue
        omega_valid = omega_src[valid]
        kz_valid = kz[valid]
        column = spec_wkx[:, index]
        sample = np.interp(omega_valid, omega, column.real) + 1j * np.interp(
            omega_valid, omega, column.imag
        )
        jacobian = np.clip((kz_valid + eps) / (omega_valid + eps), 0.0, 8.0)
        amplitude = np.power(jacobian, max(0.0, jacobian_power))
        obliquity = np.clip(kz_valid / (kmag[valid] + eps), 0.0, 1.0)
        if obliquity_power > 0.0:
            amplitude *= np.power(obliquity, obliquity_power)
        edge = (omega_max - omega_valid) / (omega_max + eps)
        soft = np.where(edge >= mask_softness, 1.0, np.clip(edge / mask_softness, 0.0, 1.0))
        mapped[valid, index] = np.asarray(sample * amplitude * soft, dtype=np.complex64)
        coverage += int(np.count_nonzero(valid))
    if kz_smooth > 1:
        mapped = (
            ndimage.uniform_filter1d(mapped.real, size=kz_smooth, axis=0, mode="nearest")
            + 1j * ndimage.uniform_filter1d(mapped.imag, size=kz_smooth, axis=0, mode="nearest")
        ).astype(np.complex64)
    _poll(checker, "Stolt inverse transform")
    migrated = sp_fft.irfft(
        sp_fft.ifft(sp_fft.ifftshift(mapped, axes=1), axis=1),
        n=nt_pad,
        axis=0,
    )[:nt, :nx]
    migrated = np.asarray(migrated, dtype=np.float32)
    if depth_gain > 0.0:
        depth = np.linspace(0.0, 1.0, nt, dtype=np.float32)
        migrated *= (1.0 + depth_gain * np.power(depth, depth_gain_power))[:, None]
    if clip_percentile < 99.999:
        threshold = float(np.percentile(np.abs(migrated), clip_percentile))
        if threshold > 0.0:
            migrated = np.asarray(threshold * np.tanh(migrated / threshold), dtype=np.float32)
    metadata = _stolt_metadata(
        dx=dx, dt=dt, velocity=velocity, pad_x=pad_x, pad_t=pad_t,
        nt_pad=nt_pad, nx_pad=nx_pad, jacobian_power=jacobian_power,
        obliquity_power=obliquity_power, mask_softness=mask_softness,
        kz_smooth=kz_smooth, depth_gain=depth_gain,
        depth_gain_power=depth_gain_power, clip_percentile=clip_percentile,
        coverage=coverage, total=total,
    )
    return normalize_output("stolt_migration", migrated, metadata, warnings)


def _write_low_rank_blocks(
    source: np.ndarray,
    output: np.ndarray,
    factors: LowRankFactors,
    *,
    start_rank: int,
    end_rank: int,
    subtract: bool,
    block_rows: int,
    checker: CancelCheck,
    progress: Callable[[int, int, str], None] | None = None,
) -> None:
    spans = list(range(0, source.shape[0], max(1, int(block_rows))))
    for index, start in enumerate(spans, start=1):
        _poll(checker, "low-rank file-backed reconstruction")
        end = min(start + max(1, int(block_rows)), source.shape[0])
        reconstructed = (
            factors.left[start:end, start_rank:end_rank]
            * factors.singular_values[start_rank:end_rank]
        ) @ factors.right[start_rank:end_rank, :]
        if subtract:
            output[start:end] = np.asarray(source[start:end], dtype=np.float32) - reconstructed
        else:
            output[start:end] = reconstructed
        if hasattr(output, "flush"):
            output.flush()
        if progress is not None:
            progress(index, len(spans), f"低秩重建块 {index}/{len(spans)}")


def file_svd_background_native(
    source: np.ndarray,
    output: np.ndarray,
    params: dict[str, Any],
    context: Any,
    block_rows: int,
) -> dict[str, Any]:
    rank = max(1, as_int(params.get("rank"), 1))
    factors = leading_svd(source, rank, {**params, "_execution_context": context})
    _write_low_rank_blocks(
        source,
        output,
        factors,
        start_rank=0,
        end_rank=factors.effective_rank,
        subtract=True,
        block_rows=block_rows,
        checker=context.is_cancelled,
        progress=context.report_progress,
    )
    return {
        "method": "svd_bg",
        "removed_rank": factors.effective_rank,
        **_factor_metadata(factors),
        "execution_mode": "file_backed_low_rank",
    }


def file_svd_subspace_native(
    source: np.ndarray,
    output: np.ndarray,
    params: dict[str, Any],
    context: Any,
    block_rows: int,
) -> dict[str, Any]:
    rank_start = max(1, as_int(params.get("rank_start"), 1))
    rank_end = min(min(source.shape), max(rank_start, as_int(params.get("rank_end"), 2)))
    factors = leading_svd(source, rank_end, {**params, "_execution_context": context})
    start = min(rank_start - 1, factors.effective_rank)
    end = min(rank_end, factors.effective_rank)
    if start >= end:
        output[:] = 0.0
        if hasattr(output, "flush"):
            output.flush()
    else:
        _write_low_rank_blocks(
            source,
            output,
            factors,
            start_rank=start,
            end_rank=end,
            subtract=False,
            block_rows=block_rows,
            checker=context.is_cancelled,
            progress=context.report_progress,
        )
    return {
        "method": "svd_subspace",
        "rank_start": rank_start,
        "rank_end": rank_end,
        "rank_count": max(0, end - start),
        **_factor_metadata(factors),
        "execution_mode": "file_backed_low_rank",
    }
