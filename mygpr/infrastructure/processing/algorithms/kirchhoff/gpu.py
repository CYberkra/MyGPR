"""Optional CuPy backend for native Kirchhoff migration.

CuPy is imported only inside execution functions so the backend package remains
importable on CPU-only systems.
"""
from __future__ import annotations

import math
import logging

import numpy as np

from mygpr.infrastructure.processing.algorithms.kirchhoff.shared import (
    _build_kirchhoff_display_header_updates,
    _build_kirchhoff_header_updates,
    _compute_travel_time,
    _postprocess_kir_profile,
    _require_trace_vector,
    _resample_to_input_grid,
    _resolve_elevation_axis_top_m,
    _ricker,
    _setup_grid,
)

_LOGGER = logging.getLogger(__name__)


def _run_kirchhoff_gpu(
    arr: np.ndarray,
    freq: float,
    depth: float,
    velocity: float,
    alpha: float,
    weight: float,
    num_cal: int,
    topo_cor: int,
    hei_cor: int,
    length_m: float,
    time_window_ns: float,
    cancel_checker,
    header_info: dict,
    trace_metadata: dict,
):
    """Run Kirchhoff migration on GPU with CuPy."""
    import cupy as cp
    import time

    # Timing dictionary for profiling
    timings = {}
    t_start = time.perf_counter()

    factor, nx_matrix, nt_matrix, nz_matrix, dx, dt_s = _setup_grid(
        length_m, depth, freq, time_window_ns, num_cal
    )
    dz = dx
    resized_traces = max(1, int(math.ceil(length_m / dx)))

    timings["setup_grid"] = (time.perf_counter() - t_start) * 1000
    t0 = time.perf_counter()

    # Move data to GPU
    arr_gpu = cp.asarray(arr, dtype=cp.float64)

    # Resize on GPU
    zoom_factors = (nt_matrix / arr.shape[0], resized_traces / arr.shape[1])
    data_resized = _cupy_zoom(arr_gpu, zoom_factors, order=1)

    # Apply alpha power
    signs = cp.sign(data_resized)
    data_resized = cp.power(cp.abs(data_resized), alpha) * signs

    timings["data_prep"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()

    correction_info = {
        "height_mute_applied": False,
        "height_correction_stage": "none",
        "topography_applied": False,
        "topography_stage": "none",
        "n_mute_min": 0,
        "n_mute_max": 0,
        "topography_pad_rows": 0,
    }
    elevation_axis_top_m = None

    if hei_cor == 2:
        flight_height = _require_trace_vector(
            trace_metadata,
            "flight_height_m",
            data_resized.shape[1],
            "hei_cor=2 需要 flight_height_m 航空元数据",
        )
        flight_height_gpu = cp.asarray(flight_height, dtype=cp.float64)
        data_resized, n_mute_arr = _apply_height_correction_time_gpu(
            data_resized, flight_height_gpu, dt_s
        )
        correction_info.update(
            {
                "height_mute_applied": True,
                "height_correction_stage": "pre",
                "n_mute_min": int(cp.min(n_mute_arr).get()) if n_mute_arr.size else 0,
                "n_mute_max": int(cp.max(n_mute_arr).get()) if n_mute_arr.size else 0,
            }
        )

    if topo_cor == 2:
        elevation_axis_top_m = _resolve_elevation_axis_top_m(
            header_info, trace_metadata, "ground_elevation_m"
        )
        ground_elevation = _require_trace_vector(
            trace_metadata,
            "ground_elevation_m",
            data_resized.shape[1],
            "topo_cor=2 需要 ground_elevation_m 航空元数据",
        )
        ground_elevation_gpu = cp.asarray(ground_elevation, dtype=cp.float64)
        data_resized, pad_rows, _ = _apply_topography_correction_depth_gpu(
            data_resized, ground_elevation_gpu, dz
        )
        correction_info.update(
            {
                "topography_applied": True,
                "topography_stage": "pre",
                "topography_pad_rows": int(pad_rows),
            }
        )

    timings["pre_correction"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()

    # Create velocity model on GPU
    velocity_model = cp.full((nx_matrix, nz_matrix), velocity, dtype=cp.float64)
    velocity_model = _cupy_smooth2a(velocity_model, nr=10)

    # Compute travel time on CPU (kept on CPU for stability)
    travel_time = _compute_travel_time(
        cp.asnumpy(velocity_model),
        nz_matrix,
        nx_matrix,
        dx,
        dt_s,
        cancel_checker=cancel_checker,
    )

    timings["travel_time"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()

    # Stack on GPU
    shot_results, kstart = _run_kirchhoff_stack_gpu(
        travel_time,
        data_resized,
        freq,
        dt_s,
        nz_matrix,
        nx_matrix,
        cancel_checker=cancel_checker,
    )

    timings["kirchhoff_stack"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()

    # Assemble on GPU
    kir_profile = _assemble_kir_profile_gpu(
        shot_results, nx_matrix, length_m, depth, dx
    )

    timings["assemble"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()

    # 保持和 2026-04-09(kir) / CaGPR 一致：GPU 主链完成后回到 CPU 做同一套后处理。
    migrated = _postprocess_kir_profile(
        cp.asnumpy(kir_profile).astype(np.float64, copy=False), weight=weight
    )

    timings["postprocess"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()

    imaging_shape = (int(migrated.shape[0]), int(migrated.shape[1]))
    display_migrated = migrated.copy()
    output_migrated = migrated.copy()

    if hei_cor == 1:
        flight_height = _require_trace_vector(
            trace_metadata,
            "flight_height_m",
            arr.shape[1],
            "hei_cor=1 需要 flight_height_m 航空元数据",
        )
        flight_height_gpu = cp.asarray(flight_height, dtype=cp.float64)
        display_migrated, n_mute_arr = _apply_height_correction_depth_gpu(
            display_migrated, flight_height_gpu, dz, velocity
        )
        output_migrated, _ = _apply_height_correction_depth_gpu(
            output_migrated, flight_height_gpu, dz, velocity
        )
        correction_info.update(
            {
                "height_mute_applied": True,
                "height_correction_stage": "post",
                "n_mute_min": int(cp.min(n_mute_arr).get()) if n_mute_arr.size else 0,
                "n_mute_max": int(cp.max(n_mute_arr).get()) if n_mute_arr.size else 0,
            }
        )

    if topo_cor == 1:
        elevation_axis_top_m = _resolve_elevation_axis_top_m(
            header_info, trace_metadata, "ground_elevation_m"
        )
        ground_elevation = _require_trace_vector(
            trace_metadata,
            "ground_elevation_m",
            arr.shape[1],
            "topo_cor=1 需要 ground_elevation_m 航空元数据",
        )
        ground_elevation_gpu = cp.asarray(ground_elevation, dtype=cp.float64)
        display_migrated, pad_rows, _ = _apply_topography_correction_depth_gpu(
            display_migrated, ground_elevation_gpu, dz
        )
        output_migrated, _, _ = _apply_topography_correction_depth_gpu(
            output_migrated, ground_elevation_gpu, dz
        )
        correction_info.update(
            {
                "topography_applied": True,
                "topography_stage": "post",
                "topography_pad_rows": int(pad_rows),
            }
        )

    timings["post_correction"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()

    # Move to CPU for final output
    display_data = cp.asnumpy(display_migrated).astype(np.float32)
    output_data = cp.asnumpy(output_migrated)

    timings["d2h_transfer"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()

    display_header_info_updates = _build_kirchhoff_display_header_updates(
        header_info,
        display_data.shape,
        dz,
        dx,
        topo_cor=topo_cor,
        elevation_axis_top_m=elevation_axis_top_m,
    )

    output, output_dz, output_dx = _resample_to_input_grid(
        output_data, arr.shape, dz, dx
    )

    timings["resample"] = (time.perf_counter() - t0) * 1000

    total_time = (time.perf_counter() - t_start) * 1000
    timings["total"] = total_time

    # 计时 breakdown 走 logger（不 print，避免 GUI 模式下丢失且污染 stdout）
    if _LOGGER.isEnabledFor(logging.INFO):
        lines = ["=" * 60, "Kirchhoff GPU Timing Breakdown", "=" * 60]
        for phase, ms in timings.items():
            if phase != "total":
                pct = (ms / total_time) * 100
                lines.append(f"  {phase:20s}: {ms:8.1f} ms ({pct:5.1f}%)")
        lines.append("-" * 60)
        lines.append(f"  {'TOTAL':20s}: {total_time:8.1f} ms")
        lines.append("=" * 60)
        _LOGGER.info("\n" + "\n".join(lines))

    corrected_shape = (int(output_migrated.shape[0]), int(output_migrated.shape[1]))

    header_info_updates = _build_kirchhoff_header_updates(
        header_info,
        output.shape,
        output_dz,
        output_dx,
        topo_cor=topo_cor,
        elevation_axis_top_m=elevation_axis_top_m,
    )

    return output.astype(np.float32, copy=False), {
        "mapped_params": {
            "freq": freq,
            "depth": depth,
            "v": velocity,
            "alpha": alpha,
            "weight": weight,
            "num_cal": num_cal,
            "topo_cor": topo_cor,
            "hei_cor": hei_cor,
            "length_m": length_m,
            "time_window_ns": time_window_ns,
            "factor": int(factor),
            "dx": float(dx),
            "dz": float(dz),
            "output_dx": float(output_dx),
            "output_dz": float(output_dz),
            "dt_s": float(dt_s),
            "nt_matrix": int(nt_matrix),
            "nx_matrix": int(nx_matrix),
            "nz_matrix": int(nz_matrix),
            "imaging_shape": imaging_shape,
            "corrected_shape": corrected_shape,
            "output_shape": (int(output.shape[0]), int(output.shape[1])),
            "ricker_trim_samples": int(kstart),
            "note": "cagpr_kirchhoff_main_chain_resampled_gpu_fast",
            **correction_info,
            **{f"time_{k}": v for k, v in timings.items()},
        },
        "header_info_updates": header_info_updates,
        "display_data": display_data,
        "display_header_info_updates": display_header_info_updates,
    }


def _cupy_zoom(arr, zoom_factors, order=1):
    """GPU version of scipy.ndimage.zoom using cupyx."""
    import cupyx.scipy.ndimage as cupy_ndimage

    return cupy_ndimage.zoom(arr, zoom_factors, order=order)


def _cupy_smooth2a(matrix_in, nr=10, nc=None):
    """GPU version of _smooth2a."""
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage

    if nc is None:
        nc = nr
    nr = max(0, int(nr))
    nc = max(0, int(nc))
    if nr == 0 and nc == 0:
        return cp.array(matrix_in, copy=True)

    arr = cp.asarray(matrix_in, dtype=cp.float64)
    valid = cp.isfinite(arr).astype(cp.float64)
    work = cp.nan_to_num(arr, nan=0.0)
    size = (2 * nr + 1, 2 * nc + 1)

    num = cupy_ndimage.uniform_filter(work, size=size, mode="nearest") * (
        size[0] * size[1]
    )
    den = cupy_ndimage.uniform_filter(valid, size=size, mode="nearest") * (
        size[0] * size[1]
    )

    out = cp.zeros_like(num)
    mask = den > 0
    out[mask] = num[mask] / cp.maximum(den[mask], 1.0e-12)
    return out


def _run_kirchhoff_stack_gpu(
    travel_time: np.ndarray,
    data_resized,
    freq: float,
    dt_s: float,
    nz_matrix: int,
    nx_matrix: int,
    *,
    cancel_checker=None,
):
    """GPU version of _run_kirchhoff_stack."""
    import cupy as cp

    nt = data_resized.shape[0]
    signal, _ = _ricker(freq, nt, dt_s)
    kstart = int(np.argmax(signal))

    data2 = data_resized[kstart:, :]
    padded = cp.zeros((data2.shape[0] + 3 * nt, data2.shape[1]), dtype=cp.float64)
    padded[: data2.shape[0], :] = data2

    x_shot_grid = np.arange(1, data_resized.shape[1] + 1)
    shot_results = []

    travel_time_gpu = cp.asarray(travel_time, dtype=cp.float64)

    for shot_idx, xs in enumerate(x_shot_grid, start=1):
        if cancel_checker is not None and bool(cancel_checker()):
            raise Exception("用户已取消（Kirchhoff迁移）")

        result = _migrate_gpu(
            travel_time_gpu,
            x_shot_grid[shot_idx - 1],
            x_shot_grid[shot_idx - 1],
            padded,
            dt_s,
            nz_matrix,
            nx_matrix,
            xs,
        )
        shot_results.append(result)

        if cancel_checker is not None and shot_idx % 8 == 0 and bool(cancel_checker()):
            raise Exception("用户已取消（Kirchhoff炮点累加）")

    return shot_results, kstart


def _migrate_gpu(
    travel_time,
    x_rec_grid,
    x_shot_and_rec_grid,
    shot,
    dt_s: float,
    nz: int,
    nx: int,
    ixs: int,
):
    """GPU version of _migrate."""
    import cupy as cp

    image = cp.zeros((nz, nx), dtype=cp.float64)
    x_rec_grid = np.atleast_1d(x_rec_grid)

    for ixr in range(len(x_rec_grid)):
        xr = x_rec_grid[ixr]
        matches = np.where(np.atleast_1d(x_shot_and_rec_grid) == np.atleast_1d(xr))[0]
        if len(matches) == 0:
            continue
        idx_x_rec = matches[0]

        time_indices = _shot_to_rec_time_gpu(travel_time, ixs - 1, idx_x_rec, dt_s, nx)
        max_it = int(cp.max(time_indices).get())

        if max_it >= shot.shape[0]:
            pad_rows = max_it - shot.shape[0] + 1
            padded_shot = cp.pad(shot, ((0, pad_rows), (0, 0)), mode="constant")
        else:
            padded_shot = shot

        migrated_values = padded_shot[time_indices, ixs - 1]
        image = cp.reshape(migrated_values, (nz, nx))

    return image


def _shot_to_rec_time_gpu(travel_time, ixs: int, ixr: int, dt_s: float, nx: int):
    """GPU version of _shot_to_rec_time."""
    import cupy as cp

    del ixr
    if nx < travel_time.shape[1]:
        base = cp.round(travel_time[:, :ixs] / dt_s).astype(cp.int32) + 1
        return base + base
    base = cp.round(travel_time / dt_s).astype(cp.int32) + 1
    return base + base


def _assemble_kir_profile_gpu(
    shot_results,
    nx_cal: int,
    length_m: float,
    depth: float,
    dx: float,
):
    """GPU version of _assemble_kir_profile."""
    import cupy as cp

    nx = int(math.ceil(length_m / dx))
    nz = int(math.ceil(depth / dx))
    med = nx_cal + 1000
    ns = max(nx, 1)
    move = max(1, round(nx / ns))

    image = cp.zeros((nx + 2 * med, nz), dtype=cp.float64)

    for s, shot in enumerate(shot_results):
        loaded_slice = shot.T
        row_start = s * move
        row_end = min(row_start + nx_cal, image.shape[0])
        col_end = min(loaded_slice.shape[1], image.shape[1])
        if row_end > row_start and col_end > 0:
            image[row_start:row_end, :col_end] += loaded_slice[
                : row_end - row_start, :col_end
            ]

    x0 = int(nx_cal / 2)
    kir_profile = image[x0 : x0 + nx, :].T
    return kir_profile


def _postprocess_kir_profile_gpu(kir_profile, weight: float):
    """GPU version of _postprocess_kir_profile."""
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage

    data_min = float(cp.min(kir_profile).get())
    data_max = float(cp.max(kir_profile).get())
    denom = max(data_max - data_min, 1e-12)
    normalized = 2.0 * ((kir_profile - data_min) / denom) - 1.0

    normalized = cupy_ndimage.laplace(normalized)

    if weight > 0:
        normalized = _denoise_tv_bregman_gpu(
            normalized, weight, max_iter=1000, eps=1e-6
        )

    return normalized


def _postprocess_kir_profile_gpu_fast(kir_profile, weight: float):
    """Optimized GPU post-processing with faster TV-Bregman convergence.

    Uses reduced iterations (100 vs 1000) and relaxed tolerance for 5-10x speedup
    while maintaining acceptable image quality for GPR visualization.
    """
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage

    # Min-max normalization
    data_min = float(cp.min(kir_profile).get())
    data_max = float(cp.max(kir_profile).get())
    denom = max(data_max - data_min, 1e-12)
    normalized = 2.0 * ((kir_profile - data_min) / denom) - 1.0

    # Laplace filter
    normalized = cupy_ndimage.laplace(normalized)

    # Fast TV-Bregman: 100 iterations instead of 1000
    if weight > 0:
        normalized = _denoise_tv_bregman_gpu_fast(
            normalized, weight, max_iter=100, eps=1e-4
        )

    return normalized


def _denoise_tv_bregman_gpu_fast(
    image, lamda: float, max_iter: int = 100, eps: float = 1.0e-4
):
    """Fast GPU TV-Bregman with reduced precision requirements.

    Optimized for visualization speed rather than full numerical precision.
    """
    import cupy as cp

    lamda = float(lamda)
    if lamda <= 0:
        return cp.array(image, copy=True)

    original = cp.asarray(image, dtype=cp.float64)
    if not cp.isfinite(original).all():
        original = cp.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)

    scale = max(float(cp.max(cp.abs(original)).get()), 1.0)
    normalized = original / scale

    u = cp.array(normalized, copy=True)
    dx = cp.zeros_like(u)
    dy = cp.zeros_like(u)
    bx = cp.zeros_like(u)
    by = cp.zeros_like(u)

    for iter_num in range(max_iter):
        u_old = u.copy()

        # Update u
        ux = cp.roll(u, -1, axis=1) - u
        uy = cp.roll(u, -1, axis=0) - u

        dxx = dx - bx
        dyy = dy - by

        numerator = normalized + lamda * (
            cp.roll(dxx, 1, axis=1) - dxx + cp.roll(dyy, 1, axis=0) - dyy
        )
        denominator = 1.0 + 4.0 * lamda
        u = numerator / denominator

        # Update dual variables
        ux = cp.roll(u, -1, axis=1) - u
        uy = cp.roll(u, -1, axis=0) - u

        tx = ux + bx
        ty = uy + by

        norm = cp.sqrt(tx**2 + ty**2)
        factor = 1.0 / (1.0 + norm / lamda)

        dx = tx * factor
        dy = ty * factor

        bx = bx + ux - dx
        by = by + uy - dy

        # Check convergence every 10 iterations (faster than every iteration)
        if iter_num % 10 == 0:
            diff = cp.max(cp.abs(u - u_old)).get()
            if diff < eps:
                break

    return u * scale


def _denoise_tv_bregman_gpu(
    image, lamda: float, max_iter: int = 100, eps: float = 1.0e-4
):
    """GPU version of _denoise_tv_bregman."""
    import cupy as cp

    lamda = float(lamda)
    if lamda <= 0:
        return cp.array(image, copy=True)

    original = cp.asarray(image, dtype=cp.float64)
    if not cp.isfinite(original).all():
        original = cp.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)

    scale = max(float(cp.max(cp.abs(original)).get()), 1.0)
    normalized = original / scale

    u = cp.array(normalized, copy=True)
    dx = cp.zeros_like(u)
    dy = cp.zeros_like(u)
    bx = cp.zeros_like(u)
    by = cp.zeros_like(u)

    for _ in range(max_iter):
        u_old = u.copy()

        # Update u
        ux = cp.roll(u, -1, axis=1) - u
        uy = cp.roll(u, -1, axis=0) - u

        dxx = dx - bx
        dyy = dy - by

        numerator = normalized + lamda * (
            cp.roll(dxx, 1, axis=1) - dxx + cp.roll(dyy, 1, axis=0) - dyy
        )
        denominator = 1.0 + 4.0 * lamda
        u = numerator / denominator

        # Update dual variables
        ux = cp.roll(u, -1, axis=1) - u
        uy = cp.roll(u, -1, axis=0) - u

        tx = ux + bx
        ty = uy + by

        norm = cp.sqrt(tx**2 + ty**2)
        factor = 1.0 / (1.0 + norm / lamda)

        dx = tx * factor
        dy = ty * factor

        bx = bx + ux - dx
        by = by + uy - dy

        diff = cp.max(cp.abs(u - u_old)).get()
        if diff < eps:
            break

    return u * scale


def _apply_height_correction_time_gpu(data, flight_height_m, dt_s, c_air_m_per_ns=0.30):
    """GPU version of _apply_height_correction_time."""
    import cupy as cp

    n_t, n_x = data.shape
    flight = _resample_trace_vector_gpu(flight_height_m, n_x)
    t_air_ns = 2.0 * cp.maximum(flight, 0.0) / float(c_air_m_per_ns)
    n_mute = cp.ceil(t_air_ns * 1.0e-9 / max(float(dt_s), 1e-12)).astype(cp.int32)
    n_mute = cp.clip(n_mute, 0, n_t)
    corrected = _shift_up_with_zero_pad_gpu(data, n_mute)
    return corrected, n_mute


def _apply_height_correction_depth_gpu(
    migrated, flight_height_m, dz, velocity_m_per_ns
):
    """GPU version of _apply_height_correction_depth."""
    import cupy as cp

    n_z, n_x = migrated.shape
    flight = _resample_trace_vector_gpu(flight_height_m, n_x)
    scaled_flight = flight * float(velocity_m_per_ns) / 0.30
    n_mute = cp.ceil(cp.maximum(scaled_flight, 0.0) / max(float(dz), 1e-12)).astype(
        cp.int32
    )
    n_mute = cp.clip(n_mute, 0, n_z)
    corrected = _shift_up_with_zero_pad_gpu(migrated, n_mute)
    return corrected, n_mute


def _apply_topography_correction_depth_gpu(data, ground_elevation_m, dz):
    """GPU version of _apply_topography_correction_depth."""
    import cupy as cp

    rows, cols = data.shape
    topo = _resample_trace_vector_gpu(ground_elevation_m, cols)
    topo_index = cp.round(topo / max(float(dz), 1e-12)).astype(cp.int32)
    topo_index = int(cp.max(topo_index).get()) - topo_index
    pad_rows = (
        int(cp.max(topo_index).get() - cp.min(topo_index).get())
        if topo_index.size
        else 0
    )

    corrected = cp.zeros((rows + pad_rows, cols), dtype=cp.float64)
    for i in range(cols):
        start = int(topo_index[i].get())
        if start >= corrected.shape[0] - 1:
            continue
        end = min(start + 1 + rows, corrected.shape[0])
        seg_len = end - (start + 1)
        if seg_len > 0:
            corrected[start + 1 : end, i] = data[:seg_len, i]
    return corrected, pad_rows, topo_index


def _resample_trace_vector_gpu(values, target_size: int):
    """GPU version of _resample_trace_vector."""
    import cupy as cp

    target_size = max(1, int(target_size))
    vector = cp.asarray(values, dtype=cp.float64).reshape(-1)
    if vector.size == target_size:
        return cp.array(vector, copy=True)
    if vector.size == 1:
        return cp.full(target_size, float(vector[0].get()), dtype=cp.float64)

    x_old = cp.arange(vector.size, dtype=cp.float64)
    x_new = cp.linspace(0.0, float(vector.size - 1), target_size)
    return cp.interp(x_new, x_old, vector).astype(cp.float64)


def _shift_up_with_zero_pad_gpu(data, n_mute):
    """GPU version of _shift_up_with_zero_pad."""
    import cupy as cp

    corrected = cp.zeros_like(data)
    n_rows, n_cols = data.shape
    mute_arr = cp.asarray(n_mute, dtype=cp.int32).reshape(-1)
    if mute_arr.size != n_cols:
        mute_arr = _resample_trace_vector_gpu(
            mute_arr.astype(cp.float64), n_cols
        ).astype(cp.int32)
    mute_arr = cp.clip(mute_arr, 0, n_rows)

    for j in range(n_cols):
        mute = int(mute_arr[j].get())
        if mute >= n_rows:
            continue
        corrected[: n_rows - mute, j] = data[mute:, j]
    return corrected


def _resample_to_input_grid_gpu(migrated, target_shape, dz, dx):
    """GPU version of _resample_to_input_grid."""
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage

    src_samples = max(1, int(migrated.shape[0]))
    src_traces = max(1, int(migrated.shape[1]))
    target_samples = max(1, int(target_shape[0]))
    target_traces = max(1, int(target_shape[1]))

    if (src_samples, src_traces) == (target_samples, target_traces):
        resized = cp.array(migrated, copy=True)
    else:
        resized = cupy_ndimage.zoom(
            migrated,
            (target_samples / src_samples, target_traces / src_traces),
            order=1,
        )

    resized = cp.nan_to_num(resized, nan=0.0, posinf=0.0, neginf=0.0)
    output_dz = float(dz) * src_samples / target_samples
    output_dx = float(dx) * src_traces / target_traces
    return resized, output_dz, output_dx
