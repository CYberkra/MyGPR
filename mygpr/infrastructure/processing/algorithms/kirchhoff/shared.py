"""Shared numerical kernels for native Kirchhoff migration.

This module is UI-independent and contains the CPU geometry, travel-time,
correction, post-processing, and metadata helpers used by both CPU and GPU
backends.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

def _to_float64_2d(data) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Kirchhoff migration expects a 2D array")
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _require_trace_vector(
    trace_metadata: dict[str, np.ndarray],
    key: str,
    target_traces: int,
    error_message: str,
) -> np.ndarray:
    """Fetch and resample per-trace airborne metadata for the current data width."""
    values = trace_metadata.get(key)
    if values is None:
        raise ValueError(error_message)
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError(error_message)
    if not np.isfinite(vector).all():
        finite = np.isfinite(vector)
        fill = float(np.mean(vector[finite])) if finite.any() else 0.0
        vector = np.nan_to_num(vector, nan=fill, posinf=fill, neginf=fill)
    return _resample_trace_vector(vector, target_traces)


def _resample_trace_vector(values: np.ndarray, target_size: int) -> np.ndarray:
    """Linearly resample a per-trace vector to the requested width."""
    target_size = max(1, int(target_size))
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == target_size:
        return np.array(vector, copy=True)
    if vector.size == 1:
        return np.full(target_size, float(vector[0]), dtype=np.float64)
    x_old = np.arange(vector.size, dtype=np.float64)
    x_new = np.linspace(0.0, float(vector.size - 1), target_size)
    return np.interp(x_new, x_old, vector).astype(np.float64)


def _apply_height_correction_time(
    data: np.ndarray,
    flight_height_m: np.ndarray,
    dt_s: float,
    c_air_m_per_ns: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """Mute direct-air travel before migration, matching CaGPR hei_cor=2."""
    n_t, n_x = data.shape
    flight = _resample_trace_vector(flight_height_m, n_x)
    t_air_ns = 2.0 * np.maximum(flight, 0.0) / float(c_air_m_per_ns)
    n_mute = np.ceil(t_air_ns * 1.0e-9 / max(float(dt_s), 1e-12)).astype(int)
    n_mute = np.clip(n_mute, 0, n_t)
    corrected = _shift_up_with_zero_pad(data, n_mute)
    return corrected, n_mute


def _apply_height_correction_depth(
    migrated: np.ndarray,
    flight_height_m: np.ndarray,
    dz: float,
    velocity_m_per_ns: float,
    *,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mute air-path rows in the migrated depth-domain image."""
    n_z, n_x = migrated.shape
    flight = _resample_trace_vector(flight_height_m, n_x)
    scaled_flight = flight * float(velocity_m_per_ns) / 0.30
    n_mute = np.ceil(np.maximum(scaled_flight, 0.0) / max(float(dz), 1e-12)).astype(int)
    n_mute = np.clip(n_mute, 0, n_z)
    corrected = _shift_up_with_pad_value(migrated, n_mute, fill_value=fill_value)
    return corrected, n_mute


def _apply_topography_correction_depth(
    migrated: np.ndarray,
    ground_elevation_m: np.ndarray,
    dz: float,
    *,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Shift each trace by relative ground elevation, matching CaGPR post-stack flow."""
    rows, cols = migrated.shape
    topo = _resample_trace_vector(ground_elevation_m, cols)
    topo_index = np.round(topo / max(float(dz), 1e-12)).astype(int)
    topo_index = int(np.max(topo_index)) - topo_index
    pad_rows = int(np.max(topo_index) - np.min(topo_index)) if topo_index.size else 0
    corrected = np.full((rows + pad_rows, cols), fill_value, dtype=np.float64)
    for i in range(cols):
        start = int(topo_index[i])
        if start >= corrected.shape[0] - 1:
            continue
        end = min(start + 1 + rows, corrected.shape[0])
        seg_len = end - (start + 1)
        if seg_len > 0:
            corrected[start + 1 : end, i] = migrated[:seg_len, i]
    return corrected, pad_rows, topo_index.astype(np.int32)


def _shift_up_with_zero_pad(data: np.ndarray, n_mute: np.ndarray) -> np.ndarray:
    """Drop leading rows trace-by-trace and pad zeros at the bottom."""
    return _shift_up_with_pad_value(data, n_mute, fill_value=0.0)


def _shift_up_with_pad_value(
    data: np.ndarray, n_mute: np.ndarray, *, fill_value: float
) -> np.ndarray:
    """Drop leading rows trace-by-trace and pad a configurable value."""
    corrected = np.full_like(data, fill_value)
    n_rows, n_cols = data.shape
    mute_arr = np.asarray(n_mute, dtype=np.int32).reshape(-1)
    if mute_arr.size != n_cols:
        mute_arr = _resample_trace_vector(mute_arr.astype(np.float64), n_cols).astype(
            np.int32
        )
    mute_arr = np.clip(mute_arr, 0, n_rows)
    for j in range(n_cols):
        mute = int(mute_arr[j])
        if mute >= n_rows:
            continue
        corrected[: n_rows - mute, j] = data[mute:, j]
    return corrected


def _resample_to_input_grid(
    migrated: np.ndarray,
    target_shape: tuple[int, int],
    dz: float,
    dx: float,
) -> tuple[np.ndarray, float, float]:
    """Resize the migrated image back to the current input grid."""
    src_samples = max(1, int(migrated.shape[0]))
    src_traces = max(1, int(migrated.shape[1]))
    target_samples = max(1, int(target_shape[0]))
    target_traces = max(1, int(target_shape[1]))

    if (src_samples, src_traces) == (target_samples, target_traces):
        resized = np.array(migrated, copy=True)
    else:
        resized = ndimage.zoom(
            migrated,
            (target_samples / src_samples, target_traces / src_traces),
            order=1,
        )

    resized = np.nan_to_num(resized, nan=0.0, posinf=0.0, neginf=0.0)
    output_dz = float(dz) * src_samples / target_samples
    output_dx = float(dx) * src_traces / target_traces
    return np.asarray(resized, dtype=np.float64), output_dz, output_dx


def _setup_grid(length_m, depth, freq, time_window_ns, num_cal):
    """Compute imaging grid parameters matching CaGPR logic."""
    c = 3.0e8
    factor = 60
    dx = c / (factor * float(freq))
    dz = dx
    dt_s = dx / (2.0 * c)
    nt_matrix = int(math.ceil(float(time_window_ns) / (dt_s * 1.0e9)))
    nz_matrix = int(math.ceil(float(depth) / dz))
    nx_matrix = 2 * nz_matrix
    nx = int(math.ceil(float(length_m) / dx))
    divisor = max(1, int(num_cal))

    while nx % divisor != 0:
        factor += 1
        dx = c / (factor * float(freq))
        dz = dx
        dt_s = dx / (2.0 * c)
        nt_matrix = int(math.ceil(float(time_window_ns) / (dt_s * 1.0e9)))
        nz_matrix = int(math.ceil(float(depth) / dz))
        nx_matrix = 2 * nz_matrix
        nx = int(math.ceil(float(length_m) / dx))

    nt_matrix = max(1, nt_matrix)
    nz_matrix = max(1, nz_matrix)
    nx_matrix = max(1, nx_matrix)
    return factor, nx_matrix, nt_matrix, nz_matrix, dx, dt_s


def _compute_travel_time(
    velocity_model: np.ndarray,
    nz: int,
    nx: int,
    dx: float,
    dt_s: float,
    *,
    cancel_checker=None,
):
    """Compute travel-time table using 2D time-to-depth conversion."""
    # CaGPR uses velocity in m/ns for UI parameters, but Time2d expects
    # slowness derived from m/s.
    slowness = 1.0 / (velocity_model.T * 1.0e9)
    fs_z = 6
    fs_x = 6
    fs = _fstar(fs_z, fs_x)
    iz = np.arange(fs_z, nz + fs_z)
    ix = np.arange(fs_x, nx + fs_x)
    fs_z2 = 2 * fs_z - 1
    fs_x2 = 2 * fs_x - 1
    S = np.ones((nz + fs_z2 - 1, nx + fs_x2 - 1))
    S[np.ix_(iz - 1, ix - 1)] = slowness
    S[nz + fs_z - 1, ix - 1] = 2 * S[nz + fs_z - 2, ix - 1] - S[nz + fs_z - 3, ix - 1]
    S[np.ix_(iz - 1, [nx + fs_x - 1])] = (
        2 * S[np.ix_(iz - 1, [nx + fs_x - 2])] - S[np.ix_(iz - 1, [nx + fs_x - 3])]
    )
    S[nz + fs_z - 1, nx + fs_x - 1] = (
        2 * S[nz + fs_z - 2, nx + fs_x - 2] - S[nz + fs_z - 3, nx + fs_x - 3]
    )
    src_x = max(1, int(nx / 2))
    travel_time = _time2d(
        S, [1, src_x], dx, nz, nx, fs_z, fs_x, fs, cancel_checker=cancel_checker
    )
    return travel_time


def _run_kirchhoff_stack(
    travel_time: np.ndarray,
    data_resized: np.ndarray,
    freq: float,
    dt_s: float,
    nz_matrix: int,
    nx_matrix: int,
    *,
    cancel_checker=None,
):
    """Run Kirchhoff stack accumulation across all shots."""
    nt = data_resized.shape[0]
    signal, _ = _ricker(freq, nt, dt_s)
    kstart = int(np.argmax(signal))
    data2 = data_resized[kstart:, :]
    padded = np.zeros((data2.shape[0] + 3 * nt, data2.shape[1]), dtype=np.float64)
    padded[: data2.shape[0], :] = data2

    x_shot_grid = np.arange(1, data_resized.shape[1] + 1)
    shot_results = []
    total_shots = len(x_shot_grid)
    for shot_idx, xs in enumerate(x_shot_grid, start=1):
        if cancel_checker is not None and bool(cancel_checker()):
            raise Exception("用户已取消（Kirchhoff迁移）")
        shot_results.append(
            _migrate(
                travel_time,
                x_shot_grid[shot_idx - 1],
                x_shot_grid[shot_idx - 1],
                padded,
                dt_s,
                nz_matrix,
                nx_matrix,
                xs,
            )
        )
        if cancel_checker is not None and shot_idx % 8 == 0 and bool(cancel_checker()):
            raise Exception("用户已取消（Kirchhoff炮点累加）")
        _ = total_shots
    return shot_results, kstart


def _assemble_kir_profile(
    shot_results: list[np.ndarray],
    nx_cal: int,
    length_m: float,
    depth: float,
    dx: float,
) -> np.ndarray:
    nx = int(math.ceil(length_m / dx))
    nz = int(math.ceil(depth / dx))
    med = nx_cal + 1000
    ns = max(nx, 1)
    move = max(1, round(nx / ns))
    image = np.zeros((nx + 2 * med, nz), dtype=np.float64)
    for s, shot in enumerate(shot_results):
        loaded_slice = np.asarray(shot, dtype=np.float64).T
        row_start = s * move
        row_end = min(row_start + nx_cal, image.shape[0])
        col_end = min(loaded_slice.shape[1], image.shape[1])
        if row_end > row_start and col_end > 0:
            image[row_start:row_end, :col_end] += loaded_slice[
                : row_end - row_start, :col_end
            ]

    x0 = int(nx_cal / 2)
    kir_profile = image[x0 : x0 + nx, :].T
    return np.asarray(kir_profile, dtype=np.float64)


def _postprocess_kir_profile(kir_profile: np.ndarray, weight: float) -> np.ndarray:
    data_min = float(np.min(kir_profile))
    data_max = float(np.max(kir_profile))
    denom = max(data_max - data_min, 1e-12)
    normalized = 2.0 * ((kir_profile - data_min) / denom) - 1.0
    normalized = _higher_order_laplace_filter(normalized, order=1)
    if weight > 0:
        normalized = _denoise_tv_bregman(normalized, weight, max_iter=1000, eps=1e-6)
    return np.asarray(normalized, dtype=np.float64)


def _higher_order_laplace_filter(data: np.ndarray, order: int = 1) -> np.ndarray:
    del order
    filtered = np.array(data, copy=True)
    return ndimage.laplace(filtered)


def _denoise_tv_bregman(
    image: np.ndarray, lamda: float, max_iter: int = 100, eps: float = 1.0e-4
) -> np.ndarray:
    lamda = float(lamda)
    if lamda <= 0:
        return np.array(image, copy=True)

    original = np.asarray(image, dtype=np.float64)
    if not np.isfinite(original).all():
        original = np.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)

    orig_min = float(np.min(original))
    orig_max = float(np.max(original))

    image_3d = _atleast_3d(original)
    rows, cols, _dims = image_3d.shape
    image_padded = np.pad(image_3d, [(1, 1), (1, 1), (0, 0)], mode="symmetric")

    dx = np.zeros_like(image_padded)
    dy = np.zeros_like(image_padded)
    bx = np.zeros_like(image_padded)
    by = np.zeros_like(image_padded)

    mu = 1.0
    norm_factor = 1.0 + 4.0 * lamda
    lamda2 = lamda

    for _ in range(max_iter):
        u_prev = image_padded[1:-1, 1:-1, :]

        ux = image_padded[1 : rows + 1, 2 : cols + 2, :] - u_prev
        uy = image_padded[2 : rows + 2, 1 : cols + 1, :] - u_prev

        u_new = (
            lamda
            * (
                image_padded[2 : rows + 2, 1 : cols + 1, :]
                + image_padded[0:rows, 1 : cols + 1, :]
                + image_padded[1 : rows + 1, 2 : cols + 2, :]
                + image_padded[1 : rows + 1, 0:cols, :]
                + image_3d
                + dx[1 : rows + 1, 0:cols, :]
                - dx[1 : rows + 1, 1 : cols + 1, :]
                + dy[0:rows, 1 : cols + 1, :]
                - dy[1 : rows + 1, 1 : cols + 1, :]
                - bx[1 : rows + 1, 0:cols, :]
                + bx[1 : rows + 1, 1 : cols + 1, :]
                - by[0:rows, 1 : cols + 1, :]
                + by[1 : rows + 1, 1 : cols + 1, :]
            )
            / norm_factor
        )

        image_padded[1 : rows + 1, 1 : cols + 1, :] = u_new
        image_padded = _fill_boundary(image_padded)

        residual = np.linalg.norm(
            u_new.ravel() - u_prev.ravel()
        ) + lamda2 * np.linalg.norm(u_prev.ravel())
        if residual < eps:
            break

        tx = ux + bx[1 : rows + 1, 1 : cols + 1, :]
        ty = uy + by[1 : rows + 1, 1 : cols + 1, :]
        s = np.sqrt(tx**2 + ty**2) + 1.0e-6

        dx_new = (mu * s * tx) / (mu * s + 1.0)
        dy_new = (mu * s * ty) / (mu * s + 1.0)

        bx[1 : rows + 1, 1 : cols + 1, :] = (
            bx[1 : rows + 1, 1 : cols + 1, :] + ux - dx_new
        )
        by[1 : rows + 1, 1 : cols + 1, :] = (
            by[1 : rows + 1, 1 : cols + 1, :] + uy - dy_new
        )

        dx[1 : rows + 1, 1 : cols + 1, :] = dx_new
        dy[1 : rows + 1, 1 : cols + 1, :] = dy_new

    padded_min = float(np.min(image_padded))
    padded_max = float(np.max(image_padded))
    denom = max(padded_max - padded_min, 1.0e-12)
    out_normalized = (image_padded[1:-1, 1:-1, :] - padded_min) / denom
    out = out_normalized * (orig_max - orig_min) + orig_min
    return out[:, :, 0]


def _atleast_3d(image: np.ndarray) -> np.ndarray:
    if image.ndim < 3:
        return image[..., np.newaxis]
    return image


def _fill_boundary(image: np.ndarray) -> np.ndarray:
    image[0, 1:-1, :] = image[1, 1:-1, :]
    image[-1, 1:-1, :] = image[-2, 1:-1, :]
    image[1:-1, 0, :] = image[1:-1, 1, :]
    image[1:-1, -1, :] = image[1:-1, -2, :]
    image[0, 0, :] = image[1, 1, :]
    image[0, -1, :] = image[1, -2, :]
    image[-1, 0, :] = image[-2, 1, :]
    image[-1, -1, :] = image[-2, -2, :]
    return image


def _migrate(
    travel_time: np.ndarray,
    x_rec_grid,
    x_shot_and_rec_grid,
    shot: np.ndarray,
    dt_s: float,
    nz: int,
    nx: int,
    ixs: int,
) -> np.ndarray:
    image = np.zeros((nz, nx), dtype=np.float64)
    x_rec_grid = np.atleast_1d(x_rec_grid)
    for ixr in range(len(x_rec_grid)):
        xr = x_rec_grid[ixr]
        matches = np.where(np.atleast_1d(x_shot_and_rec_grid) == np.atleast_1d(xr))[0]
        if len(matches) == 0:
            continue
        idx_x_rec = matches[0]
        time_indices = _shot_to_rec_time(travel_time, ixs - 1, idx_x_rec, dt_s, nx)
        max_it = int(np.max(time_indices))
        if max_it >= shot.shape[0]:
            pad_rows = max_it - shot.shape[0] + 1
            padded_shot = np.pad(shot, ((0, pad_rows), (0, 0)), mode="constant")
        else:
            padded_shot = shot
        migrated_values = padded_shot[time_indices, ixs - 1]
        image = np.reshape(migrated_values, (nz, nx))
    return image


def _shot_to_rec_time(
    travel_time: np.ndarray, ixs: int, ixr: int, dt_s: float, nx: int
):
    del ixr
    if nx < travel_time.shape[1]:
        base = np.round(travel_time[:, :ixs] / dt_s).astype(int) + 1
        return base + base
    base = np.round(travel_time / dt_s).astype(int) + 1
    return base + base


def _fstar(sz: int, sx: int) -> np.ndarray:
    sz2 = 2 * sz - 1
    sx2 = 2 * sx - 1
    sz1 = sz2 - 1
    sx1 = sx2 - 1
    nrow = sz2 * sx2
    ncol = sz1 * sx1
    a = np.zeros((nrow, ncol), dtype=np.float64)
    nray = 0
    rayxz = np.zeros((2, 1000), dtype=np.float64)
    temp = np.zeros((2, 1000), dtype=np.float64)

    for kz in range(1, sz2 + 1):
        z0 = kz - 1
        for kx in range(1, sx2 + 1):
            x0 = kx - 1
            nray += 1
            dxx = sx - kx
            dzz = sz - kz

            if dxx == 0 or dzz == 0:
                if dxx == 0 and dzz != 0:
                    np_val = 0
                    if dzz > 0:
                        for kk in range(kz, sz + 1):
                            np_val += 1
                            temp[0, np_val - 1] = x0
                            temp[1, np_val - 1] = kk - 1
                    else:
                        for kk in range(kz, sz - 1, -1):
                            np_val += 1
                            temp[0, np_val - 1] = x0
                            temp[1, np_val - 1] = kk - 1
                else:
                    np_val = 0
                    if dxx > 0:
                        for kk in range(kx, sx + 1):
                            np_val += 1
                            temp[0, np_val - 1] = kk - 1
                            temp[1, np_val - 1] = z0
                    else:
                        for kk in range(kx, sx - 1, -1):
                            np_val += 1
                            temp[0, np_val - 1] = kk - 1
                            temp[1, np_val - 1] = z0
            else:
                slope = dzz / dxx
                r_slope = 1.0 / slope
                seg = 1
                rayxz[:, 0] = [x0, z0]

                if slope > 0:
                    if dxx > 0:
                        x = x0
                        for _ix in range(kx + 1, sx + 1):
                            seg += 1
                            x += 1
                            z = slope * (x - x0)
                            rayxz[:, seg - 1] = [x, z + z0]

                        z = z0
                        for _iz in range(kz + 1, sz + 1):
                            seg += 1
                            z += 1
                            x = (z - z0) * r_slope
                            rayxz[:, seg - 1] = [x + x0, z]
                    else:
                        x = x0
                        for _ix in range(kx - 1, sx - 1, -1):
                            seg += 1
                            x -= 1
                            z = slope * (x - x0)
                            rayxz[:, seg - 1] = [x, z + z0]

                        z = z0
                        for _iz in range(kz - 1, sz - 1, -1):
                            seg += 1
                            z -= 1
                            x = (z - z0) * r_slope
                            rayxz[:, seg - 1] = [x + x0, z]
                else:
                    if dxx < 0:
                        x = x0
                        for _ix in range(kx - 1, sx - 1, -1):
                            seg += 1
                            x -= 1
                            z = slope * (x - x0)
                            rayxz[:, seg - 1] = [x, z + z0]

                        z = z0
                        for _iz in range(kz + 1, sz + 1):
                            seg += 1
                            z += 1
                            x = (z - z0) * r_slope
                            rayxz[:, seg - 1] = [x + x0, z]
                    else:
                        x = x0
                        for _ix in range(kx + 1, sx + 1):
                            seg += 1
                            x += 1
                            z = slope * (x - x0)
                            rayxz[:, seg - 1] = [x, z + z0]

                        z = z0
                        for _iz in range(kz - 1, sz - 1, -1):
                            seg += 1
                            z -= 1
                            x = (z - z0) * r_slope
                            rayxz[:, seg - 1] = [x + x0, z]

                sorted_indices = np.argsort(rayxz[0, :seg])
                rayxz[:, :seg] = rayxz[:, sorted_indices]
                temp[:, 0] = rayxz[:, 0]
                np_val = 1

                for k in range(1, seg):
                    dist = np.linalg.norm(rayxz[:, k] - rayxz[:, k - 1])
                    if dist > 1.0e-5:
                        np_val += 1
                        temp[:, np_val - 1] = rayxz[:, k]

            for k in range(1, np_val):
                dist = np.linalg.norm(temp[:, k] - temp[:, k - 1])
                aa = 0.5 * (temp[:, k] + temp[:, k - 1])
                indx = int(np.floor(aa[0]))
                indz = int(np.floor(aa[1]))
                ind = indz * sx1 + indx
                a[nray - 1, ind] = dist

    return a


def _time2d(
    slowness: np.ndarray,
    shot,
    dx: float,
    nz: int,
    nx: int,
    fs_z: int,
    fs_x: int,
    fs: np.ndarray,
    *,
    cancel_checker=None,
) -> np.ndarray:
    t0 = 1.0e8
    fs_z2 = 2 * fs_z - 1
    fs_x2 = 2 * fs_x - 1
    zs = shot[0] + fs_z - 1
    xs = shot[1] + fs_x - 1
    mx_v = float(np.max(slowness))
    t = np.ones((nz + fs_z2, nx + fs_x2), dtype=np.float64) * t0
    marker = np.copy(t)

    marker[fs_z - 1 : nz + fs_z, fs_x - 1 : nx + fs_x] = 0
    iz = np.arange(fs_z, nz + fs_z)
    ix = np.arange(fs_x, nx + fs_x)
    t[zs - 1, xs] = 0
    marker[zs - 1, xs] = t0

    z1 = np.arange(-fs_z + 1, fs_z)
    z2 = np.arange(-fs_z + 1, fs_z - 1)
    z3 = z1 + zs
    x1 = np.arange(-fs_x + 1, fs_x)
    x2 = np.arange(-fs_x + 1, fs_x - 1)
    x3 = x1 + xs
    local_s = slowness[np.ix_(z2 + zs - 1, x2 + xs)]
    local_t = t[np.ix_(z3 - 1, x3)]
    result = fs @ local_s.flatten()[:, np.newaxis]
    result += t[zs - 1, xs]
    t[np.ix_(z3 - 1, x3)] = np.minimum(result.reshape(fs_z2, fs_x2), local_t)
    mx_t = float(np.max(t[zs - 2 : zs + 1, xs - 1 : xs + 2]))

    iteration = 0
    while True:
        iteration += 1
        if (
            cancel_checker is not None
            and iteration % 32 == 0
            and bool(cancel_checker())
        ):
            raise Exception("用户已取消（Kirchhoff走时表计算）")

        indx = t + marker <= mx_t + mx_v
        if not np.any(indx):
            indx = marker == 0

        idx, idz = np.where(indx.T)
        marker[indx] = t0
        for i in range(len(idz)):
            z = idz[i]
            x = idx[i]
            mx_t = max(mx_t, float(t[z, x]))
            local_s = slowness[z + z2[:, np.newaxis], x + x2]
            z3 = z + z1
            x3 = x + x1
            local_t = t[np.ix_(z3, x3)]
            result = fs @ local_s.flatten()[:, np.newaxis]
            result += t[z, x]
            t[np.ix_(z3, x3)] = np.minimum(result.reshape(fs_z2, fs_x2), local_t)
        if np.all(marker[iz[:, np.newaxis] - 1, ix - 1]):
            break
        mx_t = float(np.max(t[idz, idx]))

    return t[iz[:, np.newaxis] - 1, ix - 1] * dx


def _ricker(freq: float, n: int, dt_s: float, t0=None):
    if t0 is None:
        t0 = 1.0 / freq
    total_t = dt_s * (n - 1)
    t = np.arange(0.0, total_t + dt_s, dt_s)
    tau = t - t0
    s = (1.0 - 2.0 * (tau**2) * (freq**2) * (np.pi**2)) * np.exp(
        -(tau**2) * (np.pi**2) * (freq**2)
    )
    return s, t


def _smooth2a(matrix_in: np.ndarray, nr: int, nc: int | None = None) -> np.ndarray:
    if nc is None:
        nc = nr
    nr = max(0, int(nr))
    nc = max(0, int(nc))
    if nr == 0 and nc == 0:
        return np.array(matrix_in, copy=True)
    arr = np.asarray(matrix_in, dtype=np.float64)
    valid = np.isfinite(arr).astype(np.float64)
    work = np.nan_to_num(arr, nan=0.0)
    size = (2 * nr + 1, 2 * nc + 1)
    num = ndimage.uniform_filter(work, size=size, mode="nearest") * (size[0] * size[1])
    den = ndimage.uniform_filter(valid, size=size, mode="nearest") * (size[0] * size[1])
    out = np.divide(
        num, np.maximum(den, 1.0e-12), out=np.zeros_like(num), where=den > 0
    )
    return out


def _resolve_elevation_axis_top_m(
    header_info: dict[str, object] | None,
    trace_metadata: dict[str, np.ndarray],
    key: str,
) -> float | None:
    """Prefer original airborne elevation maxima for elevation-axis labeling."""
    if header_info and header_info.get("ground_elevation_max_m") is not None:
        return float(header_info.get("ground_elevation_max_m"))
    values = trace_metadata.get(key)
    if values is None:
        return None
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = vector[np.isfinite(vector)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def _build_kirchhoff_header_updates(
    header_info: dict[str, object] | None,
    output_shape: tuple[int, int],
    depth_step_m: float,
    trace_interval_m: float,
    *,
    topo_cor: int,
    elevation_axis_top_m: float | None,
) -> dict[str, object]:
    """Update depth-axis fields while preserving airborne summary metadata."""
    samples = max(1, int(output_shape[0]))
    updates: dict[str, object] = {
        "a_scan_length": samples,
        "num_traces": int(output_shape[1]),
        "trace_interval_m": float(trace_interval_m),
        "total_time_ns": 0.0,
        "is_depth": topo_cor <= 0,
        "is_elevation": topo_cor > 0,
        "depth_step_m": float(depth_step_m),
        "depth_max_m": float(depth_step_m * max(samples - 1, 0)),
        "display_hint": "signed_migration",
        "display_center_zero": True,
        "display_show_cbar": True,
        "display_percentile_abs_high": 99.5,
        "display_cagpr_contrast": 1.0,
        "display_fixed_unit_range": True,
        "display_skip_preprocess": True,
        "display_title": "Kirchhoff Migration profile",
        "display_xlabel": "水平距离 (m)",
        "display_colorbar_label": "信号幅度",
    }
    updates["display_ylabel"] = "高程 (m)" if topo_cor > 0 else "深度 (m)"
    if topo_cor > 0 and elevation_axis_top_m is not None:
        updates["elevation_top_m"] = float(elevation_axis_top_m)
        updates["elevation_bottom_m"] = float(
            elevation_axis_top_m - depth_step_m * max(samples - 1, 0)
        )
    else:
        updates["elevation_top_m"] = None
        updates["elevation_bottom_m"] = None
    if header_info and "has_airborne_metadata" in header_info:
        updates["has_airborne_metadata"] = bool(
            header_info.get("has_airborne_metadata")
        )
    return updates


def _build_kirchhoff_display_header_updates(
    header_info: dict[str, object] | None,
    output_shape: tuple[int, int],
    depth_step_m: float,
    trace_interval_m: float,
    *,
    topo_cor: int,
    elevation_axis_top_m: float | None,
) -> dict[str, object]:
    """Build GUI-only display metadata for Kirchhoff images."""
    updates = _build_kirchhoff_header_updates(
        header_info,
        output_shape,
        depth_step_m,
        trace_interval_m,
        topo_cor=topo_cor,
        elevation_axis_top_m=elevation_axis_top_m,
    )
    updates.update(
        {
            "display_title": "Kirchhoff Migration profile",
            "display_xlabel": "水平距离 (m)",
            "display_colorbar_label": "信号幅度",
            "display_skip_preprocess": True,
            "display_bad_color": "white",
        }
    )
    return updates
