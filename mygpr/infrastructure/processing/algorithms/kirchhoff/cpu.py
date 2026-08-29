"""CPU backend for native Kirchhoff migration."""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from mygpr.infrastructure.processing.algorithms.kirchhoff.shared import (
    _apply_height_correction_depth,
    _apply_height_correction_time,
    _apply_topography_correction_depth,
    _assemble_kir_profile,
    _build_kirchhoff_display_header_updates,
    _build_kirchhoff_header_updates,
    _compute_travel_time,
    _postprocess_kir_profile,
    _require_trace_vector,
    _resample_to_input_grid,
    _resolve_elevation_axis_top_m,
    _run_kirchhoff_stack,
    _setup_grid,
    _smooth2a,
)

def _run_kirchhoff_cpu(
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
    """Run Kirchhoff migration on CPU."""
    factor, nx_matrix, nt_matrix, nz_matrix, dx, dt_s = _setup_grid(
        length_m, depth, freq, time_window_ns, num_cal
    )
    dz = dx
    resized_traces = max(1, int(math.ceil(length_m / dx)))
    data_resized = ndimage.zoom(
        arr,
        (nt_matrix / arr.shape[0], resized_traces / arr.shape[1]),
        order=1,
    ).astype(np.float64, copy=False)

    signs = np.sign(data_resized)
    data_resized = np.power(np.abs(data_resized), alpha) * signs

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
        data_resized, n_mute_arr = _apply_height_correction_time(
            data_resized, flight_height, dt_s
        )
        correction_info.update(
            {
                "height_mute_applied": True,
                "height_correction_stage": "pre",
                "n_mute_min": int(np.min(n_mute_arr)) if n_mute_arr.size else 0,
                "n_mute_max": int(np.max(n_mute_arr)) if n_mute_arr.size else 0,
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
        data_resized, pad_rows, _ = _apply_topography_correction_depth(
            data_resized, ground_elevation, dz
        )
        correction_info.update(
            {
                "topography_applied": True,
                "topography_stage": "pre",
                "topography_pad_rows": int(pad_rows),
            }
        )

    velocity_model = np.full((nx_matrix, nz_matrix), velocity, dtype=np.float64)
    velocity_model = _smooth2a(velocity_model, nr=10)

    travel_time = _compute_travel_time(
        velocity_model, nz_matrix, nx_matrix, dx, dt_s, cancel_checker=cancel_checker
    )
    shot_results, kstart = _run_kirchhoff_stack(
        travel_time,
        data_resized,
        freq,
        dt_s,
        nz_matrix,
        nx_matrix,
        cancel_checker=cancel_checker,
    )

    kir_profile = _assemble_kir_profile(shot_results, nx_matrix, length_m, depth, dx)
    migrated = _postprocess_kir_profile(kir_profile, weight=weight)
    imaging_shape = (int(migrated.shape[0]), int(migrated.shape[1]))
    display_migrated = np.array(migrated, copy=True)
    output_migrated = np.array(migrated, copy=True)

    if hei_cor == 1:
        flight_height = _require_trace_vector(
            trace_metadata,
            "flight_height_m",
            arr.shape[1],
            "hei_cor=1 需要 flight_height_m 航空元数据",
        )
        display_migrated, n_mute_arr = _apply_height_correction_depth(
            display_migrated, flight_height, dz, velocity, fill_value=np.nan
        )
        output_migrated, _ = _apply_height_correction_depth(
            output_migrated, flight_height, dz, velocity, fill_value=0.0
        )
        correction_info.update(
            {
                "height_mute_applied": True,
                "height_correction_stage": "post",
                "n_mute_min": int(np.min(n_mute_arr)) if n_mute_arr.size else 0,
                "n_mute_max": int(np.max(n_mute_arr)) if n_mute_arr.size else 0,
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
        display_migrated, pad_rows, _ = _apply_topography_correction_depth(
            display_migrated, ground_elevation, dz, fill_value=np.nan
        )
        output_migrated, _, _ = _apply_topography_correction_depth(
            output_migrated, ground_elevation, dz, fill_value=0.0
        )
        correction_info.update(
            {
                "topography_applied": True,
                "topography_stage": "post",
                "topography_pad_rows": int(pad_rows),
            }
        )

    display_data = np.asarray(display_migrated, dtype=np.float32)
    display_header_info_updates = _build_kirchhoff_display_header_updates(
        header_info,
        display_data.shape,
        dz,
        dx,
        topo_cor=topo_cor,
        elevation_axis_top_m=elevation_axis_top_m,
    )

    output, output_dz, output_dx = _resample_to_input_grid(
        output_migrated, arr.shape, dz, dx
    )
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
            "note": "cagpr_kirchhoff_main_chain_resampled",
            **correction_info,
        },
        "header_info_updates": header_info_updates,
        "display_data": display_data,
        "display_header_info_updates": display_header_info_updates,
    }
