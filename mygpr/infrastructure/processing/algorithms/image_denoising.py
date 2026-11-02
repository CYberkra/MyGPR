"""Full-resolution Gaussian and uniform-patch NLM filters for signed B-scans.

Both ndarray and file-backed execution use the same halo-tile kernels. NLM
uses mean squared patch distance and exp(-distance/h**2), including the self
patch; it does not quantize, normalize, take an envelope, or clip amplitudes.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import correlate1d, gaussian_filter


def _number(params: dict, key: str, default: float, low: float, high: float) -> float:
    raw = params.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} 必须为有限数值") from exc
    if isinstance(raw, (bool, np.bool_)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} 必须在 {low}～{high} 之间")
    return value


def _integer(params: dict, key: str, default: int, low: int, high: int, *, odd=False) -> int:
    value = _number(params, key, default, low, high)
    if value != int(value) or (odd and int(value) % 2 != 1):
        raise ValueError(f"{key} 必须为{'奇数' if odd else ''}整数")
    return int(value)


def _check(params: dict, context=None) -> None:
    context = context or params.get("_execution_context")
    if context is not None:
        context.raise_if_cancelled()
    checker = params.get("cancel_checker")
    if callable(checker) and checker():
        raise RuntimeError("processing cancelled during image denoising")


def _tiles(shape: tuple, rows: int):
    for r in range(0, shape[0], rows):
        for c in range(0, shape[1], 256):
            yield r, min(r + rows, shape[0]), c, min(c + 256, shape[1])


def _scan(source, rows: int, params: dict, context) -> float:
    """Bounded validation and global RMS, identical for every output tile."""
    squares = 0.0
    for r0, r1, c0, c1 in _tiles(source.shape, rows):
        _check(params, context)
        block = np.asarray(source[r0:r1, c0:c1], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("图像滤波输入含 NaN/Inf，请先检查或修复数据")
        if np.max(np.abs(block)) > np.finfo(np.float32).max:
            raise ValueError("图像滤波输入超出 float32 范围")
        squares += float(np.sum(block * block, dtype=np.float64))
    return math.sqrt(squares / (source.shape[0] * source.shape[1]))


def _reflect_indices(start: int, end: int, size: int) -> np.ndarray:
    # Half-sample symmetric reflection: ... b a | a b c d | d c ...
    folded = np.arange(start, end, dtype=np.int64) % (2 * size)
    return np.where(folded < size, folded, 2 * size - 1 - folded)


def _halo(source, r0, r1, c0, c1, hr, hc):
    rr = _reflect_indices(r0 - hr, r1 + hr, source.shape[0])
    cc = _reflect_indices(c0 - hc, c1 + hc, source.shape[1])
    return np.asarray(source[np.ix_(rr, cc)], dtype=np.float64)


def _nlm_tile(block, shape, patch, search, h, params, context):
    height, width = shape
    pr, pc = patch[0] // 2, patch[1] // 2
    sr, sc = search
    target = block[sr:sr+height+2*pr, sc:sc+width+2*pc]
    weighted = np.zeros(shape, dtype=np.float64)
    weights = np.zeros(shape, dtype=np.float64)
    for dr in range(-sr, sr + 1):
        for dc in range(-sc, sc + 1):
            _check(params, context)
            neighbor = block[sr+dr:sr+dr+height+2*pr, sc+dc:sc+dc+width+2*pc]
            difference = target - neighbor
            # Direct finite-support positive sums avoid subtracting cumulative
            # sums: a strong direct wave must not erase later weak distances.
            distance = correlate1d(difference * difference, np.full(patch[0], 1 / patch[0]),
                                   axis=0, mode="constant")
            distance = correlate1d(distance, np.full(patch[1], 1 / patch[1]),
                                   axis=1, mode="constant")
            distance = distance[pr:pr+height, pc:pc+width]
            np.maximum(distance, 0.0, out=distance)
            # Do not square a tiny h: h**2 could underflow and make self=0/0.
            # Overflow of normalized distance means a correctly zero weight.
            with np.errstate(over="ignore", under="ignore"):
                scaled = np.sqrt(distance) / h
                weight = np.exp(-(scaled * scaled))
            weighted += weight * neighbor[pr:pr+height, pc:pc+width]
            weights += weight
    return weighted / weights


def _execute(source, output, params: dict[str, Any], method: str, context=None, rows=128):
    if len(source.shape) != 2 or not source.shape[0] or not source.shape[1]:
        raise ValueError("图像滤波输入必须为非空二维雷达矩阵")
    rows = max(1, min(128, int(rows)))
    meta: dict[str, Any]
    if method == "gaussian_smoothing":
        sigma = (_number(params, "sigma_samples", 0.0, 0, 32),
                 _number(params, "sigma_traces", 1.0, 0, 32))
        halo = tuple(int(4 * value + 0.5) for value in sigma)
        meta = {"sigma_samples": sigma[0], "sigma_traces": sigma[1], "truncate": 4.0}
    else:
        patch = (_integer(params, "patch_samples", 3, 1, 15, odd=True),
                 _integer(params, "patch_traces", 5, 1, 15, odd=True))
        search = (_integer(params, "search_samples", 3, 0, 15),
                  _integer(params, "search_traces", 5, 0, 15))
        strength = _number(params, "h", 0.6, 0, 1e38)
        mode = params.get("h_mode", "relative_rms")
        if mode not in {"relative_rms", "absolute"}:
            raise ValueError("NLM 强度模式必须为 relative_rms 或 absolute")
        halo = tuple(s + p // 2 for s, p in zip(search, patch))
        meta = {"patch_samples": patch[0], "patch_traces": patch[1],
                "search_samples": search[0], "search_traces": search[1],
                "h": strength, "h_mode": mode, "patch_weighting": "uniform",
                "noise_variance_correction": False}
    rms = _scan(source, rows, params, context)
    if method == "non_local_means":
        h = strength * rms if mode == "relative_rms" else strength
        meta.update(global_rms=rms, h_effective=h)
    count = math.ceil(source.shape[0] / rows) * math.ceil(source.shape[1] / 256)
    for index, (r0, r1, c0, c1) in enumerate(_tiles(source.shape, rows), 1):
        _check(params, context)
        if method == "non_local_means" and h == 0:
            result = source[r0:r1, c0:c1]
        else:
            block = _halo(source, r0, r1, c0, c1, *halo)
            if method == "gaussian_smoothing":
                filtered = gaussian_filter(block, sigma, mode="reflect", truncate=4.0)
                result = filtered[halo[0]:halo[0]+r1-r0, halo[1]:halo[1]+c1-c0]
            else:
                result = _nlm_tile(block, (r1-r0, c1-c0), patch, search, h, params, context)
        output[r0:r1, c0:c1] = result
        if context is not None:
            context.report_progress(index, count, f"{method}: {index}/{count}")
    return {"method_id": method, "method": method, "boundary": "reflect",
            "input_shape": list(source.shape), "tile_count": count, **meta}


def method_gaussian(data, params):
    source = np.asarray(data)
    output = np.empty(source.shape, dtype=np.float32)
    return output, _execute(source, output, params, "gaussian_smoothing")


def method_nlm(data, params):
    source = np.asarray(data)
    output = np.empty(source.shape, dtype=np.float32)
    return output, _execute(source, output, params, "non_local_means")


def file_gaussian(source, output, params, context, row_block_size):
    return _execute(source, output, params, "gaussian_smoothing", context, row_block_size)


def file_nlm(source, output, params, context, row_block_size):
    return _execute(source, output, params, "non_local_means", context, row_block_size)
