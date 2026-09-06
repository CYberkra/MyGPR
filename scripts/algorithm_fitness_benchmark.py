"""MyGPR 算法体检基准 (UAV-GPR 视角).

跑法:
    .venv/Scripts/python.exe scripts/algorithm_fitness_benchmark.py --run-all

- 父进程模式 (--run-all): 聚合全部任务, 输出 output/autoresearch/algorithm_fitness.json
  与 METRIC 行 (fitness_score / broken_count / timeout_count / redundant_pair_count /
  total_runtime_s)。每个方法-场景对在子进程里执行 (--worker), 超时按 timeout 计。
- 单任务模式 (--worker --method X --scene Y --json): stdout 打印单条 JSON。
- 冗余检测: 同族任务输出矩阵保存为 .npy (临时目录), 两两 Pearson 相关 > 0.995 判冗余。
- 所有合成场景用固定种子 (np.random.default_rng), 全程确定性。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "output" / "autoresearch"
WORKER_TIMEOUT_S = 90.0
REDUNDANCY_CORR_THRESHOLD = 0.995
REPO_TAG = "repo"
C = 0.1  # m/ns, 合成场景统一波速

# ---------------------------------------------------------------- 场景构造 ---

def _meta(header_ns, trace_interval_m, extra=None):
    m = {
        "header_info": {
            "a_scan_length": None,
            "num_traces": None,
            "trace_interval_m": trace_interval_m,
            "total_time_ns": header_ns,
        },
        "trace_metadata": {},
    }
    if extra:
        m.update(extra)
    return m


def scene_zero_time() -> tuple[np.ndarray, dict]:
    """零时基准: 直接复用仓库 fixture (ref zero idx 18, 700ns)。"""
    from core.benchmark_registry import generate_benchmark_sample

    data, meta = generate_benchmark_sample("zero_time_reference")
    meta["ref_zero_idx"] = 18
    meta["time_step_ns"] = 700.0 / (data.shape[0] - 1)
    return data, meta


def scene_drift() -> tuple[np.ndarray, dict]:
    from core.benchmark_registry import generate_benchmark_sample

    data, meta = generate_benchmark_sample("drift_background_reference")
    return data, meta


def scene_clutter() -> tuple[np.ndarray, dict]:
    from core.benchmark_registry import generate_benchmark_sample

    data, meta = generate_benchmark_sample("clutter_gain_reference")
    return data, meta


def scene_motion() -> tuple[np.ndarray, dict]:
    from core.benchmark_registry import generate_benchmark_sample

    data, meta = generate_benchmark_sample("motion_compensation_v1")
    meta["time_step_ns"] = 180.0 / (data.shape[0] - 1)
    mc = meta.setdefault("expected_metrics", {}).get("metric_config") or {}
    meta["metric_config"] = mc
    return data, meta


def scene_denoise_snr() -> tuple[np.ndarray, dict]:
    """干净合成 + 高斯噪声 + 椒盐尖峰, 留干净参考算 SNR 提升 (dB)。"""
    from core.benchmark_registry import generate_benchmark_sample

    data, meta = generate_benchmark_sample("drift_background_reference")
    rng = np.random.default_rng(20260905)
    clean = data.astype(np.float64).copy()
    noisy = clean + 0.12 * rng.normal(size=clean.shape)
    n_spikes = 24
    for _ in range(n_spikes):
        r = int(rng.integers(0, clean.shape[0]))
        c = int(rng.integers(0, clean.shape[1]))
        noisy[r, c] += float(rng.choice([-1.0, 1.0])) * 6.0
    meta = dict(meta)
    meta["clean_reference"] = clean
    return noisy, meta


def scene_qc_outliers() -> tuple[np.ndarray, dict]:
    """注入已知坏道 (静默道 + 尖峰道), 考核 trace_qc 检测 F1。"""
    from core.benchmark_registry import generate_benchmark_sample

    data, meta = generate_benchmark_sample("zero_time_reference")
    rng = np.random.default_rng(20260906)
    n = data.shape[1]
    dead_idx = [3, 17, 41, 58]
    spike_idx = [9, 30, 66]
    for j in dead_idx:
        data[:, j] = 1e-6 * rng.normal(size=data.shape[0])
    for j in spike_idx:
        rows = rng.integers(0, data.shape[0], size=6)
        data[rows, j] += float(rng.choice([-1.0, 1.0])) * 25.0
    meta = dict(meta)
    meta["injected_bad_traces"] = sorted(dead_idx + spike_idx)
    return data, meta


def scene_fk_dip() -> tuple[np.ndarray, dict]:
    """水平反射 + 一条强倾斜相干噪声 (60 度左右), 考核 fk_filter 去斜率保平层。"""
    samples, traces = 220, 96
    dt_ns = 3.2
    t = np.arange(samples)[:, None] * dt_ns
    x = np.arange(traces)[None, :] * 0.09
    rng = np.random.default_rng(20260907)
    wave = np.exp(-((t - 90.0) / 14.0) ** 2) * np.cos(2 * np.pi * 0.06 * (t - 90.0))
    wave2 = np.exp(-((t - 150.0) / 12.0) ** 2) * np.cos(2 * np.pi * 0.09 * (t - 150.0))
    data = 1.2 * wave * np.ones_like(x) + 0.9 * wave2 * np.ones_like(x)
    slope = 1.0 / math.tan(math.radians(58.0))  # ns/m → 视速度
    dip = np.exp(-((t - x / slope - 20.0) / 10.0) ** 2) * np.cos(
        2 * np.pi * 0.08 * (t - x / slope - 20.0)
    )
    data = data + 1.8 * dip + 0.02 * rng.normal(size=(samples, traces))
    meta = _meta(samples * dt_ns, 0.09)
    meta["dip_slope_ns_per_m"] = slope
    return data, meta


def scene_migration() -> tuple[np.ndarray, dict]:
    """已知速度的点绕射双曲线 (v=0.1 m/ns, dx=0.09 m), 考核偏移聚焦。"""
    samples, traces = 200, 96
    dt_ns = 3.5
    t0_ns = 120.0
    x = (np.arange(traces) - traces // 2) * 0.09
    t = np.arange(samples)[:, None] * dt_ns
    rng = np.random.default_rng(20260908)
    hyper_t = np.sqrt(t0_ns**2 + (2.0 * x / C) ** 2)  # 双程走时
    amp = 1.0 / np.maximum(hyper_t / t0_ns, 1e-6) ** 0.5
    wave = np.exp(-((t - hyper_t[None, :]) / 8.0) ** 2) * np.cos(
        2 * np.pi * 0.07 * (t - hyper_t[None, :])
    )
    data = amp * wave + 0.015 * rng.normal(size=(samples, traces))
    meta = _meta(samples * dt_ns, 0.09)
    meta["diffraction"] = {
        "t0_ns": t0_ns,
        "vertex_trace": traces // 2,
        "dx_m": 0.09,
        "dt_ns": dt_ns,
    }
    return data, meta


def scene_envelope() -> tuple[np.ndarray, dict]:
    """已知脉冲行位置, 考核希尔伯特包络峰值定位。"""
    samples, traces = 160, 24
    t = np.arange(samples)[:, None] * 4.0
    rng = np.random.default_rng(20260909)
    pulse_rows = [40, 80, 120]
    data = np.zeros((samples, traces))
    for r in pulse_rows:
        tc = r * 4.0
        env = np.exp(-((t - tc) / 16.0) ** 2)
        data += 1.5 * env * np.cos(2 * np.pi * 0.10 * (t - tc))
    data += 0.05 * rng.normal(size=(samples, traces))
    meta = _meta(samples * 4.0, 0.09)
    meta["pulse_rows"] = pulse_rows
    return data, meta


SCENES = {
    "zero_time": scene_zero_time,
    "drift": scene_drift,
    "clutter": scene_clutter,
    "motion": scene_motion,
    "denoise_snr": scene_denoise_snr,
    "qc_outliers": scene_qc_outliers,
    "fk_dip": scene_fk_dip,
    "migration": scene_migration,
    "envelope": scene_envelope,
}


# ---------------------------------------------------------------- 任务表 ---
# 每个 (scene, method) 一条; params 为该场景下最合理参数; family 用于冗余分组与评分族。

def _kirchhoff_sensible_params(shape, meta):
    samples, traces = shape
    dx = 0.09
    length_m = traces * dx
    # 网格类参数是场景地面真值契约 (fixture), weight 留空以测真实 schema 默认
    # (run#13 起 weight 默认 0.05 = 轻度 TV, 此前 0.5 过度平滑压制绕射尾巴)。
    return {
        "freq": 2.0e7,
        "depth": 8.0,
        "time_window_ns": 160.0,
        "length_m": length_m,
        "v": C,
        "backend": "cpu",
    }


def _rtm_params(shape, meta):
    samples, traces = shape
    return {
        "dt_ns": 3.5,
        "dx_m": 0.09,
        "v": C,
        "time_window_ns": samples * 3.5,
        "length_m": traces * 0.09,
        "depth_m": 18.0,
        "max_grid_elements": 4_000_000,
    }


def _time_step_ns(meta):
    v = meta.get("time_step_ns")
    if v:
        return float(v)
    hdr = meta.get("header_info", {})
    total = float(hdr.get("total_time_ns") or 0)
    if total > 0:
        return total / 191.0
    return 3.665


TASKS = [
    # --- 零时/裁剪族 (zero_time fixture) ---
    ("zero_time", "set_zero_time", {"new_zero_time": 18 * 3.665, "time_step_s": 3.665e-9}, "zero"),
    ("zero_time", "time_cut", {"mode": "remove_below", "time_end_ns": 500.0, "time_window_ns": 700.0}, "zero"),
    ("zero_time", "dewow", {}, "zero"),
    # --- 背景抑制族 (drift fixture) ---
    ("drift", "subtracting_average_2D", {"ntraces": 501}, "background"),
    ("drift", "median_background_2D", {}, "background"),
    ("drift", "sliding_avg", {"window_size": 10, "axis": 1}, "background"),
    ("drift", "svd_bg", {"rank": 1}, "background"),
    ("drift", "rpca_background", {"lam": 0.08}, "background"),
    ("drift", "ccbs", {}, "background"),
    # --- 增益族 (clutter fixture) ---
    ("clutter", "agcGain", {"window": 11}, "gain"),
    ("clutter", "sec_gain", {}, "gain"),
    ("clutter", "compensatingGain", {}, "gain"),
    ("clutter", "energy_decay_gain", {"strength": 1.0}, "gain"),
    ("clutter", "amplitude_scale", {"mode": "constant", "scale": 2.0}, "gain"),
    ("clutter", "frequency_filter_1d", {"sample_rate_mhz": 272.9, "filter_type": "bandpass", "low_freq_mhz": 40.0, "high_freq_mhz": 120.0}, "filter"),
    ("envelope", "hilbert_envelope", {"normalize": True, "log_compress": False}, "envelope"),
    ("fk_dip", "fk_filter", {}, "filter"),
    # --- 去噪族 (denoise_snr scene) ---
    ("denoise_snr", "wavelet_2d", {}, "denoise"),
    ("denoise_snr", "wavelet_svd", {}, "denoise"),
    ("denoise_snr", "svd_subspace", {}, "denoise"),
    ("denoise_snr", "hankel_svd", {"aggressiveness": 0.5}, "denoise"),
    ("denoise_snr", "trace_median_filter", {"window_traces": 5}, "denoise"),
    ("denoise_snr", "trace_savgol_filter", {"window_traces": 7, "polyorder": 2}, "denoise"),
    ("denoise_snr", "running_average_2D", {"ntraces": 9}, "denoise"),
    # --- QC (qc_outliers scene) ---
    ("qc_outliers", "trace_qc", {}, "qc"),
    # --- 运动补偿族 (motion fixture) ---
    ("motion", "motion_compensation_height", {"wave_speed_m_per_ns": C}, "motion"),
    ("motion", "motion_compensation_speed", {}, "motion"),
    ("motion", "motion_compensation_attitude", {}, "motion"),
    ("motion", "motion_compensation_v2", {"air_wave_speed_m_per_ns": C}, "motion"),
    ("motion", "motion_compensation_vibration", {}, "motion"),
    ("motion", "trajectory_smoothing", {"method": "savgol", "window_length": 21, "polyorder": 3}, "motion"),
    # --- 偏移族 (migration scene / kirchhoff 用 sensible 参数) ---
    ("migration", "stolt_migration", {"dx": 0.09, "dt": 3.5, "v": C}, "migration"),
    ("migration", "time_to_depth", {"dt": 3.5, "v": C, "dz": 0.02}, "migration"),
    ("migration", "kirchhoff_migration", "KIRCHHOFF_SENSIBLE", "migration"),
    ("migration", "rtm_migration", "RTM_PARAMS", "migration"),
]

FAMILIES = {}
for _scene, _method, _params, _family in TASKS:
    FAMILIES.setdefault(_family, []).append((_scene, _method))


def _finite(x) -> bool:
    return x is not None and math.isfinite(float(x))


def _clamp01(x: float) -> float:
    return 0.0 if not math.isfinite(x) else max(0.0, min(1.0, x))


def _snr_db(x: np.ndarray, ref: np.ndarray) -> float:
    err = x - ref
    return 10.0 * math.log10(
        (float(np.mean(ref**2)) + 1e-12) / (float(np.mean(err**2)) + 1e-12)
    )


def _horizontal_coherence(arr: np.ndarray) -> float:
    """列间平均相关系数, 越高说明水平条带越强。"""
    a = arr - arr.mean(axis=0, keepdims=True)
    norm = np.sqrt((a * a).sum(axis=0)) + 1e-12
    corr = (a.T @ a) / np.outer(norm, norm)
    n = corr.shape[0]
    if n <= 1:
        return 0.0
    iu = np.triu_indices(n, k=1)
    return float(corr[iu].mean())


def _quality_snapshot(before, after, meta, zero_idx=None):
    """成对调用仓库统一质量指标 (before, after); 失败返回空 dict。"""
    from mygpr.domain.autotune.quality_metrics import (
        compute_benchmark_metrics,
        compute_motion_quality_metrics,
    )

    out = {}
    try:
        out.update(compute_benchmark_metrics(before, after, zero_idx))
    except Exception:
        pass
    mc = meta.get("metric_config") or {}
    gt_meta = meta.get("ground_truth_trace_metadata")
    if mc and gt_meta is not None:
        try:
            out.update(
                compute_motion_quality_metrics(
                    after,
                    meta.get("trace_metadata", {}),
                    gt_meta,
                    ground_truth_data=meta.get("ground_truth_data"),
                    ridge_row_range=mc.get("ridge_row_range"),
                    target_row_range=mc.get("target_row_range"),
                    banding_trace_band=mc.get("banding_trace_band", (0.05, 0.18)),
                    banding_row_range=mc.get("banding_row_range"),
                )
            )
        except Exception:
            pass
    return out


def _motion_score(q_after: dict, q_before: dict) -> float:
    """运动补偿打分: 各轨迹误差指标相对补偿前的下降比例取平均。"""
    ratios = []
    for key in (
        "raw_ridge_rmse_samples",
        "path_rmse_m",
        "footprint_rmse_m",
        "periodic_banding_ratio",
    ):
        a = q_after.get(key)
        b = q_before.get(key)
        if not (_finite(a) and _finite(b)):
            continue
        ratios.append(_clamp01(1.0 - float(a) / max(float(b), 1e-9)))
    return float(np.mean(ratios)) if ratios else 0.0

def _score_task(scene, method, before, after, out, meta, res_meta, warnings) -> tuple[float, dict]:
    """按族打分, 返回 (score, detail)。所有分量归一到 [0,1] 再平均。"""
    zero_idx = meta.get("ref_zero_idx")
    comps: list[float] = []
    detail: dict = {}
    # 退化输出防护: 输入非平凡而输出全零 = 灾难性失败 (如 kirchhoff TV 发散),
    # 直接 0 分, 不走族内分量平均的兜底假象。
    if not np.any(out) and np.any(before):
        return 0.0, {"degenerate_output": True}

    if scene == "zero_time" and method == "set_zero_time":
        # run#15 仪器修正: set_zero_time 正确执行后波形上移 shift_samples 行, 原参考零位
        # 内容落到第 (ref_zero_idx − shift_samples) 行 (本 fixture: 18−18=0, 新零=第0行)。
        # after 侧指标必须在【新零位】测量——旧实现仍在原零位 (ref_zero_idx) 测量, 对理想
        # 零化输出给出 fb=0.033/pe=0.796 (分 0.0415), 而对"什么都不做"给出 0.776 —— 仪器
        # 倒置, 奖励不作为、惩罚教科书正确行为。before 侧保持原零位不变 (配对语义)。
        shift = res_meta.get("shift_samples")
        base = zero_idx if isinstance(zero_idx, (int, float)) else 0
        if isinstance(shift, (int, float)):
            after_zero_idx = max(0, int(round(float(base) - float(shift))))
        else:
            # shift 未知时按方法契约回退: 成功执行后新零恒为第 0 行
            after_zero_idx = 0
        from mygpr.domain.autotune.quality_metrics import (
            first_break_sharpness as _fb,
            pre_zero_energy_ratio as _pe,
        )
        fb_new = _fb(after, after_zero_idx)
        pe_new = _pe(after, after_zero_idx)
        if _finite(fb_new) and _finite(pe_new):
            comps.append(_clamp01(fb_new / 0.4))
            comps.append(_clamp01(1.0 - pe_new / 0.5))
        detail.update({"first_break": fb_new, "pre_zero_ratio": pe_new,
                       "after_zero_idx": after_zero_idx})

    if scene == "zero_time" and method == "time_cut":
        rows_before = before.shape[0]
        rows_after = after.shape[0]
        expect_keep = int(round(500.0 / (700.0 / (rows_before - 1)))) + 1
        keep_err = abs(rows_after - expect_keep) / rows_before
        comps.append(_clamp01(1.0 - keep_err * 4.0))
        if rows_after > 0 and rows_after <= rows_before:
            upper_err = float(np.mean((after - before[:rows_after]) ** 2))
            scale = float(np.mean(before[:rows_after] ** 2)) + 1e-12
            comps.append(_clamp01(1.0 - math.sqrt(upper_err / scale) * 5.0))
        detail["rows_after"] = rows_after

    if method == "dewow":
        q = _quality_snapshot(before, after, meta)
        lfe = q.get("low_freq_energy_reduction") or 0.0
        comps.append(_clamp01(lfe * 3.0))
        edge = q.get("edge_preservation")
        if _finite(edge):
            comps.append(_clamp01(edge / 0.8))
        sal = q.get("local_saliency_preservation")
        if _finite(sal):
            comps.append(_clamp01(sal / 0.8))
        detail["low_freq_reduction"] = lfe

    if family_of(scene, method) == "background":
        q = _quality_snapshot(before, after, meta)
        coh = q.get("horizontal_coherence_reduction") or 0.0
        comps.append(_clamp01(0.5 + coh))
        sal = q.get("local_saliency_preservation")
        if _finite(sal):
            comps.append(_clamp01(sal / 0.8))
        lfe = q.get("low_freq_energy_reduction")
        if _finite(lfe):
            comps.append(_clamp01(lfe * 2.0))
        detail.update({"coherence_red": coh, "saliency": sal})

    if family_of(scene, method) == "gain":
        q = _quality_snapshot(before, after, meta)
        dz = q.get("deep_zone_contrast_gain")
        if _finite(dz):
            comps.append(_clamp01(dz / 2.0))
        clip = q.get("clipping_ratio_after") or 0.0
        hot = q.get("hot_pixel_ratio_after") or 0.0
        comps.append(_clamp01(1.0 - clip * 20.0))
        comps.append(_clamp01(1.0 - hot * 20.0))
        edge = q.get("edge_preservation")
        if _finite(edge):
            comps.append(_clamp01(edge / 0.8))
        detail.update({"deep_zone_gain": dz, "clip": clip, "hot": hot})

    if method == "amplitude_scale":
        exact = float(np.mean((after - 2.0 * before) ** 2)) / (float(np.mean(before**2)) + 1e-12)
        comps.append(_clamp01(1.0 - math.sqrt(exact) * 10.0))
        detail["exact_scale_err"] = exact

    if method == "frequency_filter_1d":
        def band_energy(arr, lo_mhz, hi_mhz):
            n_rows = arr.shape[0]
            dt_ns = 700.0 / (n_rows - 1)
            spec = np.abs(np.fft.rfft(arr - arr.mean(axis=0), axis=0)) ** 2
            freqs = np.fft.rfftfreq(n_rows, d=dt_ns / 1e3)
            band = (freqs >= lo_mhz) & (freqs <= hi_mhz)
            total = float(spec.sum())
            return float(spec[band].sum()) / (total + 1e-12)

        in_ratio = band_energy(after, 40.0, 120.0)
        comps.append(_clamp01(in_ratio * 1.5))
        detail["in_band_ratio"] = in_ratio



    if method == "fk_filter":
        # 倾斜噪声衰减 vs 平层保持: 中心行带能量比 + 相干能量差
        def dip_energy(arr):
            spec2 = np.fft.fft2(arr - arr.mean())
            ky = np.fft.fftfreq(arr.shape[0])
            kx = np.fft.fftfreq(arr.shape[1])
            KX, KY = np.meshgrid(kx, ky)
            ang = np.degrees(np.arctan2(np.abs(KY), np.abs(KX) + 1e-12))
            return float(np.abs(spec2)[(ang >= 10) & (ang <= 65)].sum())

        de_b = dip_energy(before)
        de_a = dip_energy(after)
        comps.append(_clamp01(1.0 - de_a / max(de_b, 1e-12)))
        # 平层保持: 非噪声扇区能量
        def flat_energy(arr):
            spec2 = np.fft.fft2(arr - arr.mean())
            ky = np.fft.fftfreq(arr.shape[0])
            kx = np.fft.fftfreq(arr.shape[1])
            KX, KY = np.meshgrid(kx, ky)
            ang = np.degrees(np.arctan2(np.abs(KY), np.abs(KX) + 1e-12))
            return float(np.abs(spec2)[ang < 10].sum())

        fe_b = flat_energy(before)
        fe_a = flat_energy(after)
        comps.append(_clamp01(fe_a / max(fe_b, 1e-12)))
        detail.update({"dip_suppression": comps[-2], "flat_preserve": comps[-1]})

    if family_of(scene, method) == "denoise":
        clean = meta.get("clean_reference")
        if clean is not None:
            snr_b = _snr_db(before, clean)
            snr_a = _snr_db(after, clean)
            comps.append(_clamp01((snr_a - snr_b) / 12.0))  # +12dB 封顶
            shape_err = float(np.mean((after - clean) ** 2)) / (float(np.mean(clean**2)) + 1e-12)
            comps.append(_clamp01(1.0 - shape_err * 2.0))
            detail.update({"snr_before": snr_b, "snr_after": snr_a})

    if method == "trace_qc":
        truth = set(meta.get("injected_bad_traces", []))
        mask = res_meta.get("trace_qc_bad_mask")
        if mask is None:
            updates = res_meta.get("trace_metadata_updates")
            if isinstance(updates, dict):
                mask = updates.get("trace_qc_bad_mask")
        if mask is None:
            comps.append(0.0)
            detail["qc_mask_missing"] = True
        else:
            pred = set(np.nonzero(np.asarray(mask).reshape(-1).astype(int))[0].tolist())
            tp = len(pred & truth)
            fp = len(pred - truth)
            fn = len(truth - pred)
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            comps.append(f1)
            detail.update({"f1": f1, "tp": tp, "fp": fp, "fn": fn})

    if family_of(scene, method) == "motion":
        # 等距重采样会改变道数 (95 vs 96): 成对指标按最短道数对齐后再算
        if after.shape[1] != before.shape[1]:
            n = min(after.shape[1], before.shape[1])
            after_aligned = after[:, :n]
        else:
            after_aligned = after
        q_a = _quality_snapshot(before, after_aligned, meta)
        q_b = _quality_snapshot(before, before, meta)
        comps.append(_motion_score(q_a, q_b))
        tp = q_a.get("target_preservation_ratio")
        if _finite(tp) and tp < 0.8:
            comps.append(_clamp01(tp / 0.8))
        detail.update({k: v for k, v in q_a.items() if _finite(v)})

    if family_of(scene, method) == "migration":
        d = meta.get("diffraction")
        if d:
            vc = d["vertex_trace"]
            t0 = d["t0_ns"]
            dt = d["dt_ns"]
            row0 = int(round(t0 / dt))
            apex = after[max(row0 - 6, 0) : row0 + 7, vc]
            apex_before = before[max(row0 - 6, 0) : row0 + 6 + 1, vc]
            # 聚焦: 顶点列能量集中度 (apex 能量 / 全域能量)
            e_all = float(np.mean(after**2))
            e_apex = float(np.mean(apex**2))
            comps.append(_clamp01((e_apex / (e_all + 1e-12)) * 40.0))
            # 双曲线压平: 顶点外剩余能量
            tail_mask = np.ones(after.shape[1], dtype=bool)
            tail_mask[vc] = False
            tail = after[max(row0 - 10, 0) : row0 + 11, :][:, tail_mask]
            tail_b = before[max(row0 - 10, 0) : row0 + 11, :][:, tail_mask]
            comps.append(_clamp01(1.0 - float(np.mean(tail**2)) / (float(np.mean(tail_b**2)) + 1e-12)))
            detail["apex_concentration"] = comps[-2]

    if method == "hilbert_envelope":
        rows = meta.get("pulse_rows", [])
        env = after
        peaks = np.argmax(np.abs(env), axis=0)
        acc = np.mean([1.0 if min(abs(p - r) for r in rows) <= 2 else 0.0 for p in peaks])
        comps.append(acc)
        detail["peak_accuracy"] = acc

    if not comps:
        # 兜底: 有限输出 + 非平凡变化 → 0.5
        finite_ok = bool(np.all(np.isfinite(out)))
        changed = not np.array_equal(out, before)
        comps.append(0.5 if (finite_ok and changed) else 0.0)

    score = float(np.mean(comps))
    return score, detail


_FAMILY_MAP = {}
for _scene, _method, _params, _family in TASKS:
    _FAMILY_MAP[(_scene, _method)] = _family


def family_of(scene, method) -> str:
    return _FAMILY_MAP.get((scene, method), "misc")


def _resolve_params(scene, method, params, data, meta):
    if params == "KIRCHHOFF_SENSIBLE":
        return _kirchhoff_sensible_params(data.shape, meta)
    if params == "RTM_PARAMS":
        return _rtm_params(data.shape, meta)
    return dict(params)


def _run_single(scene, method):
    """在子进程里跑单个任务, 返回 (status, score, detail, out_npy_path, corr_tag)。"""

    t0 = time.perf_counter()
    data, meta = SCENES[scene]()
    before = np.asarray(data, dtype=np.float64)
    params = _resolve_params(scene, method, TASK_PARAMS[(scene, method)], data, meta)
    try:
        from mygpr.domain.processing.models import ProcessingRequest
        from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor
        from mygpr.application.jobs.context import ExecutionContext

        req = ProcessingRequest(
            method_id=method,
            data=before,
            params=params,
            header_info=dict(meta.get("header_info", {})),
            trace_metadata=dict(meta.get("trace_metadata", {})),
        )
        res = NativeProcessingExecutor().execute(req, ExecutionContext.null())
        out = np.asarray(res.data, dtype=np.float64)
        res_meta = dict(res.metadata or {})
        warnings = list(res.runtime_warnings or [])
    except Exception as exc:  # noqa: BLE001
        dur = time.perf_counter() - t0
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_s": dur,
        }

    if out.shape != before.shape:
        # 网格/轴变换类方法 (time_to_depth 等): 记录 shape change, 不比对
        status = "shape_change"
    elif not np.all(np.isfinite(out)):
        status = "nonfinite"
    else:
        status = "ok"

    try:
        score, detail = _score_task(
            scene, method, before, out, out, meta, res_meta, warnings
        )
    except Exception as exc:  # noqa: BLE001
        score, detail = 0.0, {"score_error": f"{type(exc).__name__}: {exc}"}
    # 冗余分析: 保存输出矩阵
    npy_path = None
    if status == "ok" and out.shape == before.shape:
        tmpdir = os.environ.get("FITNESS_NPY_DIR")
        if tmpdir:
            Path(tmpdir).mkdir(parents=True, exist_ok=True)
            npy_path = os.path.join(tmpdir, f"{scene}__{method}.npy")
            np.save(npy_path, out.astype(np.float32))
    return {
        "status": status,
        "score": score,
        "detail": detail,
        "runtime_s": time.perf_counter() - t0,
        "warnings": warnings[:5],
        "npy_path": npy_path,
        "data_noop": bool(np.array_equal(out, before)),
    }

    if as_json:
        sys.stdout.write(json.dumps({"scene": scene, "method": method, **r}) + "\n")
    else:
        print(json.dumps({"scene": scene, "method": method, **r}))
    return 0


def _worker_mode(scene, method, as_json: bool) -> int:
    r = _run_single(scene, method)
    if as_json:
        sys.stdout.write(json.dumps({"scene": scene, "method": method, **r}) + "\n")
    else:
        print(json.dumps({"scene": scene, "method": method, **r}))
    return 0



def _run_all() -> int:
    import tempfile


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npy_dir = tempfile.mkdtemp(prefix="mygpr_fitness_npy_")
    env = dict(os.environ)
    env["FITNESS_NPY_DIR"] = npy_dir
    env["PYTHONIOENCODING"] = "utf-8"

    results = []
    for scene, method, params, family in TASKS:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--scene",
                    scene,
                    "--method",
                    method,
                    "--json",
                ],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(REPO_ROOT),
                timeout=WORKER_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - t0
            results.append(
                {
                    "scene": scene,
                    "method": method,
                    "family": family,
                    "wall_s": wall,
                    "status": "timeout",
                    "score": 0.15,
                    "error": f"worker timeout after {WORKER_TIMEOUT_S:.0f}s",
                }
            )
            print(f"  [ timeout] {scene}/{method}: score=0.15 wall={wall:.1f}s", flush=True)
            continue
        wall = time.perf_counter() - t0
        entry = {
            "scene": scene,
            "method": method,
            "family": family,
            "wall_s": wall,
        }
        if proc.returncode != 0:
            err_tail = (proc.stderr or "")[-400:]
            if "TimeoutExpired" in err_tail or wall >= WORKER_TIMEOUT_S - 0.5:
                entry.update({"status": "timeout", "score": 0.15, "error": err_tail})
            else:
                entry.update({"status": "error", "score": 0.0, "error": err_tail})
        else:
            try:
                payload = json.loads(proc.stdout.strip().splitlines()[-1])
            except Exception as exc:  # noqa: BLE001
                entry.update({"status": "error", "score": 0.0, "error": f"bad worker stdout: {exc}"})
            else:
                entry.update({k: payload.get(k) for k in ("status", "score", "detail", "runtime_s", "warnings", "npy_path", "error", "data_noop")})
        results.append(entry)
        flag = entry.get("status", "?")
        print(f"  [{flag:>8}] {scene}/{method}: score={entry.get('score')} wall={wall:.1f}s", flush=True)

    # ---- 冗余检测 ----
    redundant_pairs = []
    by_family = {}
    for e in results:
        if e.get("status") == "ok" and e.get("npy_path") and not e.get("data_noop"):
            by_family.setdefault(e["family"], []).append(e)
    for fam, entries in by_family.items():
        mats = []
        for e in entries:
            try:
                m = np.load(e["npy_path"]).astype(np.float64)
                mats.append((e, m.reshape(-1)))
            except Exception:
                pass
        for i in range(len(mats)):
            for j in range(i + 1, len(mats)):
                a, b = mats[i][1], mats[j][1]
                if a.size != b.size:
                    continue
                sa, sb = a.std(), b.std()
                if sa < 1e-9 or sb < 1e-9:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])
                if corr > REDUNDANCY_CORR_THRESHOLD:
                    redundant_pairs.append(
                        {"family": fam, "a": mats[i][0]["method"], "b": mats[j][0]["method"], "corr": corr}
                    )

    # ---- 聚合 ----
    scores = [e["score"] for e in results if isinstance(e.get("score"), (int, float))]
    broken = sum(1 for e in results if e.get("status") == "error")
    timeouts = sum(1 for e in results if e.get("status") == "timeout")
    total_runtime = sum(e.get("wall_s", 0.0) for e in results)
    fitness = 100.0 * float(np.mean(scores)) if scores else 0.0

    report = {
        "fitness_score": fitness,
        "broken_count": broken,
        "timeout_count": timeouts,
        "redundant_pairs": redundant_pairs,
        "total_runtime_s": total_runtime,
        "results": results,
    }
    out_path = OUTPUT_DIR / "algorithm_fitness.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== per-method ===")
    for e in results:
        print(f"{e['scene']:<12} {e['method']:<28} {e.get('status','?'):>8} score={e.get('score')}")
    print("\nredundant pairs:")
    for p in redundant_pairs:
        print(f"  {p['family']}: {p['a']} ≡ {p['b']} (corr={p['corr']:.4f})")
    if not redundant_pairs:
        print("  (none)")

    print(f"METRIC fitness_score={fitness:.2f}")
    print(f"METRIC broken_count={broken}")
    print(f"METRIC timeout_count={timeouts}")
    print(f"METRIC redundant_pair_count={len(redundant_pairs)}")
    print(f"METRIC total_runtime_s={total_runtime:.1f}")
    return 0


TASK_PARAMS = {(s, m): p for s, m, p, _f in TASKS}


def main() -> int:
    ap = argparse.ArgumentParser(description="MyGPR 算法体检基准")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--scene")
    ap.add_argument("--method")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if args.worker:
        if not args.scene or not args.method:
            print("worker 模式需要 --scene/--method", file=sys.stderr)
            return 2
        return _worker_mode(args.scene, args.method, args.json)
    return _run_all()


if __name__ == "__main__":
    sys.exit(main())
