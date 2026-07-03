#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting helpers for the MyGPR field workbench.

The plotting helpers are intentionally UI-only preview/rendering utilities.  They
must not contain project persistence or processing execution logic.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Arc, Rectangle

from ui.field_panels.field_ui_styles import TEXT

def synthetic_bscan(rows: int = 220, cols: int = 360, seed: int = 4) -> np.ndarray:
    """Generate a dense GPR-like B-scan texture for UI preview screenshots.

    The previous preview intentionally used very clean hyperbolas.  For the
    field-workbench visual target we need a busier, radargram-like texture with
    shallow ringing, random clutter and overlapping diffraction responses.
    This remains deterministic and lightweight; it does not touch real backend
    data paths.
    """
    rng = np.random.default_rng(seed)
    y = np.arange(rows, dtype=float)[:, None]
    x = np.arange(cols, dtype=float)[None, :]
    yy = y / max(rows - 1, 1)

    # Layered background ringing and shallow horizontal reflectors.
    data = 0.16 * rng.normal(size=(rows, cols))
    data += 0.34 * np.sin(0.23 * y + 0.012 * x) * np.exp(-yy * 1.2)
    data += 0.20 * np.sin(0.48 * y + 0.035 * np.sin(x / 18.0)) * np.exp(-yy * 1.8)
    data += 0.11 * np.cos(0.92 * y + 0.010 * x) * np.exp(-yy * 2.5)
    for depth, amp, freq in [(18, 0.85, 1.1), (33, 0.46, 0.9), (48, 0.30, 0.75)]:
        band = np.exp(-((y - depth) ** 2) / (2 * 5.5**2))
        data += amp * band * np.cos(freq * (y - depth) + 0.03 * x)

    # Main diffraction hyperbolas.
    centers = [(42, 62, 9, 1.3), (92, 74, 8, 1.1), (152, 58, 10, 1.4), (235, 78, 9, 1.2), (303, 92, 8, 0.9)]
    for cx, cy, sigma, amp in centers:
        curve = cy + 0.0105 * (x - cx) ** 2
        wave = np.exp(-((y - curve) ** 2) / (2 * sigma**2)) * np.cos((y - curve) * 1.25)
        aperture = np.exp(-((x - cx) ** 2) / (2 * (cols * 0.17) ** 2))
        data += amp * wave * aperture

    # Smaller clutter diffractions and local discontinuities.
    for _ in range(16):
        cx = rng.uniform(15, cols - 15)
        cy = rng.uniform(55, rows * 0.78)
        sigma = rng.uniform(4.0, 8.0)
        curvature = rng.uniform(0.006, 0.018)
        amp = rng.uniform(0.18, 0.55) * rng.choice([-1, 1])
        curve = cy + curvature * (x - cx) ** 2
        aperture = np.exp(-((x - cx) ** 2) / (2 * rng.uniform(25, 70) ** 2))
        data += amp * np.exp(-((y - curve) ** 2) / (2 * sigma**2)) * np.cos((y - curve) * rng.uniform(0.9, 1.5)) * aperture

    # Column-wise gain/noise variation to avoid a too-perfect mockup.
    trace_gain = 1.0 + 0.08 * np.sin(np.linspace(0, 5 * np.pi, cols))[None, :] + 0.04 * rng.normal(size=(1, cols))
    data *= trace_gain
    data += 0.05 * rng.normal(size=(rows, 1))
    data -= data.mean()
    data /= max(float(np.percentile(np.abs(data), 99.5)), 1e-6)
    return np.clip(data, -1.0, 1.0)

def _style_axis(ax, *, title: str | None = None) -> None:
    ax.tick_params(labelsize=7, colors="#40566C", length=2)
    for spine in ax.spines.values():
        spine.set_color("#D5DEE7")
    ax.grid(True, color="#EAF0F6", linewidth=0.55, alpha=0.65)
    if title:
        ax.set_title(title, loc="left", fontsize=9, color=TEXT, pad=6, fontweight="bold")


def draw_bscan(
    canvas: FigureCanvas,
    *,
    title: str = "B-scan 剖面（处理后）",
    diff: bool = False,
    gain: float = 1.0,
    seed: int | None = None,
    data_matrix: np.ndarray | None = None,
    base_matrix: np.ndarray | None = None,
    distance_axis_m: np.ndarray | None = None,
    depth_axis_m: np.ndarray | None = None,
) -> None:
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    if data_matrix is None:
        data = synthetic_bscan(seed=seed if seed is not None else (8 if diff else 2))
    else:
        data = np.asarray(data_matrix, dtype=float)
        scale = float(np.nanpercentile(np.abs(data), 99.5)) if data.size else 1.0
        if not np.isfinite(scale) or scale <= 1e-9:
            scale = 1.0
        data = np.clip((data - float(np.nanmean(data))) / scale, -1.0, 1.0)
    if diff:
        if base_matrix is None:
            base = synthetic_bscan(seed=2)
            if base.shape != data.shape:
                base = np.resize(base, data.shape)
        else:
            base = np.asarray(base_matrix, dtype=float)
            scale = float(np.nanpercentile(np.abs(base), 99.5)) if base.size else 1.0
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = 1.0
            base = np.clip((base - float(np.nanmean(base))) / scale, -1.0, 1.0)
        data = (data - base) * max(0.55, gain)
        im = ax.imshow(data, cmap="seismic", aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    else:
        data = np.clip(data * max(0.65, gain), -1.0, 1.0)
        im = ax.imshow(data, cmap="gray", aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    ax.set_title(title, fontsize=9, loc="left", fontweight="bold", color=TEXT)
    ax.set_xlabel("距离 (m)", fontsize=8, color="#394B5F")
    ax.set_ylabel("深度 (m)" if depth_axis_m is not None and len(depth_axis_m) else "时间 (ns)", fontsize=8, color="#394B5F")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(np.linspace(0, data.shape[1] - 1, 7))
    x_end = float(distance_axis_m[-1]) if distance_axis_m is not None and len(distance_axis_m) else 212.0
    ax.set_xticklabels([str(int(v)) for v in np.linspace(0, x_end, 7)])
    ax.set_yticks(np.linspace(0, data.shape[0] - 1, 6))
    y_end = float(depth_axis_m[-1]) if depth_axis_m is not None and len(depth_axis_m) else 5.0
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, y_end, 6)])
    ax.tick_params(labelsize=7, colors="#33465A")
    for spine in ax.spines.values():
        spine.set_color("#D5DEE7")
    if diff:
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.12)
        cbar.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=1.0)
    canvas.draw_idle()


def _normalize_bscan_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return synthetic_bscan(rows=190, cols=420, seed=11)
    scale = float(np.nanpercentile(np.abs(arr), 99.5)) if arr.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return np.clip((arr - float(np.nanmean(arr))) / scale, -1.0, 1.0)


def _target_column_from_mileage(mileage: float, distance_axis_m: np.ndarray, cols: int) -> float:
    if distance_axis_m is not None and len(distance_axis_m) >= 2:
        axis = np.asarray(distance_axis_m, dtype=float)
        if np.isfinite(axis).all() and float(axis[-1] - axis[0]) != 0.0:
            frac = (float(mileage) - float(axis[0])) / float(axis[-1] - axis[0])
            return float(np.clip(frac, 0.0, 1.0) * max(cols - 1, 1))
    return float(mileage) / 200.0 * max(cols - 1, 1)


def _target_row_from_depth(depth_m: float, depth_axis_m: np.ndarray | None, rows: int) -> float:
    if depth_axis_m is not None and len(depth_axis_m) >= 2:
        axis = np.asarray(depth_axis_m, dtype=float)
        if np.isfinite(axis).all() and float(axis[-1] - axis[0]) != 0.0:
            frac = (float(depth_m) - float(axis[0])) / float(axis[-1] - axis[0])
            return float(np.clip(frac, 0.0, 1.0) * max(rows - 1, 1))
    return float(np.clip(45 + depth_m * 22, 0, max(rows - 1, 1)))


def draw_target_bscan(
    canvas: FigureCanvas,
    targets: list[dict] | None = None,
    *,
    selected: int = 2,
    data_matrix: np.ndarray | None = None,
    distance_axis_m: np.ndarray | None = None,
    depth_axis_m: np.ndarray | None = None,
    vertical_axis: np.ndarray | None = None,
    vertical_axis_label: str = "时间 (ns)",
    source_label: str = "",
) -> None:
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    data = _normalize_bscan_matrix(data_matrix) if data_matrix is not None else synthetic_bscan(rows=190, cols=420, seed=11)
    rows, cols = data.shape
    ax.imshow(data, cmap="gray", aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    default_targets = [
        {"name": "T-01", "mileage": 18.62, "depth": 1.35, "color": "#25B26B", "width": 46, "height": 70},
        {"name": "T-02", "mileage": 62.47, "depth": 2.02, "color": "#7C4DFF", "width": 45, "height": 58},
        {"name": "T-03", "mileage": 96.83, "depth": 1.60, "color": "#F04444", "width": 100, "height": 78},
        {"name": "T-04", "mileage": 142.18, "depth": 1.25, "color": "#2B86F6", "width": 48, "height": 64},
        {"name": "T-05", "mileage": 179.41, "depth": 1.90, "color": "#F5A623", "width": 38, "height": 50},
    ]
    active_targets = targets or default_targets
    for idx, target in enumerate(active_targets):
        color = target.get("color", "#0D91B2")
        label = target.get("name", f"T-{idx+1:02d}")
        x = _target_column_from_mileage(float(target.get("mileage", 80.0)), distance_axis_m, cols)
        y = _target_row_from_depth(float(target.get("depth", 1.4)), depth_axis_m, rows)
        w = float(target.get("width", max(cols * 0.12, 28)))
        h = float(target.get("height", max(rows * 0.22, 28)))
        # Keep annotations inside the displayed B-scan even when source artifacts
        # have a different sample count from the raw demo data.
        y_arc = float(np.clip(y - h * 0.45, 8, max(rows - h * 0.35, 12)))
        y_base = float(np.clip(y, 0, rows - 1))
        ax.add_patch(Arc((x, y_arc + h / 2), w, h, theta1=200, theta2=340, color=color, lw=2.0))
        ax.plot([x, x], [y_arc + h * 0.48, y_base], linestyle="--", color=color, lw=1.0, alpha=0.9)
        ax.scatter([x], [y_base], s=42, color="white", edgecolor=color, linewidth=1.8, zorder=4)
        if idx == selected:
            ax.add_patch(Rectangle((x - w / 2 - 7, y_arc - 12), w + 14, h + 28, facecolor=color, edgecolor=color, alpha=0.18, lw=1.8))
        ax.text(
            x - 10,
            max(y_arc - 7, 1),
            label,
            fontsize=8,
            color="white",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.24", facecolor=color, edgecolor=color),
        )
    title = "目标定位 B-scan"
    if source_label:
        title += f"｜{source_label}"
    ax.set_title(title, loc="left", fontsize=9, color=TEXT, pad=6, fontweight="bold")
    ax.set_xlabel("里程 (m)", fontsize=8)
    ax.set_ylabel(vertical_axis_label, fontsize=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    xticks = np.linspace(0, cols - 1, 11)
    ax.set_xticks(xticks)
    x_end = float(distance_axis_m[-1]) if distance_axis_m is not None and len(distance_axis_m) else 200.0
    ax.set_xticklabels([str(int(v)) for v in np.linspace(0, x_end, 11)])
    yticks = np.linspace(0, rows - 1, 6)
    ax.set_yticks(yticks)
    if vertical_axis is not None and len(vertical_axis) >= 2:
        y_end = float(vertical_axis[-1])
    else:
        y_end = 5.0 if "深度" in vertical_axis_label else 250.0
    ax.set_yticklabels([f"{v:.1f}" if "深度" in vertical_axis_label else str(int(v)) for v in np.linspace(0, y_end, 6)])
    ax.tick_params(labelsize=7, colors="#33465A")
    for spine in ax.spines.values():
        spine.set_color("#C8D4E0")
    fig.tight_layout(pad=0.8)
    canvas.draw_idle()

def draw_map(canvas: FigureCanvas, *, detailed: bool = True) -> None:
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor("#EEF3F7")
    rng = np.random.default_rng(3)
    for _ in range(18):
        y = rng.uniform(0, 100)
        ax.plot([-5, 105], [y, y + rng.normal(0, 7)], color="#FFFFFF", lw=rng.uniform(2.5, 5.2), alpha=0.95)
        ax.plot([-5, 105], [y, y + rng.normal(0, 7)], color="#D3DDE6", lw=0.7, alpha=0.8)
    for _ in range(14):
        x = rng.uniform(0, 100)
        ax.plot([x, x + rng.normal(0, 6)], [-5, 105], color="#FFFFFF", lw=rng.uniform(2.0, 4.5), alpha=0.95)
        ax.plot([x, x + rng.normal(0, 6)], [-5, 105], color="#D3DDE6", lw=0.7, alpha=0.8)
    ax.add_patch(Rectangle((72, -2), 16, 120, angle=-6, facecolor="#FCECCB", edgecolor="none", alpha=0.85))
    ax.add_patch(Rectangle((50, 0), 8, 100, angle=5, facecolor="#D7F0E8", edgecolor="none", alpha=0.8))
    lines = {
        "L01": (np.array([8, 12, 18, 18, 36, 54, 72, 94]), np.array([22, 58, 83, 91, 92, 91, 86, 88]), "#1677FF"),
        "L02": (np.array([42, 54, 68, 78, 92]), np.array([76, 78, 82, 84, 92]), "#31A36A"),
        "L03": (np.array([11, 23, 38, 50, 55, 60, 70, 82, 96]), np.array([43, 44, 46, 50, 34, 38, 43, 46, 50]), "#0097A7"),
        "L04": (np.array([9, 26, 42, 60, 76, 91]), np.array([20, 21, 18, 17, 25, 45]), "#8E61E9"),
    }
    for name, (x, y, color) in lines.items():
        ax.plot(x, y, color=color, lw=2.2, marker="o", markersize=4, markerfacecolor="white", markeredgewidth=1.2)
        ax.text(x[len(x)//2], y[len(y)//2] + 3.5, name, color="white", fontsize=8, weight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor=color))
    if detailed:
        for x, y in [(56, 50), (58, 45), (61, 40), (63, 36), (54, 72), (72, 56), (47, 44), (78, 36)]:
            ax.scatter([x], [y], s=32, facecolors="#FFE7D5", edgecolors="#F04438", linewidths=1.4, zorder=5)
        ax.text(74, 63, "已验匹配点", color="#E47F21", fontsize=8)
        ax.text(88, 9, "N", fontsize=11, weight="bold", color="#1C2B38")
        ax.annotate("", xy=(92, 28), xytext=(92, 11), arrowprops=dict(arrowstyle="-|>", color="#1C2B38", lw=1.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.05)
    canvas.draw_idle()


def draw_profile(canvas: FigureCanvas) -> None:
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 220, 160)
    y = 24 + 1.6 * np.sin(x / 16) + 0.8 * np.sin(x / 7) + 1.2 * np.exp(-((x - 28) ** 2) / 240)
    ax.plot(x, y, color="#0D91B2", lw=1.5)
    ax.fill_between(x, y, y.min() - 2, color="#0D91B2", alpha=0.12)
    ax.axvline(121.4, color="#0D91B2", lw=1.0, ls="--")
    ax.scatter([121.4], [np.interp(121.4, x, y)], color="white", edgecolor="#0D91B2", s=30, zorder=4)
    ax.set_xlabel("距离 (m)", fontsize=8)
    ax.set_ylabel("高程 (m)", fontsize=8)
    _style_axis(ax, title="高程剖面（沿测线：L03）")
    fig.tight_layout(pad=0.8)
    canvas.draw_idle()


def draw_dem(canvas: FigureCanvas) -> None:
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 45)
    X, Y = np.meshgrid(x, y)
    Z = 22 + 7 * X + 2 * np.sin(7 * Y) - 2.5 * np.exp(-((X - 0.35) ** 2 + (Y - 0.55) ** 2) / 0.04)
    im = ax.imshow(Z, cmap="terrain", aspect="auto", interpolation="bilinear")
    ax.plot([10, 90], [30, 20], color="#1677FF", lw=1.6)
    ax.plot([35, 70], [5, 42], color="#F04438", lw=1.2, ls="--")
    ax.add_patch(Rectangle((48, 8), 34, 28, facecolor="none", edgecolor="#F04438", lw=1.1, linestyle="--"))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.15)
    canvas.draw_idle()


def draw_line_strip(canvas: FigureCanvas, *, marker: float = 96.83) -> None:
    fig = canvas.figure
    fig.clear()
    ax = fig.add_subplot(111)
    draw_bg = False
    if draw_bg:
        pass
    ax.set_facecolor("#F1F5F8")
    x = np.linspace(0, 212, 7)
    y = 0.52 + 0.04 * np.sin(x / 34)
    ax.plot(x, y, color="#0788A6", lw=2.2, marker="o", markersize=4, markerfacecolor="white")
    ax.scatter([marker], [0.54], marker="v", s=75, color="#F04438", zorder=5)
    for road_y in [0.25, 0.50, 0.75]:
        ax.plot([0, 212], [road_y, road_y + 0.05], color="white", lw=5, alpha=0.9)
        ax.plot([0, 212], [road_y, road_y + 0.05], color="#D8E1E8", lw=0.6)
    ax.set_xlim(0, 212)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.tick_params(labelsize=7, colors="#33465A")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.2)
    canvas.draw_idle()


def draw_overview_flow(canvas: FigureCanvas) -> None:
    # not used; kept for future custom painter replacement
    return None


__all__ = [
    "synthetic_bscan",
    "draw_bscan",
    "draw_target_bscan",
    "draw_map",
    "draw_profile",
    "draw_dem",
    "draw_line_strip",
    "draw_overview_flow",
    "_style_axis",
]
