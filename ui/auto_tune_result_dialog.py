#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-tune candidate result dialog."""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QHBoxLayout,
    QGroupBox,
)
from qfluentwidgets import PushButton

from core.theme_manager import get_theme_manager


class AutoTuneResultDialog(QDialog):
    """Display auto-tune summary and all candidate scores."""

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.result = result or {}
        self.sorted_trials = sorted(
            self.result.get("all_trials", []),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        self.setWindowTitle("自动选参结果")
        self.resize(1080, 780)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("自动选参候选评分")
        title.setProperty("class", "titleLarge")
        layout.addWidget(title)

        summary = QLabel(self._build_summary_text())
        summary.setWordWrap(True)
        layout.addWidget(summary)

        profile_row = QHBoxLayout()
        profile_row.setSpacing(8)
        for key in ["conservative", "balanced", "aggressive"]:
            profile_row.addWidget(self._create_profile_box(key), stretch=1)
        layout.addLayout(profile_row)

        self.reason_text = QTextEdit()
        self.reason_text.setReadOnly(True)
        self.reason_text.setMaximumHeight(120)
        self.reason_text.setPlainText(self._build_reason_text())
        layout.addWidget(self.reason_text)

        self.fig = Figure(figsize=(10.5, 4.2), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self._plot_visuals()
        layout.addWidget(self.canvas)

        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(
            [
                "序号",
                "阶段",
                "参数组合",
                "总评分",
                "ROI评分",
                "全图评分",
                "保护项",
                "关键指标",
                "惩罚项",
                "选择理由",
            ]
        )
        self._populate_table()
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch()
        btn_close = PushButton("关闭")
        btn_close.clicked.connect(self.accept)
        button_row.addWidget(btn_close)
        layout.addLayout(button_row)

    def _build_summary_text(self) -> str:
        method_name = self.result.get("method_name", "未知方法")
        best_score = float(self.result.get("best_score", 0.0))
        best_params = self.result.get("best_params", {})
        trials = len(self.sorted_trials)
        family = self.result.get("family", "")
        recommended_key = self.result.get("recommended_profile", "balanced")
        recommended = (self.result.get("profiles", {}) or {}).get(
            recommended_key, {}
        ).get("label") or recommended_key
        roi_info = self.result.get("roi_info", {}) or {}
        roi_label = roi_info.get("label") or roi_info.get("source") or "全图"
        confidence = float(self.result.get("selection_confidence", 0.0))
        margin = float(self.result.get("selection_margin", 0.0))
        failed_trials = len(self.result.get("failed_trials", []))
        execution_stats = self.result.get("execution_stats", {}) or {}
        cache_hits = int(execution_stats.get("cache_hit_count", 0))
        return (
            f"方法: {method_name} | 类型: {family} | ROI: {roi_label} | 候选数: {trials} | 失败候选: {failed_trials} | 缓存命中: {cache_hits} | 推荐档: {recommended} | 稳定性: {confidence:.2f} (margin={margin:.3f}) | 最优得分: {best_score:.4f} | "
            f"最优参数: {json.dumps(best_params, ensure_ascii=False)}"
        )

    def _build_reason_text(self) -> str:
        lines = ["最终选择理由:"]
        lines.append(self.result.get("best_reason", "无"))
        roi_info = self.result.get("roi_info", {}) or {}
        if roi_info:
            lines.append("")
            lines.append(
                f"ROI 来源: {roi_info.get('label') or roi_info.get('source') or '全图'}"
            )
            lines.append(
                f"粗筛/细化: {len(self.result.get('coarse_trials', []))} / {len(self.result.get('fine_trials', []))}"
            )
        execution_stats = self.result.get("execution_stats", {}) or {}
        if execution_stats:
            lines.append("")
            lines.append("执行统计:")
            lines.append(
                f"- 总候选: {int(execution_stats.get('total_trial_count', len(self.sorted_trials)))}"
            )
            lines.append(
                f"- 有效候选: {int(execution_stats.get('valid_trial_count', len(self.sorted_trials)))}"
            )
            lines.append(
                f"- 失败候选: {int(execution_stats.get('failed_trial_count', len(self.result.get('failed_trials', []))))}"
            )
            lines.append(
                f"- 缓存命中: {int(execution_stats.get('cache_hit_count', 0))}"
            )
        failed_trials = self.result.get("failed_trials", []) or []
        if failed_trials:
            lines.append("")
            lines.append("失败候选示例:")
            for trial in failed_trials[:3]:
                lines.append(
                    f"- {json.dumps(trial.get('params', {}), ensure_ascii=False)} => {trial.get('error_type', 'Error')}: {trial.get('error', trial.get('reason', '未知错误'))}"
                )
        metrics = self.result.get("best_metrics", {})
        penalties = self.result.get("best_penalties", {})
        if metrics:
            lines.append("")
            lines.append("关键指标:")
            for key, value in metrics.items():
                lines.append(
                    f"- {key}: {value:.4f}"
                    if isinstance(value, (int, float))
                    else f"- {key}: {value}"
                )
        if penalties:
            lines.append("")
            lines.append("惩罚项:")
            for key, value in penalties.items():
                lines.append(
                    f"- {key}: {value:.4f}"
                    if isinstance(value, (int, float))
                    else f"- {key}: {value}"
                )
        return "\n".join(lines)

    def _populate_table(self):
        trials = self.sorted_trials
        self.table.setRowCount(len(trials))
        for row, trial in enumerate(trials):
            params_text = json.dumps(trial.get("params", {}), ensure_ascii=False)
            metrics_text = json.dumps(trial.get("metrics", {}), ensure_ascii=False)
            penalties_text = json.dumps(trial.get("penalties", {}), ensure_ascii=False)
            values = [
                str(row + 1),
                str(trial.get("stage", "-")),
                params_text,
                f"{float(trial.get('score', 0.0)):.4f}",
                f"{float(trial.get('roi_score', 0.0)):.4f}",
                f"{float(trial.get('full_score', 0.0)):.4f}",
                f"{float(trial.get('guard_score', 0.0)):.4f}",
                metrics_text,
                penalties_text,
                str(trial.get("reason", "")),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def _create_profile_box(self, key: str) -> QGroupBox:
        profile = (self.result.get("profiles", {}) or {}).get(key) or {}
        box = QGroupBox(profile.get("label", key))
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        if not profile:
            label = QLabel("暂无候选")
            label.setWordWrap(True)
            layout.addWidget(label)
            return box

        score_label = QLabel(f"score = {float(profile.get('score', 0.0)):.4f}")
        score_label.setProperty("class", "titleSmall")
        layout.addWidget(score_label)

        if key == self.result.get("recommended_profile", "balanced"):
            recommended_label = QLabel("当前推荐")
            recommended_label.setProperty("class", "statusInfo")
            layout.addWidget(recommended_label)

        stage_label = QLabel(f"来源阶段: {profile.get('stage', '-')}")
        stage_label.setProperty("class", "textSecondary")
        layout.addWidget(stage_label)

        params_label = QLabel(
            f"参数: {json.dumps(profile.get('params', {}), ensure_ascii=False)}"
        )
        params_label.setWordWrap(True)
        layout.addWidget(params_label)

        metrics = profile.get("metrics", {}) or {}
        metric_parts = []
        for metric_key, value in list(metrics.items())[:3]:
            if isinstance(value, (int, float)):
                metric_parts.append(f"{metric_key}={value:.4f}")
            else:
                metric_parts.append(f"{metric_key}={value}")
        if metric_parts:
            metric_label = QLabel("指标: " + ", ".join(metric_parts))
            metric_label.setWordWrap(True)
            layout.addWidget(metric_label)

        reason_label = QLabel(str(profile.get("reason", "")))
        reason_label.setWordWrap(True)
        reason_label.setProperty("class", "textSecondary")
        layout.addWidget(reason_label)
        return box

    def _get_plot_palette(self) -> dict:
        """获取图表主题配色。"""
        theme = get_theme_manager().get_current_theme()
        if theme == "dark":
            return {
                "theme": "dark",
                "fig_face": "#1f2125",
                "ax_face": "#23252a",
                "text": "#e8e8e8",
                "hint": "#b7bcc6",
                "spine": "#5a606b",
                "grid": "#4b515c",
                "legend_face": "#2a2d33",
                "legend_edge": "#434852",
                "primary": "#7ab8ff",
                "accent": "#58c4ff",
                "success": "#6dd7a3",
                "warning": "#f4bf4f",
                "error": "#ff8f8f",
                "edge": "#6b7280",
            }
        return {
            "theme": "light",
            "fig_face": "#ffffff",
            "ax_face": "#f8f8f8",
            "text": "#333333",
            "hint": "#666666",
            "spine": "#bbbbbb",
            "grid": "#d9dee7",
            "legend_face": "#ffffff",
            "legend_edge": "#d1d5db",
            "primary": "#0ea5e9",
            "accent": "#0284c7",
            "success": "#10b981",
            "warning": "#f59e0b",
            "error": "#dc2626",
            "edge": "#333333",
        }

    def _apply_axes_theme(self, ax, palette: dict, *, grid: bool = True):
        ax.set_facecolor(palette["ax_face"])
        ax.tick_params(colors=palette["text"])
        ax.xaxis.label.set_color(palette["text"])
        ax.yaxis.label.set_color(palette["text"])
        ax.title.set_color(palette["text"])
        for spine in ax.spines.values():
            spine.set_color(palette["spine"])
        ax.grid(grid, alpha=0.25, color=palette["grid"])
        legend = ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor(palette["legend_face"])
            frame.set_edgecolor(palette["legend_edge"])
            frame.set_alpha(0.9)
            for text in legend.get_texts():
                text.set_color(palette["text"])

    def _style_figure(self):
        palette = self._get_plot_palette()
        self.fig.patch.set_facecolor(palette["fig_face"])
        for ax in self.fig.axes:
            self._apply_axes_theme(ax, palette)
        return palette

    def _style_colorbar(self, colorbar, palette: dict):
        colorbar.ax.tick_params(colors=palette["text"])
        colorbar.outline.set_edgecolor(palette["spine"])
        colorbar.ax.yaxis.label.set_color(palette["text"])
        colorbar.ax.set_facecolor(palette["ax_face"])

    def _plot_visuals(self):
        """Draw visual comparison charts for candidate trials."""
        self.fig.clear()
        palette = self._get_plot_palette()
        if not self.sorted_trials:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "暂无候选评分图表", ha="center", va="center", color=palette["hint"])
            ax.set_axis_off()
            self._style_figure()
            self.canvas.draw()
            return

        family = self.result.get("family", "")
        if family == "background":
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            self._plot_background_score_curve(ax1)
            self._plot_background_tradeoff(ax2)
        elif family == "gain":
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            plotted_heatmap = self._plot_gain_heatmap(ax1)
            if not plotted_heatmap:
                self._plot_top_scores(ax1, title="候选得分排名")
            self._plot_gain_tradeoff(ax2)
        else:
            ax = self.fig.add_subplot(111)
            self._plot_top_scores(ax, title="候选得分排名")

        self._style_figure()
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_top_scores(self, ax, title: str):
        top_trials = self.sorted_trials[:8]
        labels = [self._trial_label(trial) for trial in top_trials][::-1]
        scores = [float(trial.get("score", 0.0)) for trial in top_trials][::-1]
        ax.barh(labels, scores, color=self._get_plot_palette()["primary"])
        ax.set_title(title)
        ax.set_xlabel("总评分")
        ax.grid(True, axis="x", alpha=0.2)

    def _plot_background_score_curve(self, ax):
        name, values = self._extract_primary_numeric_param(self.sorted_trials)
        scores = np.array(
            [float(trial.get("score", 0.0)) for trial in self.sorted_trials]
        )
        if name and values is not None and len(values) == len(scores):
            order = np.argsort(values)
            x = values[order]
            y = scores[order]
            palette = self._get_plot_palette()
            ax.plot(x, y, marker="o", linewidth=1.8, color=palette["accent"])
            best_idx = int(np.argmax(y))
            ax.scatter([x[best_idx]], [y[best_idx]], color=palette["error"], s=70, zorder=3)
            ax.annotate(
                "最优",
                (x[best_idx], y[best_idx]),
                textcoords="offset points",
                xytext=(6, 6),
                color=palette["text"],
            )
            ax.set_xlabel(name)
        else:
            x = np.arange(1, len(scores) + 1)
            ax.plot(x, scores, marker="o", linewidth=1.8, color=self._get_plot_palette()["accent"])
            ax.set_xlabel("候选序号")
        ax.set_ylabel("总评分")
        ax.set_title("背景抑制候选得分曲线")
        ax.grid(True, alpha=0.25)

    def _plot_background_tradeoff(self, ax):
        trials = self.sorted_trials
        name, values = self._extract_primary_numeric_param(trials)
        coherence = np.array(
            [
                float(trial.get("metrics", {}).get("horizontal_coherence", 0.0))
                for trial in trials
            ]
        )
        saliency = np.array(
            [
                float(trial.get("metrics", {}).get("local_saliency_preservation", 0.0))
                for trial in trials
            ]
        )
        edge = np.array(
            [
                float(trial.get("metrics", {}).get("edge_preservation", 0.0))
                for trial in trials
            ]
        )
        coherence_better = 1.0 - self._normalize_series(coherence)
        saliency_norm = self._normalize_series(saliency)
        edge_norm = self._normalize_series(edge)

        if name and values is not None and len(values) == len(trials):
            order = np.argsort(values)
            x = values[order]
            ax.plot(x, coherence_better[order], marker="o", label="背景残留抑制")
            ax.plot(x, saliency_norm[order], marker="s", label="显著结构保留")
            ax.plot(x, edge_norm[order], marker="^", label="边缘保留")
            ax.set_xlabel(name)
        else:
            x = np.arange(1, len(trials) + 1)
            ax.plot(x, coherence_better, marker="o", label="背景残留抑制")
            ax.plot(x, saliency_norm, marker="s", label="显著结构保留")
            ax.plot(x, edge_norm, marker="^", label="边缘保留")
            ax.set_xlabel("候选序号")
        ax.set_ylabel("归一化指标（越高越好）")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("背景抑制 trade-off")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    def _plot_gain_heatmap(self, ax) -> bool:
        trials = self.sorted_trials
        gain_max_values = sorted(
            {
                float(trial.get("params", {}).get("gain_max"))
                for trial in trials
                if "gain_max" in trial.get("params", {})
            }
        )
        power_values = sorted(
            {
                float(trial.get("params", {}).get("power"))
                for trial in trials
                if "power" in trial.get("params", {})
            }
        )
        if len(gain_max_values) < 2 or len(power_values) < 2:
            return False

        heatmap = np.full(
            (len(power_values), len(gain_max_values)), np.nan, dtype=float
        )
        for trial in trials:
            params = trial.get("params", {})
            if "gain_max" not in params or "power" not in params:
                continue
            x = gain_max_values.index(float(params["gain_max"]))
            y = power_values.index(float(params["power"]))
            heatmap[y, x] = float(trial.get("score", 0.0))

        palette = self._get_plot_palette()
        im = ax.imshow(heatmap, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(np.arange(len(gain_max_values)))
        ax.set_xticklabels([f"{v:g}" for v in gain_max_values])
        ax.set_yticks(np.arange(len(power_values)))
        ax.set_yticklabels([f"{v:g}" for v in power_values])
        ax.set_xlabel("gain_max")
        ax.set_ylabel("power")
        ax.set_title("增益候选评分热力图")
        cbar = self.fig.colorbar(im, ax=ax, shrink=0.85)
        self._style_colorbar(cbar, palette)

        best = self.sorted_trials[0]
        best_params = best.get("params", {})
        if "gain_max" in best_params and "power" in best_params:
            bx = gain_max_values.index(float(best_params["gain_max"]))
            by = power_values.index(float(best_params["power"]))
            ax.scatter(
                [bx], [by], s=140, facecolors="none", edgecolors=palette["error"], linewidths=2
            )
        return True

    def _plot_gain_tradeoff(self, ax):
        trials = self.sorted_trials
        deep_gain = np.array(
            [
                float(trial.get("metrics", {}).get("deep_gain_ratio", 0.0))
                for trial in trials
            ]
        )
        clip = np.array(
            [
                float(trial.get("metrics", {}).get("clipping_ratio", 0.0))
                for trial in trials
            ]
        )
        hot = np.array(
            [
                float(trial.get("metrics", {}).get("hot_pixel_ratio", 0.0))
                for trial in trials
            ]
        )
        scores = np.array([float(trial.get("score", 0.0)) for trial in trials])
        size = 90.0 + 280.0 * (1.0 - np.clip(self._normalize_series(hot), 0.0, 1.0))

        palette = self._get_plot_palette()
        scatter = ax.scatter(
            deep_gain,
            clip,
            c=scores,
            s=size,
            cmap="plasma",
            alpha=0.85,
            edgecolors=palette["edge"],
            linewidths=0.5,
        )
        ax.set_xlabel("深部可见度提升")
        ax.set_ylabel("近饱和比例")
        ax.invert_yaxis()
        ax.set_title("增益提升 / 过曝 trade-off")
        ax.grid(True, alpha=0.25)
        cbar = self.fig.colorbar(scatter, ax=ax, shrink=0.85)
        self._style_colorbar(cbar, palette)

        for idx, trial in enumerate(trials[:3]):
            ax.annotate(
                f"#{idx + 1} {self._trial_label(trial)}",
                (deep_gain[idx], clip[idx]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color=palette["text"],
            )

    def _extract_primary_numeric_param(self, trials):
        if not trials:
            return None, None
        keys = []
        for key in trials[0].get("params", {}).keys():
            if str(key).startswith("_"):
                continue
            if all(
                isinstance(trial.get("params", {}).get(key), (int, float))
                for trial in trials
            ):
                keys.append(key)
        if not keys:
            return None, None
        key = keys[0]
        values = np.asarray(
            [float(trial.get("params", {}).get(key)) for trial in trials], dtype=float
        )
        return key, values

    def _normalize_series(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1.0e-12:
            return np.ones_like(arr)
        return (arr - vmin) / (vmax - vmin)

    def _trial_label(self, trial: dict) -> str:
        params = trial.get("params", {})
        parts = []
        for key, value in params.items():
            if str(key).startswith("_"):
                continue
            if isinstance(value, float):
                parts.append(f"{key}={value:g}")
            else:
                parts.append(f"{key}={value}")
            if len(parts) == 2:
                break
        return ", ".join(parts) if parts else "候选"
