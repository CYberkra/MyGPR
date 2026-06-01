# -*- coding: utf-8 -*-
"""Processing-lineage stepper and display override controller for MyGPR.

This controller owns the lightweight B-scan processing-chain UI under the main
plot and the temporary history-step preview state.  It intentionally keeps the
first V0.8 extraction conservative: it reads and writes the existing host-window
attributes so the rest of the GUI can keep using the historical ``GPRGuiQt``
method names through thin compatibility wrappers.
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QWidget


class ProcessingLineageController:
    """Manage processing-chain chips, temporary history preview, and lineage text."""

    def __init__(self, host):
        self.host = host

    def create_stepper_bar(self) -> QFrame:
        """Create a compact, clickable processing-lineage stepper under the B-scan."""
        host = self.host
        bar = QFrame()
        bar.setObjectName("ProcessingStepperBar")
        bar.setMinimumHeight(40)
        bar.setMaximumHeight(48)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(7, 4, 7, 4)
        layout.setSpacing(6)

        title = QLabel("链路")
        title.setObjectName("ProcessingStepperTitle")
        title.setToolTip("当前 B-scan 的处理链路；点击步骤可临时查看历史结果。")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setObjectName("ProcessingStepperScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(32)
        scroll.setMaximumHeight(38)

        host_widget = QWidget()
        host_widget.setObjectName("ProcessingStepperHost")
        row = QHBoxLayout(host_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        scroll.setWidget(host_widget)

        host._plot_stepper_scroll = scroll
        host._plot_stepper_host = host_widget
        host._plot_stepper_layout = row
        host._lineage_step_buttons = []
        layout.addWidget(scroll, 1)
        return bar

    def compact_step_label(self, label: str, index: int = 0) -> str:
        """Return a short label for processing-lineage chips."""
        label = str(label or "").strip() or f"Step {index + 1}"
        aliases = {
            "原始数据": "Raw",
            "原始": "Raw",
            "raw": "Raw",
            "Raw": "Raw",
            "当前结果": "当前",
        }
        label = aliases.get(label, label)
        if "+" in label and "步" in label:
            return label
        max_len = 14 if index else 10
        if len(label) > max_len:
            return label[: max_len - 1] + "…"
        return label

    def step_tooltip(self, entry: dict, index: int, total: int) -> str:
        """Build tooltip for one processing-lineage step."""
        label = str(entry.get("label") or f"Step {index + 1}")
        data = entry.get("data")
        shape = getattr(data, "shape", None)
        header = entry.get("header_info") or {}
        lines = [f"步骤 {index + 1}/{total}: {label}"]
        if shape:
            lines.append(
                f"尺寸: {shape[0]} × {shape[1]}" if len(shape) >= 2 else f"尺寸: {shape}"
            )
        method_key = header.get("method_key") or header.get("display_method_key")
        if method_key:
            lines.append(f"方法键: {method_key}")
        display_title = header.get("display_title")
        if display_title and display_title != label:
            lines.append(f"显示标题: {display_title}")
        if index == total - 1:
            lines.append("点击：返回当前正式结果。")
        else:
            lines.append("点击：临时查看该历史步骤；不会修改当前数据。")
        return "\n".join(lines)

    def _history_entries(self) -> list[dict]:
        try:
            entries = self.host.shared_data.build_result_history_entries()
        except Exception:
            entries = []
        return entries or []

    def sync_stepper(self) -> None:
        """Refresh the clickable processing-lineage stepper."""
        host = self.host
        layout = getattr(host, "_plot_stepper_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        host._lineage_step_buttons = []

        entries = self._history_entries()
        if not entries:
            entries = [{"label": "Raw", "data": None, "header_info": {}}]

        total = len(entries)
        current_index = getattr(host, "_lineage_view_index", None)
        if current_index is None or current_index >= total:
            current_index = total - 1
            host._lineage_view_index = None
        if current_index < 0:
            current_index = total - 1
            host._lineage_view_index = None

        for idx, entry in enumerate(entries):
            label = self.compact_step_label(str(entry.get("label") or ""), idx)
            btn = QPushButton(label)
            btn.setObjectName("ProcessingStepChip")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(False)
            btn.setMinimumHeight(26)
            btn.setMaximumHeight(30)
            btn.setProperty("active", "true" if idx == current_index else "false")
            btn.setProperty("current", "true" if idx == total - 1 else "false")
            btn.setToolTip(self.step_tooltip(entry, idx, total))
            btn.clicked.connect(lambda _checked=False, step_idx=idx: self.on_step_clicked(step_idx))
            layout.addWidget(btn)
            host._lineage_step_buttons.append(btn)
            if idx < total - 1:
                arrow = QLabel("→")
                arrow.setObjectName("ProcessingStepArrow")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(arrow)
        layout.addStretch(1)

    def on_step_clicked(self, index: int) -> None:
        """Temporarily switch the main B-scan view to a processing-lineage step."""
        host = self.host
        entries = self._history_entries()
        if not entries:
            return
        index = max(0, min(int(index), len(entries) - 1))
        entry = entries[index]
        label = str(entry.get("label") or f"Step {index + 1}")
        if index == len(entries) - 1:
            host._lineage_view_index = None
            self.clear_display_override()
            host._set_runtime_summary("状态：当前正式结果", "neutral")
        else:
            data = entry.get("data")
            if data is None:
                return
            header = dict(entry.get("header_info") or {})
            header["display_title"] = f"历史步骤：{label}"
            header["display_lineage_step"] = label
            host._lineage_view_index = index
            self.set_display_override(
                np.asarray(data),
                header_info=header,
                trace_metadata=entry.get("trace_metadata"),
            )
            host._set_runtime_summary(f"状态：临时查看 · {label}", "info")
        self.sync_stepper()
        host._last_plot_signature = None
        host._refresh_plot()
        self.update_display()

    def set_display_override(
        self,
        data: np.ndarray | None,
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
    ) -> None:
        """Set temporary display-only payload for the main plot."""
        host = self.host
        host._display_data_override = (
            np.array(data, copy=False) if data is not None else None
        )
        host._display_header_info_override = dict(header_info or {}) or None
        host._display_trace_metadata_override = trace_metadata
        host._display_override_revision += 1

    def clear_display_override(self) -> None:
        """Clear temporary main-plot payload and return to the formal current data."""
        host = self.host
        had_override = host._display_data_override is not None
        host._display_data_override = None
        host._display_header_info_override = None
        host._display_trace_metadata_override = None
        host._lineage_view_index = None
        if had_override:
            host._display_override_revision += 1

    def get_active_plot_payload(
        self, fallback_data: np.ndarray | None = None
    ) -> tuple[np.ndarray | None, dict | None, dict | None]:
        """Return the data/header/metadata payload that should currently be plotted."""
        host = self.host
        if host._display_data_override is not None:
            return (
                host._display_data_override,
                host._display_header_info_override or host.header_info,
                host._display_trace_metadata_override,
            )
        if host._is_single_view_mode():
            snapshot = host._get_selected_single_view_snapshot()
            if snapshot is not None and snapshot.get("data") is not None:
                return (
                    np.array(snapshot["data"], copy=False),
                    snapshot.get("header_info") or host.header_info,
                    snapshot.get("trace_metadata"),
                )
        return (
            fallback_data if fallback_data is not None else host.data,
            host.header_info,
            host.trace_metadata,
        )

    def build_steps(self) -> list[str]:
        """Build compact processing lineage from formal shared history labels."""
        entries = self._history_entries()
        if not entries:
            return ["Raw"]
        labels: list[str] = []
        for item in entries:
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            if label in {"原始数据", "原始", "Raw", "raw"}:
                norm = "Raw"
            else:
                norm = label
            if not labels or labels[-1] != norm:
                labels.append(norm)
        return labels or ["Raw"]

    def build_text(self) -> str:
        """Return compact lineage string for toolbar/title."""
        steps = self.build_steps()
        if len(steps) <= 1:
            return "Raw"
        return " -> ".join(steps)

    def build_tooltip(self) -> str:
        """Return detailed lineage tooltip text."""
        host = self.host
        steps = self.build_steps()
        lines = [f"数据源: {host.data_path or '未加载'}", "处理链路:"]
        for idx, step in enumerate(steps, start=1):
            lines.append(f"{idx}. {step}")
        lines.append("说明: 视图交互（平移/缩放/滑动）不写入处理链路。")
        return "\n".join(lines)

    def update_display(self) -> None:
        """Sync main toolbar lineage summary, tooltip and stepper."""
        host = self.host
        if hasattr(host, "_plot_lineage_label") and host._plot_lineage_label is not None:
            lineage_text = self.build_text()
            if host._lineage_view_index is not None:
                try:
                    entries = self._history_entries()
                    active_label = str(entries[host._lineage_view_index].get("label") or "历史步骤")
                except Exception:
                    active_label = "历史步骤"
                host._plot_lineage_label.setText(
                    f"处理链路: {lineage_text} · 正在查看 {active_label}"
                )
            else:
                host._plot_lineage_label.setText(f"处理链路: {lineage_text}")
            host._plot_lineage_label.setToolTip(self.build_tooltip())
        self.sync_stepper()
