# -*- coding: utf-8 -*-
"""Processing-lineage stepper and display override controller for MyGPR.

The controller owns the compact B-scan processing-chain UI under the main plot,
temporary history-step previews, lightweight lineage comparison helpers, and
human-readable lineage export metadata.  It deliberately remains a thin GUI-side
controller: it does not mutate the formal processing graph or recompute data.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QToolButton,
)


class ProcessingLineageController:
    """Manage processing-chain chips, history preview, compare, and export metadata."""

    def __init__(self, host):
        self.host = host
        self._last_selected_index: int | None = None
        self._compare_selected_indices: set[int] = set()
        self._compare_mode: str | None = None
        self._last_stepper_signature: tuple | None = None


    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def create_stepper_bar(self) -> QFrame:
        """Create a single-row processing-lineage strip under the B-scan.

        V0.8.15 compacts the former two-row stepper + compare tray into one
        horizontal strip to recover vertical B-scan space and avoid crowding in
        the lower-left workspace.  The detailed step text is kept in tooltips.
        """
        host = self.host
        bar = QFrame()
        bar.setObjectName("ProcessingStepperBar")
        bar.setMinimumHeight(48)
        bar.setMaximumHeight(60)
        outer = QHBoxLayout(bar)
        outer.setContentsMargins(5, 2, 5, 2)
        outer.setSpacing(4)

        title = QLabel("链路")
        title.setObjectName("ProcessingStepperTitle")
        title.setMaximumWidth(34)
        title.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        title.setToolTip("当前 B-scan 的处理链路；点击步骤临时查看历史结果，点小圆点加入对比。")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setObjectName("ProcessingStepperScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(38)
        scroll.setMaximumHeight(44)

        host_widget = QWidget()
        host_widget.setObjectName("ProcessingStepperHost")
        row = QHBoxLayout(host_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        scroll.setWidget(host_widget)

        host._plot_stepper_scroll = scroll
        host._plot_stepper_host = host_widget
        host._plot_stepper_layout = row
        host._lineage_step_buttons = []
        host._lineage_step_select_buttons = []
        outer.addWidget(scroll, 1)

        # Kept for compatibility with update_step_detail(); hidden to save space.
        detail = QLabel("选择步骤查看详情")
        detail.setObjectName("ProcessingStepDetail")
        detail.setVisible(False)
        detail.setToolTip("显示当前链路步骤的参数、尺寸、warning 与状态。")
        host._lineage_step_detail_label = detail

        tray = QFrame()
        tray.setObjectName("ProcessingCompareTray")
        tray_layout = QHBoxLayout(tray)
        tray_layout.setContentsMargins(0, 0, 0, 0)
        tray_layout.setSpacing(3)
        tray.setMinimumHeight(28)
        tray.setMaximumHeight(32)

        tray.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

        tray_label = QLabel("对比 0/4")
        tray_label.setObjectName("ProcessingCompareTrayLabel")
        tray_label.setMinimumWidth(58)
        tray_label.setMaximumWidth(118)
        tray_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        tray_layout.addWidget(tray_label)
        host._lineage_compare_tray_label = tray_label

        menu_btn = QToolButton()
        menu_btn.setObjectName("ProcessingStepperMenuAction")
        menu_btn.setText("对比")
        menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        menu_btn.setMinimumSize(52, 28)
        menu_btn.setMaximumSize(68, 32)
        compare_menu = QMenu(menu_btn)
        act_slider = compare_menu.addAction("滑动对比")
        act_grid = compare_menu.addAction("网格对比")
        act_diff = compare_menu.addAction("差值图")
        compare_menu.addSeparator()
        act_clear = compare_menu.addAction("清空选择")
        act_slider.triggered.connect(lambda: self.apply_compare_mode("slider"))
        act_grid.triggered.connect(lambda: self.apply_compare_mode("grid"))
        act_diff.triggered.connect(lambda: self.apply_compare_mode("diff"))
        act_clear.triggered.connect(self.clear_compare_selection)
        menu_btn.setMenu(compare_menu)
        tray_layout.addWidget(menu_btn)
        host._lineage_compare_menu_button = menu_btn
        host._lineage_compare_actions = {
            "slider": act_slider,
            "grid": act_grid,
            "diff": act_diff,
            "clear": act_clear,
        }

        slider_btn = QPushButton("滑动对比")
        slider_btn.setObjectName("ProcessingStepperAction")
        slider_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        slider_btn.setMinimumSize(66, 28)
        slider_btn.setMaximumSize(82, 32)
        slider_btn.setCheckable(True)
        slider_btn.setEnabled(False)
        slider_btn.setVisible(False)
        slider_btn.setToolTip("滑动对比：选择恰好两个步骤后启用；再次点击取消。")
        slider_btn.toggled.connect(lambda checked: self.toggle_compare_mode("slider", checked))
        tray_layout.addWidget(slider_btn)
        host._lineage_slider_compare_button = slider_btn

        grid_btn = QPushButton("网格对比")
        grid_btn.setObjectName("ProcessingStepperAction")
        grid_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        grid_btn.setMinimumSize(66, 28)
        grid_btn.setMaximumSize(82, 32)
        grid_btn.setCheckable(True)
        grid_btn.setEnabled(False)
        grid_btn.setVisible(False)
        grid_btn.setToolTip("网格对比：选择 2–4 个步骤后启用；共享色标多图查看。")
        grid_btn.toggled.connect(lambda checked: self.toggle_compare_mode("grid", checked))
        tray_layout.addWidget(grid_btn)
        host._lineage_grid_compare_button = grid_btn

        diff_btn = QPushButton("差值图")
        diff_btn.setObjectName("ProcessingStepperAction")
        diff_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        diff_btn.setMinimumSize(56, 28)
        diff_btn.setMaximumSize(70, 32)
        diff_btn.setCheckable(True)
        diff_btn.setEnabled(False)
        diff_btn.setVisible(False)
        diff_btn.setToolTip("差值图：选择恰好两个步骤后启用；显示 |A - B|。")
        diff_btn.toggled.connect(lambda checked: self.toggle_compare_mode("diff", checked))
        tray_layout.addWidget(diff_btn)
        host._lineage_diff_compare_button = diff_btn

        clear_btn = QPushButton("清空")
        clear_btn.setObjectName("ProcessingStepperAction")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setMinimumSize(48, 28)
        clear_btn.setMaximumSize(60, 32)
        clear_btn.setEnabled(False)
        clear_btn.setVisible(False)
        clear_btn.setToolTip("清空对比篮选择并回到单图。")
        clear_btn.clicked.connect(self.clear_compare_selection)
        tray_layout.addWidget(clear_btn)
        host._lineage_clear_compare_button = clear_btn

        outer.addWidget(tray)
        host._lineage_compare_tray = tray
        self._update_compare_tray()
        return bar

    # ------------------------------------------------------------------
    # Entry helpers
    # ------------------------------------------------------------------

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

    def _history_entries(self) -> list[dict[str, Any]]:
        try:
            entries = self.host.shared_data.build_result_history_entries()
        except Exception:
            logger.warning("Failed to build history entries", exc_info=True)
            entries = []
        return entries or []

    def _history_memory_summary(self) -> dict[str, Any]:
        try:
            return self.host.shared_data.get_history_memory_summary() or {}
        except Exception:
            logger.warning("Failed to get history memory summary", exc_info=True)
            return {}

    def _entry_shape(self, entry: dict[str, Any]) -> list[int]:
        data = entry.get("data")
        shape = getattr(data, "shape", None)
        if shape:
            return [int(dim) for dim in shape]
        summary = entry.get("summary") or entry.get("array") or {}
        shape = summary.get("shape") if isinstance(summary, dict) else None
        return [int(dim) for dim in shape] if shape else []

    def _entry_shape_text(self, entry: dict[str, Any]) -> str:
        shape = self._entry_shape(entry)
        if len(shape) >= 2:
            return f"{shape[0]} × {shape[1]}"
        return " × ".join(str(v) for v in shape) if shape else "未知尺寸"

    def _entry_header(self, entry: dict[str, Any]) -> dict[str, Any]:
        header = entry.get("header_info") or entry.get("header_summary") or {}
        return header if isinstance(header, dict) else {}

    def _entry_method_key(self, entry: dict[str, Any]) -> str:
        header = self._entry_header(entry)
        return str(header.get("method_key") or header.get("display_method_key") or "")

    def _entry_params(self, entry: dict[str, Any]) -> dict[str, Any]:
        header = self._entry_header(entry)
        for key in ("params", "method_params", "mapped_params", "parameters"):
            value = header.get(key)
            if isinstance(value, dict):
                return dict(value)
        return {}

    def _entry_elapsed_ms(self, entry: dict[str, Any]) -> float | None:
        header = self._entry_header(entry)
        for key in ("elapsed_ms", "runtime_ms", "processing_elapsed_ms"):
            value = header.get(key)
            try:
                return float(value) if value is not None else None
            except Exception:
                logger.warning("Failed to parse elapsed_ms value", exc_info=True)
                continue
        return None

    def _entry_warnings(self, entry: dict[str, Any]) -> list[str]:
        header = self._entry_header(entry)
        raw_values: list[Any] = []
        for key in ("runtime_warnings", "warnings", "warning"):
            value = header.get(key)
            if not value:
                continue
            if isinstance(value, list):
                raw_values.extend(value)
            else:
                raw_values.append(value)
        warnings: list[str] = []
        for item in raw_values:
            if isinstance(item, dict):
                msg = item.get("message") or item.get("reason") or item.get("code") or str(item)
            else:
                msg = str(item)
            if msg and msg not in warnings:
                warnings.append(msg)
        return warnings

    def _step_status(self, entry: dict[str, Any], index: int, total: int, active_index: int) -> str:
        if bool(entry.get("pruned")):
            return "pruned"
        if self._entry_warnings(entry):
            if index == active_index and index != total - 1:
                return "viewing_warning"
            return "warning"
        if index == 0 and self.compact_step_label(entry.get("label") or "", index) == "Raw":
            return "raw"
        if index == active_index and index != total - 1:
            return "viewing"
        if index == total - 1:
            return "current"
        return "applied"

    def _step_text(self, entry: dict[str, Any], index: int, total: int, active_index: int) -> str:
        label = self.compact_step_label(str(entry.get("label") or ""), index)
        status = self._step_status(entry, index, total, active_index)
        if status == "current":
            return f"当前 {label}"
        if status == "viewing":
            return f"查看中 {label}"
        if status == "viewing_warning":
            return f"查看中 ⚠ {label}"
        if status == "warning":
            return f"⚠ {label}"
        if status == "applied":
            return f"✓ {label}"
        if status == "pruned":
            return f"裁剪 {label}"
        return label

    def _format_params_inline(self, params: dict[str, Any], *, limit: int = 4) -> str:
        if not params:
            return "无"
        parts: list[str] = []
        for idx, (key, value) in enumerate(params.items()):
            if idx >= limit:
                parts.append("…")
                break
            try:
                if isinstance(value, float):
                    text = f"{value:.4g}"
                else:
                    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else str(value)
            except Exception:
                logger.warning("Failed to format param value", exc_info=True)
                text = str(value)
            parts.append(f"{key}={text}")
        return ", ".join(parts)

    def step_tooltip(self, entry: dict[str, Any], index: int, total: int) -> str:
        """Build tooltip for one processing-lineage step."""
        label = str(entry.get("label") or f"Step {index + 1}")
        lines = [f"步骤 {index + 1}/{total}: {label}"]
        lines.append(f"状态: {self._status_text_for_entry(entry, index, total)}")
        lines.append(f"尺寸: {self._entry_shape_text(entry)}")
        method_key = self._entry_method_key(entry)
        if method_key:
            lines.append(f"方法键: {method_key}")
        params = self._entry_params(entry)
        if params:
            lines.append(f"参数: {self._format_params_inline(params, limit=8)}")
        elapsed_ms = self._entry_elapsed_ms(entry)
        if elapsed_ms is not None:
            lines.append(f"运行: {elapsed_ms:.1f} ms")
        warnings = self._entry_warnings(entry)
        if warnings:
            lines.append("Warning: " + "；".join(warnings[:3]))
        if bool(entry.get("pruned")):
            lines.append("该步骤完整数组已被内存策略裁剪，只保留摘要。")
        elif index == total - 1:
            lines.append("点击：返回当前正式结果。")
        else:
            lines.append("点击：临时查看该历史步骤；不会修改当前数据。")
        return "\n".join(lines)

    def _status_text_for_entry(self, entry: dict[str, Any], index: int, total: int) -> str:
        if bool(entry.get("pruned")):
            return "裁剪 / summary-only"
        if index == 0 and self.compact_step_label(entry.get("label") or "", index) == "Raw":
            return "Raw 原始输入"
        if index == total - 1:
            return "当前正式结果"
        if self._entry_warnings(entry):
            return "已应用 / 有 warning"
        return "已成功应用"

    # ------------------------------------------------------------------
    # Stepper sync and step actions
    # ------------------------------------------------------------------

    def _normalize_selected_indices(self, total: int) -> None:
        """Drop stale/pruned selections and enforce the lightweight compare limit."""
        entries = self._history_entries()
        valid: list[int] = []
        for idx in sorted(int(i) for i in self._compare_selected_indices):
            if idx < 0 or idx >= total:
                continue
            entry = entries[idx] if idx < len(entries) else {}
            if entry.get("data") is None or bool(entry.get("pruned")):
                continue
            valid.append(idx)
        self._compare_selected_indices = set(valid[:4])
        if self._compare_mode and not self._is_compare_mode_valid(self._compare_mode):
            self.deactivate_compare_mode(silent=True)

    def _entry_compare_label(self, entry: dict[str, Any], index: int, total: int | None = None) -> str:
        label = str(entry.get("label") or f"Step {index + 1}").strip()
        total = total if total is not None else len(self._history_entries())
        if index == 0:
            return "Raw"
        if total and index == total - 1:
            return "当前"
        return self.compact_step_label(label, index)

    def _selected_compare_indices_sorted(self) -> list[int]:
        entries = self._history_entries()
        total = len(entries)
        self._normalize_selected_indices(total)
        return sorted(self._compare_selected_indices)

    def _selected_compare_names(self) -> list[str]:
        entries = self._history_entries()
        total = len(entries)
        names: list[str] = []
        for idx in self._selected_compare_indices_sorted():
            if 0 <= idx < total:
                names.append(self._entry_compare_label(entries[idx], idx, total))
        return names

    def _is_compare_mode_valid(self, mode: str | None) -> bool:
        count = len(self._compare_selected_indices)
        if mode in {"slider", "diff"}:
            return count == 2
        if mode == "grid":
            return 2 <= count <= 4
        return False

    def _mode_buttons(self) -> dict[str, Any]:
        host = self.host
        return {
            "slider": getattr(host, "_lineage_slider_compare_button", None),
            "grid": getattr(host, "_lineage_grid_compare_button", None),
            "diff": getattr(host, "_lineage_diff_compare_button", None),
        }

    def _set_mode_button_checked(self, mode: str | None, checked: bool) -> None:
        btn = self._mode_buttons().get(mode or "")
        if btn is None or not hasattr(btn, "setChecked"):
            return
        try:
            old = btn.blockSignals(True)
            btn.setChecked(bool(checked))
            btn.setProperty("active", "true" if checked else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.blockSignals(old)
        except Exception:
            logger.warning("Failed to set mode button checked state", exc_info=True)
            pass

    def _sync_mode_buttons_checked(self) -> None:
        for mode in ("slider", "grid", "diff"):
            self._set_mode_button_checked(mode, self._compare_mode == mode)

    def _set_advanced_mode_diff(self) -> None:
        page = getattr(self.host, "page_advanced", None)
        if page is None:
            return
        try:
            if hasattr(page, "mode_diff"):
                page.mode_diff.setChecked(True)
            else:
                if hasattr(page, "compare_var"):
                    page.compare_var.setChecked(True)
                if hasattr(page, "diff_var"):
                    page.diff_var.setChecked(True)
                if hasattr(page, "slider_compare_var"):
                    page.slider_compare_var.setChecked(False)
            if hasattr(page, "_refresh_compare_select_visibility"):
                page._refresh_compare_select_visibility()
        except Exception:
            logger.warning("Failed to set advanced mode diff", exc_info=True)
            pass

    def _set_advanced_mode_grid(self) -> None:
        page = getattr(self.host, "page_advanced", None)
        if page is None:
            return
        try:
            if hasattr(page, "mode_compare"):
                page.mode_compare.setChecked(True)
            else:
                if hasattr(page, "compare_var"):
                    page.compare_var.setChecked(True)
                if hasattr(page, "diff_var"):
                    page.diff_var.setChecked(False)
                if hasattr(page, "slider_compare_var"):
                    page.slider_compare_var.setChecked(False)
            if hasattr(page, "_refresh_compare_select_visibility"):
                page._refresh_compare_select_visibility()
        except Exception:
            logger.warning("Failed to set advanced mode grid", exc_info=True)
            pass

    def _sync_compare_combos(self, labels: list[str]) -> None:
        page = getattr(self.host, "page_advanced", None)
        if page is None or len(labels) < 2:
            return
        try:
            if hasattr(page, "compare_left_combo"):
                page.compare_left_combo.setCurrentText(labels[0])
            if hasattr(page, "compare_right_combo"):
                page.compare_right_combo.setCurrentText(labels[1])
        except Exception:
            logger.warning("Failed to sync compare combos", exc_info=True)
            pass

    def _compact_compare_names_text(self, names: list[str], *, max_chars: int = 24) -> str:
        """Return a short inline selection summary; full names stay in tooltip."""
        if not names:
            return ""
        text = " ↔ ".join(str(n) for n in names[:2])
        if len(names) > 2:
            text += f" +{len(names) - 2}"
        if len(text) > max_chars:
            return text[: max_chars - 1] + "…"
        return text

    def _update_compare_tray(self) -> None:
        host = self.host
        entries = self._history_entries()
        total = len(entries)
        self._normalize_selected_indices(total)
        selected = self._selected_compare_indices_sorted()
        names = self._selected_compare_names()
        tray = getattr(host, "_lineage_compare_tray", None)
        label = getattr(host, "_lineage_compare_tray_label", None)
        selectable_count = sum(
            1
            for entry in entries
            if bool(entry.get("data") is not None and not bool(entry.get("pruned")))
        )
        show_tray = bool(total > 1 and selectable_count > 1)
        if tray is not None:
            tray.setVisible(show_tray)
            tray.setToolTip("处理链路对比篮；选择至少两个历史步骤后显示对比操作。")
        if label is not None:
            if not names:
                label.setText("对比 0/4")
                label.setToolTip("点步骤右侧小圆点加入对比；再次点击可取消。最多选择 4 个步骤。")
            elif len(names) == 1:
                label.setText(f"1/4 · {self._compact_compare_names_text(names, max_chars=10)}")
                label.setToolTip("已选择：" + "、".join(names) + "。滑动/差值需要 2 个步骤；网格需要 2–4 个步骤。")
            else:
                label.setText(f"{len(names)}/4 · {self._compact_compare_names_text(names, max_chars=12)}")
                label.setToolTip("已选择：" + "、".join(names) + "。对比仅为 display-only，不改变正式处理结果。")
        buttons = self._mode_buttons()
        count = len(selected)
        # Visible compare actions now live in a compact overflow menu so the
        # processing-chain chips keep their horizontal space.  The legacy
        # QPushButtons stay instantiated for old tests/call-sites but remain hidden.
        for mode, btn in buttons.items():
            if btn is not None:
                btn.setVisible(False)
                btn.setEnabled(self._is_compare_mode_valid(mode))
        clear_btn = getattr(host, "_lineage_clear_compare_button", None)
        if clear_btn is not None:
            clear_btn.setVisible(False)
            clear_btn.setEnabled(count > 0)
        menu_btn = getattr(host, "_lineage_compare_menu_button", None)
        if menu_btn is not None:
            menu_btn.setVisible(bool(count > 0))
            menu_btn.setEnabled(bool(count > 0))
            menu_btn.setText(f"对比 {count}/4" if count else "对比")
            menu_btn.setToolTip(
                "选择链路步骤后在此打开滑动对比、网格对比、差值图或清空选择。"
                if count else "点步骤右侧小圆点加入对比篮。"
            )
        actions = getattr(host, "_lineage_compare_actions", {}) or {}
        if actions:
            if actions.get("slider") is not None:
                actions["slider"].setEnabled(count == 2)
            if actions.get("diff") is not None:
                actions["diff"].setEnabled(count == 2)
            if actions.get("grid") is not None:
                actions["grid"].setEnabled(2 <= count <= 4)
            if actions.get("clear") is not None:
                actions["clear"].setEnabled(count > 0)
        self._sync_mode_buttons_checked()

    def _stepper_signature(self, entries: list[dict[str, Any]], current_index: int) -> tuple:
        """Return a compact signature for the processing-lineage strip.

        The main plot may refresh frequently during compare/slider operations.
        Rebuilding every QPushButton in the lineage strip on each paint creates
        avoidable UI churn.  This signature lets sync_stepper skip the rebuild
        when the visible stepper state did not change.  It is display-only and
        does not affect processing arrays or report output.
        """
        total = len(entries)
        items: list[tuple] = []
        for idx, entry in enumerate(entries):
            items.append(
                (
                    str(entry.get("label") or ""),
                    tuple(self._entry_shape(entry)),
                    bool(entry.get("data") is not None),
                    bool(entry.get("pruned")),
                    bool(self._entry_warnings(entry)),
                    self._step_status(entry, idx, total, current_index),
                    bool(idx in self._compare_selected_indices),
                )
            )
        pruned_count = int((self._history_memory_summary().get("pruned_count") or 0))
        return (
            total,
            int(current_index),
            tuple(sorted(int(i) for i in self._compare_selected_indices)),
            str(self._compare_mode or ""),
            pruned_count,
            tuple(items),
        )

    def sync_stepper(self, *, force: bool = False) -> None:
        """Refresh the clickable processing-lineage stepper."""
        import time

        host = self.host
        layout = getattr(host, "_plot_stepper_layout", None)
        if layout is None:
            return

        start_ts = time.perf_counter()
        entries = self._history_entries()
        if not entries:
            entries = [{"label": "Raw", "data": None, "header_info": {}}]

        total = len(entries)
        self._normalize_selected_indices(total)
        current_index = getattr(host, "_lineage_view_index", None)
        if current_index is None or current_index >= total:
            current_index = total - 1
            host._lineage_view_index = None
        if current_index < 0:
            current_index = total - 1
            host._lineage_view_index = None

        signature = self._stepper_signature(entries, current_index)
        if (not force) and signature == self._last_stepper_signature and layout.count() > 0:
            self.update_step_detail(current_index)
            monitor = getattr(host, "_perf_monitor", None)
            if monitor is not None:
                monitor.record("display.lineage_stepper_skip_ms", (time.perf_counter() - start_ts) * 1000.0)
            return

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        host._lineage_step_buttons = []
        host._lineage_step_select_buttons = []

        for idx, entry in enumerate(entries):
            item = QFrame()
            item.setObjectName("ProcessingStepItem")
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(2)

            btn = QPushButton(self._step_text(entry, idx, total, current_index))
            btn.setObjectName("ProcessingStepChip")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(False)
            btn.setMinimumHeight(28)
            btn.setMaximumHeight(34)
            status = self._step_status(entry, idx, total, current_index)
            btn.setProperty("active", "true" if idx == current_index else "false")
            btn.setProperty("current", "true" if idx == total - 1 else "false")
            btn.setProperty("status", status)
            btn.setProperty("compareSelected", "true" if idx in self._compare_selected_indices else "false")
            btn.setToolTip(self.step_tooltip(entry, idx, total))
            btn.clicked.connect(lambda _checked=False, step_idx=idx: self.on_step_clicked(step_idx))
            item_layout.addWidget(btn)
            host._lineage_step_buttons.append(btn)

            selector = QPushButton("●" if idx in self._compare_selected_indices else "○")
            selector.setObjectName("ProcessingStepSelectDot")
            selector.setCursor(Qt.CursorShape.PointingHandCursor)
            selector.setCheckable(True)
            selector.setChecked(idx in self._compare_selected_indices)
            selector.setMinimumSize(20, 28)
            selector.setMaximumSize(22, 32)
            selector.setProperty("selected", "true" if idx in self._compare_selected_indices else "false")
            can_select = bool(entry.get("data") is not None and not bool(entry.get("pruned")))
            selector.setEnabled(can_select)
            selector.setToolTip(
                "加入/移出对比篮。单击步骤本体只临时查看；小圆点才负责选择对比对象。"
                if can_select
                else "该步骤没有完整 B-scan 数据，不能加入图像对比。"
            )
            selector.toggled.connect(lambda checked, step_idx=idx: self.on_compare_selector_toggled(step_idx, checked))
            item_layout.addWidget(selector)
            host._lineage_step_select_buttons.append(selector)
            layout.addWidget(item)
            if idx < total - 1:
                arrow = QLabel("→")
                arrow.setObjectName("ProcessingStepArrow")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(arrow)

        pruned_count = int((self._history_memory_summary().get("pruned_count") or 0))
        if pruned_count:
            if total:
                arrow = QLabel("→")
                arrow.setObjectName("ProcessingStepArrow")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(arrow)
            chip = QPushButton(f"裁剪 ×{pruned_count}")
            chip.setObjectName("ProcessingStepChip")
            chip.setEnabled(False)
            chip.setMinimumHeight(28)
            chip.setMaximumHeight(34)
            chip.setProperty("status", "pruned")
            chip.setToolTip("部分历史步骤因内存预算被裁剪；交付记录中保留摘要信息。")
            layout.addWidget(chip)

        layout.addStretch(1)
        self._last_stepper_signature = signature
        self.update_step_detail(current_index)
        self._update_compare_tray()
        monitor = getattr(host, "_perf_monitor", None)
        if monitor is not None:
            monitor.record("display.lineage_stepper_rebuild_ms", (time.perf_counter() - start_ts) * 1000.0)

    def on_step_clicked(self, index: int) -> None:
        """Temporarily switch the main B-scan view to a processing-lineage step."""
        host = self.host
        entries = self._history_entries()
        if not entries:
            return
        index = max(0, min(int(index), len(entries) - 1))
        entry = entries[index]
        label = str(entry.get("label") or f"Step {index + 1}")
        self._last_selected_index = index
        if self.is_lineage_compare_active():
            self.update_step_detail(index)
            host._set_runtime_summary("状态：当前处于链路对比；点小圆点调整对比篮，或再次点击对比模式退出", "info")
            return
        if index == len(entries) - 1:
            host._lineage_view_index = None
            self.clear_display_override()
            host._set_runtime_summary("状态：当前正式结果", "neutral")
        else:
            data = entry.get("data")
            if data is None:
                host._set_runtime_summary("状态：该步骤已裁剪，不能回看完整图像", "warning")
                self.update_step_detail(index)
                return
            header = dict(entry.get("header_info") or {})
            header["display_title"] = f"历史步骤：{label}"
            header["display_lineage_step"] = label
            header["display_only"] = True
            header["display_note"] = "临时查看历史步骤；不会修改当前正式结果。"
            host._lineage_view_index = index
            self.set_display_override(
                np.asarray(data),
                header_info=header,
                trace_metadata=entry.get("trace_metadata"),
            )
            host._set_runtime_summary(
                f"状态：临时查看 · {label}；不会修改当前结果", "info"
            )
        self.sync_stepper()
        host._last_plot_signature = None
        host._refresh_plot()
        self.update_display()

    def on_compare_selector_toggled(self, index: int, checked: bool) -> None:
        """Add or remove one processing step from the lightweight compare tray."""
        host = self.host
        entries = self._history_entries()
        total = len(entries)
        index = max(0, min(int(index), max(total - 1, 0)))
        if checked:
            entry = entries[index] if index < total else {}
            if entry.get("data") is None or bool(entry.get("pruned")):
                host._set_runtime_summary("状态：该步骤无完整图像，不能加入对比篮", "warning")
                self._compare_selected_indices.discard(index)
            elif len(self._compare_selected_indices) >= 4 and index not in self._compare_selected_indices:
                host._set_runtime_summary("状态：最多选择 4 个步骤进行可视对比", "warning")
                self._compare_selected_indices.discard(index)
            else:
                self._compare_selected_indices.add(index)
        else:
            self._compare_selected_indices.discard(index)
        self._normalize_selected_indices(total)
        if self._compare_mode and self._is_compare_mode_valid(self._compare_mode):
            self.apply_compare_mode(self._compare_mode, silent=True)
        elif self._compare_mode:
            self.deactivate_compare_mode(silent=True)
        self.sync_stepper()
        self._update_compare_tray()
        if not self._compare_mode:
            names = self._selected_compare_names()
            if names:
                host._set_runtime_summary(f"状态：对比篮已选择 {len(names)} 个步骤", "info")
            else:
                host._set_runtime_summary("状态：对比篮已清空", "neutral")

    def is_lineage_compare_active(self) -> bool:
        """Return whether the current compare mode was launched from the lineage stepper."""
        return bool(self._compare_mode)

    def get_active_compare_mode(self) -> str | None:
        """Return active display-only lineage compare mode: slider/grid/diff/None."""
        return self._compare_mode

    def get_selected_compare_snapshots(self) -> list[dict[str, Any]]:
        """Return selected compare snapshots in visual order."""
        entries = self._history_entries()
        total = len(entries)
        snapshots: list[dict[str, Any]] = []
        for idx in self._selected_compare_indices_sorted():
            if idx < 0 or idx >= total:
                continue
            entry = entries[idx]
            data = entry.get("data")
            if data is None:
                continue
            label = self._entry_compare_label(entry, idx, total)
            snapshots.append(
                {
                    "label": label,
                    "data": np.asarray(data),
                    "trace_metadata": entry.get("trace_metadata"),
                    "header_info": entry.get("header_info"),
                    "source": "processing_lineage",
                    "source_index": idx,
                    "display_only": True,
                }
            )
        return snapshots

    def _set_advanced_mode_single(self) -> None:
        """Return display controls to single-image mode without relying on hidden checkboxes only."""
        page = getattr(self.host, "page_advanced", None)
        if page is None:
            return
        try:
            if hasattr(page, "mode_single"):
                page.mode_single.setChecked(True)
            if hasattr(page, "compare_var"):
                page.compare_var.setChecked(False)
            if hasattr(page, "diff_var"):
                page.diff_var.setChecked(False)
            if hasattr(page, "slider_compare_var"):
                page.slider_compare_var.setChecked(False)
            if hasattr(page, "_refresh_compare_select_visibility"):
                page._refresh_compare_select_visibility()
        except Exception:
            logger.warning("Failed to set advanced mode single", exc_info=True)
            pass

    def _set_advanced_mode_slider(self) -> None:
        """Switch display controls to slider compare and keep radio/checkbox state coherent."""
        page = getattr(self.host, "page_advanced", None)
        if page is None:
            return
        try:
            if hasattr(page, "mode_slider"):
                page.mode_slider.setChecked(True)
            else:
                if hasattr(page, "compare_var"):
                    page.compare_var.setChecked(True)
                if hasattr(page, "diff_var"):
                    page.diff_var.setChecked(False)
                if hasattr(page, "slider_compare_var"):
                    page.slider_compare_var.setChecked(True)
            if hasattr(page, "_refresh_compare_select_visibility"):
                page._refresh_compare_select_visibility()
        except Exception:
            logger.warning("Failed to set advanced mode slider", exc_info=True)
            pass

    def update_step_detail(self, index: int | None = None) -> None:
        """Update the compact detail label and compare-tray state."""
        host = self.host
        detail = getattr(host, "_lineage_step_detail_label", None)
        entries = self._history_entries()
        if not entries:
            if detail is not None:
                detail.setText("未加载链路")
                detail.setToolTip("导入数据并应用方法后显示处理链路详情。")
            self._update_compare_tray()
            return
        if index is None:
            index = getattr(host, "_lineage_view_index", None)
            if index is None:
                index = len(entries) - 1
        index = max(0, min(int(index), len(entries) - 1))
        entry = entries[index]
        label = str(entry.get("label") or f"Step {index + 1}")
        warnings = self._entry_warnings(entry)
        status = self._status_text_for_entry(entry, index, len(entries))
        text = f"{index + 1}. {self.compact_step_label(label, index)} · {status} · {self._entry_shape_text(entry)}"
        if warnings:
            text += " · ⚠"
        if detail is not None:
            detail.setText(text)
            detail.setToolTip(self.step_detail_text(index))
        self._update_compare_tray()

    def step_detail_text(self, index: int | None = None) -> str:
        """Return multi-line step detail text for tooltips/report copying."""
        entries = self._history_entries()
        if not entries:
            return "未加载链路。"
        if index is None:
            index = getattr(self.host, "_lineage_view_index", None)
            if index is None:
                index = len(entries) - 1
        index = max(0, min(int(index), len(entries) - 1))
        entry = entries[index]
        label = str(entry.get("label") or f"Step {index + 1}")
        lines = [f"步骤 {index + 1}/{len(entries)}：{label}"]
        lines.append(f"状态：{self._status_text_for_entry(entry, index, len(entries))}")
        lines.append(f"输出尺寸：{self._entry_shape_text(entry)}")
        method_key = self._entry_method_key(entry)
        if method_key:
            lines.append(f"方法键：{method_key}")
        params = self._entry_params(entry)
        if params:
            lines.append("参数：")
            for key, value in params.items():
                lines.append(f"  {key} = {value}")
        elapsed_ms = self._entry_elapsed_ms(entry)
        if elapsed_ms is not None:
            lines.append(f"运行时间：{elapsed_ms:.1f} ms")
        warnings = self._entry_warnings(entry)
        lines.append("warning：" + ("；".join(warnings) if warnings else "无"))
        if index in self._compare_selected_indices:
            lines.append("对比：已加入对比篮；对比视图仅 display-only。")
        elif index < len(entries) - 1 and not bool(entry.get("pruned")):
            lines.append("说明：当前为可回看历史步骤；查看/对比不会修改当前正式结果。")
        return "\n".join(lines)

    def toggle_compare_mode(self, mode: str, checked: bool) -> None:
        """Toggle one compare visualization mode from the tray buttons."""
        if checked:
            self.apply_compare_mode(mode)
        else:
            if self._compare_mode == mode:
                self.deactivate_compare_mode()
            else:
                self._sync_mode_buttons_checked()

    def apply_compare_mode(self, mode: str, *, silent: bool = False) -> None:
        """Apply display-only lineage compare mode to selected steps."""
        host = self.host
        mode = str(mode or "")
        self._normalize_selected_indices(len(self._history_entries()))
        if not self._is_compare_mode_valid(mode):
            if not silent:
                if mode in {"slider", "diff"}:
                    host._set_runtime_summary("状态：滑动对比/差值图需要恰好选择 2 个步骤", "warning")
                else:
                    host._set_runtime_summary("状态：网格对比需要选择 2–4 个步骤", "warning")
            self._sync_mode_buttons_checked()
            return
        snapshots = self.get_selected_compare_snapshots()
        if len(snapshots) < 2:
            self._sync_mode_buttons_checked()
            return
        try:
            host._set_compare_snapshots(snapshots)
            labels = [snap.get("label") for snap in snapshots]
            self._compare_mode = mode
            host._lineage_compare_source_index = int(snapshots[0].get("source_index", 0))
            host._lineage_compare_source_indices = [int(snap.get("source_index", 0)) for snap in snapshots]
            if mode == "slider":
                self._set_advanced_mode_slider()
                self._sync_compare_combos(labels)
            elif mode == "diff":
                self._set_advanced_mode_diff()
                self._sync_compare_combos(labels)
            else:
                self._set_advanced_mode_grid()
            host._main_slider_compare_ratio = 0.5
            host._last_plot_signature = None
            host._refresh_plot()
            self._sync_mode_buttons_checked()
            self._update_compare_tray()
            if not silent:
                mode_text = {"slider": "滑动对比", "grid": "网格对比", "diff": "差值图"}.get(mode, mode)
                host._set_runtime_summary(f"状态：链路{mode_text} · " + " ↔ ".join(labels), "info")
        except Exception as exc:
            host._set_runtime_summary(f"状态：链路对比失败 · {exc}", "danger")

    def deactivate_compare_mode(self, *, silent: bool = False) -> None:
        """Close active lineage compare mode and return to single-image view."""
        host = self.host
        try:
            if hasattr(host, "_transient_compare_snapshots"):
                host._transient_compare_snapshots = [
                    snap
                    for snap in getattr(host, "_transient_compare_snapshots", [])
                    if str(snap.get("source") or "") != "processing_lineage"
                ]
            if hasattr(host, "_refresh_compare_snapshots_from_state"):
                host._refresh_compare_snapshots_from_state()
            host._lineage_compare_source_index = None
            host._lineage_compare_source_indices = []
            self._compare_mode = None
            host._main_slider_compare_ratio = 0.5
            self.clear_display_override()
            self._set_advanced_mode_single()
            self._sync_mode_buttons_checked()
            self._update_compare_tray()
            host._last_plot_signature = None
            host._refresh_plot()
            self.update_display()
            if not silent:
                host._set_runtime_summary("状态：已取消链路对比，返回当前正式结果", "neutral")
        except Exception as exc:
            if not silent:
                host._set_runtime_summary(f"状态：取消链路对比失败 · {exc}", "warning")

    def clear_compare_selection(self) -> None:
        """Clear compare tray selection and return to single-image mode."""
        self._compare_selected_indices.clear()
        self.deactivate_compare_mode(silent=True)
        self.sync_stepper()
        self._update_compare_tray()
        self.host._set_runtime_summary("状态：对比篮已清空", "neutral")

    # Compatibility wrappers retained for older tests/call sites.
    def _set_compare_button_checked(self, checked: bool) -> None:
        self._sync_mode_buttons_checked()

    def toggle_selected_step_compare(self, checked: bool) -> None:
        if checked:
            if self._last_selected_index is not None:
                self._compare_selected_indices.add(int(self._last_selected_index))
                entries = self._history_entries()
                if entries:
                    self._compare_selected_indices.add(len(entries) - 1)
            self.apply_compare_mode("slider")
        else:
            self.deactivate_compare_mode()

    def compare_selected_step_with_current(self) -> None:
        if self._last_selected_index is not None:
            self._compare_selected_indices.add(int(self._last_selected_index))
            entries = self._history_entries()
            if entries:
                self._compare_selected_indices.add(len(entries) - 1)
        self.apply_compare_mode("slider")

    def exit_lineage_compare(self, *, silent: bool = False) -> None:
        self.deactivate_compare_mode(silent=silent)

    def on_compare_mode_disabled(self) -> None:
        """Synchronize lineage controls when global compare mode is closed elsewhere."""
        if self.is_lineage_compare_active():
            self.deactivate_compare_mode(silent=True)
        else:
            self.update_step_detail()

    # ------------------------------------------------------------------
    # Display override and plot payload
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Text/export alignment
    # ------------------------------------------------------------------

    def build_steps(self) -> list[str]:
        """Build compact processing lineage from formal shared history labels."""
        entries = self._history_entries()
        if not entries:
            return ["Raw"]
        labels: list[str] = []
        for idx, item in enumerate(entries):
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            if idx == 0 or label in {"原始数据", "原始", "Raw", "raw"}:
                norm = "Raw"
            else:
                norm = label
            if not labels or labels[-1] != norm:
                labels.append(norm)
        return labels or ["Raw"]

    def build_copy_text(self) -> str:
        """Build report-friendly lineage text with key parameters."""
        entries = self._history_entries()
        if not entries:
            return "Raw"
        parts: list[str] = []
        for idx, entry in enumerate(entries):
            label = "Raw" if idx == 0 else str(entry.get("label") or f"Step {idx + 1}")
            params = self._entry_params(entry)
            if params and idx != 0:
                label = f"{label}({self._format_params_inline(params, limit=5)})"
            parts.append(label)
        return " → ".join(parts)

    def build_text(self) -> str:
        """Return compact lineage string for toolbar/title."""
        steps = self.build_steps()
        if len(steps) <= 1:
            return "Raw"
        return " -> ".join(steps)

    def build_tooltip(self) -> str:
        """Return detailed lineage tooltip text."""
        host = self.host
        entries = self._history_entries()
        lines = [f"数据源: {host.data_path or '未加载'}", "处理链路:"]
        if entries:
            for idx, entry in enumerate(entries):
                label = "Raw" if idx == 0 else str(entry.get("label") or f"Step {idx + 1}")
                lines.append(f"{idx + 1}. {label} | {self._status_text_for_entry(entry, idx, len(entries))} | {self._entry_shape_text(entry)}")
        else:
            lines.append("1. Raw")
        lines.append("说明: 视图交互（平移/缩放/滑动/历史预览）不写入处理链路。")
        return "\n".join(lines)

    def build_export_steps(self) -> list[dict[str, Any]]:
        """Build UI-aligned processing-chain step records for report sidecars."""
        entries = self._history_entries()
        steps: list[dict[str, Any]] = []
        total = len(entries)
        for idx, entry in enumerate(entries):
            header = self._entry_header(entry)
            warnings = self._entry_warnings(entry)
            item = {
                "index": idx,
                "role": "original" if idx == 0 else ("current" if idx == total - 1 else "history"),
                "label": "Raw" if idx == 0 else str(entry.get("label") or f"Step {idx + 1}"),
                "ui_status": self._status_text_for_entry(entry, idx, total),
                "shape": self._entry_shape(entry),
                "method_key": self._entry_method_key(entry) or None,
                "display_title": header.get("display_title"),
                "params": self._entry_params(entry),
                "runtime_ms": self._entry_elapsed_ms(entry),
                "warnings": warnings,
                "autotune_scoring_record": header.get("autotune_scoring_record") if isinstance(header.get("autotune_scoring_record"), dict) else {},
                "autotune_recipe_step": header.get("autotune_recipe_step") if isinstance(header.get("autotune_recipe_step"), dict) else {},
                "autotune_recipe_plan": header.get("autotune_recipe_plan") if isinstance(header.get("autotune_recipe_plan"), dict) else {},
                "has_warning": bool(warnings),
                "has_full_data": entry.get("data") is not None,
                "memory_state": "summary_only" if bool(entry.get("pruned")) else "stored",
                "display_only_preview": bool(idx != total - 1),
                "selected_for_compare": bool(idx in self._compare_selected_indices),
                "active_lineage_compare_mode": self._compare_mode,
                "compare_display_only": bool(self._compare_mode),
                "exportable": bool(entry.get("data") is not None),
            }
            steps.append(item)
        return steps

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
                    logger.warning("Failed to get active label for lineage display", exc_info=True)
                    active_label = "历史步骤"
                host._plot_lineage_label.setText(
                    f"链路: {lineage_text} · 查看 {active_label}"
                )
            else:
                host._plot_lineage_label.setText(f"链路: {lineage_text}")
            host._plot_lineage_label.setToolTip(self.build_tooltip())
        self.sync_stepper()
