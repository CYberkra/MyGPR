#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target annotation callbacks and source binding helpers for MyGPR."""

from __future__ import annotations

import numpy as np

from core.gpr_data_model import GPRDataSet
from core.processing_artifact_index import index_processing_artifacts
from core.target_detection import detect_targets
from core.target_source_data import TargetSourceDataView, resolve_target_source_view
from core.target_source_binding import (
    TargetSourceBinding,
    artifact_target_source,
    bind_target_to_source,
    raw_target_source,
    source_label_from_target,
)
from ui.field_panels.plots import draw_target_bscan


class TargetActionsMixin:
    def _available_target_sources(self) -> list[TargetSourceBinding]:
        """Return raw and saved-artifact sources for the selected line."""
        sources: list[TargetSourceBinding] = [raw_target_source(self.active_gpr_dataset, line_id=self.selected_line)]
        if self.project_root is not None:
            try:
                for record in index_processing_artifacts(self.project_root, line_id=self.selected_line):
                    sources.append(artifact_target_source(record))
            except Exception:
                # Source listing must never break the interpretation page.
                pass
        return sources

    def _current_target_source(self) -> TargetSourceBinding:
        if self.target_source_combo is not None:
            data = self.target_source_combo.currentData()
            if isinstance(data, TargetSourceBinding):
                return data
        return raw_target_source(self.active_gpr_dataset, line_id=self.selected_line)

    def _refresh_target_source_options(self, preferred_source_id: str | None = None) -> None:
        combo = self.target_source_combo
        if combo is None:
            return
        preferred = preferred_source_id or getattr(self, "current_target_source_id", "")
        combo.blockSignals(True)
        combo.clear()
        selected_index = 0
        for index, source in enumerate(self._available_target_sources()):
            combo.addItem(source.label, source)
            if preferred and source.source_result_id == preferred:
                selected_index = index
        combo.setCurrentIndex(selected_index)
        current = combo.currentData()
        self.current_target_source_id = current.source_result_id if isinstance(current, TargetSourceBinding) else ""
        combo.blockSignals(False)

    def _on_target_source_changed(self, _index: int) -> None:
        source = self._current_target_source()
        self.current_target_source_id = source.source_result_id
        if self.target_log_label is not None:
            note = "；该来源包含坐标轴转换" if source.is_axis_transform else ""
            self.target_log_label.setText(f"ⓘ  标注来源已切换：{source.label}{note}。B-scan 已按该来源刷新。")
        self._refresh_target_widgets()

    def _current_target_source_view(self) -> TargetSourceDataView | None:
        """Return the concrete B-scan view for the selected annotation source."""
        try:
            return resolve_target_source_view(
                project_root=self.project_root,
                source=self._current_target_source(),
                raw_dataset=self.active_gpr_dataset,
                fallback_dataset=self.processed_gpr_dataset,
            )
        except Exception as exc:
            if self.target_log_label is not None:
                self.target_log_label.setText(f"⚠  标注来源读取失败：{exc}")
            return None

    def _target_source_dataset(self) -> GPRDataSet | None:
        view = self._current_target_source_view()
        return view.dataset if view is not None else None

    def _bind_target_to_current_source(self, target: dict) -> dict:
        return bind_target_to_source(target, self._current_target_source())

    def _select_target_from_table(self, row: int, _column: int = 0) -> None:
        if row < 0 or row >= len(self.targets):
            return
        self.current_target_index = row
        self._refresh_target_widgets()

    def _refresh_target_widgets(self) -> None:
        if self.current_target_index >= len(self.targets):
            self.current_target_index = max(0, len(self.targets) - 1)
        if self.target_table is not None:
            self._fill_table(self.target_table, self._target_rows(), highlight_row=self.current_target_index)
        if self.target_canvas is not None:
            self._draw_current_target_bscan(self.target_canvas)
        self._update_target_info_panel()

    def _target_color(self, index: int) -> str:
        colors = ["#25B26B", "#7C4DFF", "#F04444", "#2B86F6", "#F5A623", "#0D91B2", "#E879F9", "#22C55E"]
        return colors[index % len(colors)]

    def _add_preview_target(self, mileage: float | None = None, depth: float | None = None) -> None:
        idx = len(self.targets) + 1
        mileage = float(mileage if mileage is not None else min(196.0, 28.0 + idx * 21.0))
        depth = float(depth if depth is not None else 1.2 + (idx % 4) * 0.22)
        if self.trajectory_model is not None:
            point = self.trajectory_model.interpolate(mileage)
            x_val, y_val = point.x, point.y
        else:
            x_val, y_val = "", ""
        target = {
            "name": f"T-{idx:02d}",
            "target_id": f"T-{idx:02d}",
            "line_id": self.selected_line,
            "type": "疑似管线",
            "mileage": mileage,
            "depth": depth,
            "x": x_val,
            "y": y_val,
            "confidence": "★★★☆☆",
            "status": "待确认",
            "note": "现场新增标注，待人工复核",
            "color": self._target_color(idx - 1),
            "width": 52,
            "height": 58,
        }
        self.targets.append(self._bind_target_to_current_source(target))
        self.current_target_index = len(self.targets) - 1
        if self.target_log_label is not None:
            self.target_log_label.setText(f"＋  已新增标注 {self.targets[-1]['name']}：里程 {mileage:.2f} m，深度 {depth:.2f} m。")
        self._save_targets_to_project()
        self._refresh_target_widgets()

    def _auto_detect_targets(self) -> None:
        dataset = self._target_source_dataset()
        added = 0
        if dataset is not None:
            candidates = detect_targets(dataset, max_targets=5, start_index=len(self.targets) + 1)
            existing = {(round(float(t.get("mileage", 0.0)), 1), round(float(t.get("depth", 0.0)), 1)) for t in self.targets}
            for candidate in candidates:
                target = candidate.to_target_dict()
                key = (round(float(target["mileage"]), 1), round(float(target["depth"]), 1))
                if key in existing:
                    continue
                if self.trajectory_model is not None:
                    point = self.trajectory_model.interpolate(float(target["mileage"]))
                    target["x"] = point.x
                    target["y"] = point.y
                target["color"] = self._target_color(len(self.targets))
                target["width"] = 54
                target["height"] = 58
                self.targets.append(self._bind_target_to_current_source(target))
                existing.add(key)
                added += 1
                if added >= 2:
                    break
        else:
            if self.target_log_label is not None:
                self.target_log_label.setText("⚠  当前测线没有可识别的 GPR 矩阵，请先导入测线数据。")
            return
        self.current_target_index = max(0, len(self.targets) - 1)
        if self.target_log_label is not None:
            source = self._current_target_source().label if dataset is not None else "预览数据"
            self.target_log_label.setText(f"✥  自动识别辅助完成：基于{source}新增 {added} 个候选目标，均标记为待复核。")
        self._save_targets_to_project()
        self._refresh_target_widgets()

    def _save_targets(self) -> None:
        if not self.targets:
            return
        self.targets[self.current_target_index]["status"] = "已确认"
        self._save_targets_to_project()
        if self.target_log_label is not None:
            t = self.targets[self.current_target_index]
            self.target_log_label.setText(
                f"✓  已保存标注：{t['name']}（{t['type']}），来源：{source_label_from_target(t)}；targets/{self.selected_line}_targets.csv 与 spatial/{self.selected_line}_targets_xy.csv 已同步。"
            )
        self._refresh_target_widgets()

    def _delete_selected_target(self) -> None:
        if not self.targets:
            return
        removed = self.targets.pop(self.current_target_index)
        self.current_target_index = max(0, self.current_target_index - 1)
        self._save_targets_to_project()
        if self.target_log_label is not None:
            self.target_log_label.setText(f"⌫  已删除标注：{removed['name']}，目标 CSV 已同步。")
        self._refresh_target_widgets()

    def _draw_current_target_bscan(self, canvas) -> None:
        view = self._current_target_source_view()
        if view is None:
            self._draw_empty_plot(canvas, "暂无目标定位 B-scan\n请先导入当前测线数据")
            return
        draw_target_bscan(
            canvas,
            self.targets,
            selected=self.current_target_index,
            data_matrix=view.matrix,
            distance_axis_m=view.distance_axis_m,
            depth_axis_m=view.depth_axis_m,
            vertical_axis=view.vertical_axis,
            vertical_axis_label=view.vertical_axis_label,
            source_label=view.source.method_name or view.source.label,
        )

    def _mileage_from_target_canvas_x(self, xdata: float, view: TargetSourceDataView | None) -> float:
        if view is not None and len(view.distance_axis_m) >= 2:
            cols = max(int(view.matrix.shape[1]) - 1, 1)
            frac = max(0.0, min(1.0, float(xdata) / cols))
            return float(view.distance_axis_m[0] + frac * (view.distance_axis_m[-1] - view.distance_axis_m[0]))
        return max(0.0, min(200.0, float(xdata) / 420.0 * 200.0))

    def _depth_from_target_canvas_y(self, ydata: float, view: TargetSourceDataView | None) -> float:
        if view is not None and len(view.depth_axis_m) >= 2:
            rows = max(int(view.matrix.shape[0]) - 1, 1)
            frac = max(0.0, min(1.0, float(ydata) / rows))
            return float(view.depth_axis_m[0] + frac * (view.depth_axis_m[-1] - view.depth_axis_m[0]))
        return max(0.4, min(5.0, float(ydata) / 190.0 * 5.0))

    def _on_target_canvas_click(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        view = self._current_target_source_view()
        mileage = self._mileage_from_target_canvas_x(float(event.xdata), view)
        depth = self._depth_from_target_canvas_y(float(event.ydata), view)
        self._add_preview_target(mileage, depth)

    def _update_target_info_panel(self) -> None:
        if not self.target_field_labels:
            return
        if not self.targets:
            for label in self.target_field_labels.values():
                label.setText("--")
            if getattr(self, "target_info_title_label", None) is not None:
                self.target_info_title_label.setText("当前目标 --")
            if getattr(self, "target_info_subtitle_label", None) is not None:
                self.target_info_subtitle_label.setText("请选择或新增一个目标标注")
            if self.target_preview_canvas is not None:
                self._draw_current_line_strip(self.target_preview_canvas)
            return
        target = self.targets[self.current_target_index]
        raw_x = target.get("x", "")
        raw_y = target.get("y", "")
        try:
            coord_text = f"X {float(raw_x):.3f}    Y {float(raw_y):.3f}" if raw_x not in (None, "") and raw_y not in (None, "") else "未定位（缺少真实轨迹）"
        except Exception:
            coord_text = "未定位（坐标无效）"
        if getattr(self, "target_info_title_label", None) is not None:
            self.target_info_title_label.setText(f"当前目标  {target['name']}")
        if getattr(self, "target_info_subtitle_label", None) is not None:
            self.target_info_subtitle_label.setText(f"{target['type']}｜{target['status']}｜{target['confidence']}")
        values = {
            "类型 *": target["type"],
            "里程 (m)": f"{float(target['mileage']):.2f}",
            "深度 (m)": f"{float(target['depth']):.2f}",
            "坐标 (m)": coord_text,
            "置信度": target["confidence"],
            "状态": target["status"],
            "备注": target["note"],
            "来源处理结果": source_label_from_target(target),
        }
        for key, label in self.target_field_labels.items():
            label.setText(values.get(key, "--"))
        if self.target_preview_canvas is not None:
            self._draw_current_line_strip(self.target_preview_canvas, marker=float(target["mileage"]))


__all__ = ["TargetActionsMixin"]
