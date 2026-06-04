# -*- coding: utf-8 -*-
"""Display, plotting, preset and main figure helpers for app_qt.GPRGuiQt.

This mixin is a low-risk structural extraction from ``app_qt.py``.  It keeps
method bodies unchanged except for module-level imports, so behaviour remains
owned by the main window while the entry file stays smaller.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from core.methods_registry import PROCESSING_METHODS
from core.preset_profiles import GUI_PRESETS_V1
from PythonModule.kirchhoff_migration import load_cagpr_kir_parameter_file

logger = logging.getLogger(__name__)


class MainWindowDisplayMixin:
    def _parse_int_edit(self, edit, default: int = 0) -> int:
        """解析整数输入"""
        text = (edit.text() or "").strip()
        if text == "":
            return default
        try:
            return int(float(text))
        except Exception:
            return default

    def _get_colormap(self, header_info_override: dict | None = None):
        """获取当前色图"""
        cmap = (self.page_advanced.cmap_combo.currentText() or "gray").strip()
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        if (
            cmap == "gray"
            and header.get("display_hint") == "signed_migration"
            and not self.page_advanced.cmap_invert_var.isChecked()
        ):
            return "seismic"
        if self.page_advanced.cmap_invert_var.isChecked():
            if cmap.endswith("_r"):
                cmap = cmap[:-2]
            else:
                cmap = cmap + "_r"
        return cmap

    def _set_display_override(
        self,
        data: np.ndarray | None,
        header_info: dict | None = None,
        trace_metadata: dict | None = None,
    ):
        return self.processing_lineage_controller.set_display_override(
            data, header_info=header_info, trace_metadata=trace_metadata
        )

    def _clear_display_override(self):
        return self.processing_lineage_controller.clear_display_override()

    def _get_active_plot_payload(
        self, fallback_data: np.ndarray | None = None
    ) -> tuple[np.ndarray | None, dict | None, dict | None]:
        return self.processing_lineage_controller.get_active_plot_payload(
            fallback_data=fallback_data
        )

    def _is_single_view_mode(self) -> bool:
        """主图是否处于单图模式。"""
        return not bool(
            self.page_advanced.compare_var.isChecked()
            or self.page_advanced.diff_var.isChecked()
            or self.page_advanced.slider_compare_var.isChecked()
        )

    def _get_selected_single_view_snapshot(self):
        """返回单图下拉框当前选择的正式/临时快照。"""
        combo = getattr(self.page_advanced, "single_view_combo", None)
        if combo is None:
            return None
        label = combo.currentText()
        if not label:
            return None
        return next((s for s in self.compare_snapshots if s["label"] == label), None)

    def _normalize_plot_stage_label(self, label: str | None) -> str:
        """Return a compact stage label for the main B-scan figure title."""
        text = str(label or "").strip()
        if not text or text in {"当前", "Current", "current"}:
            return ""
        if text in {"Raw", "raw", "原始", "原始数据"}:
            return "原始数据"
        return text

    def _get_single_plot_title(self, header_info_override: dict | None = None) -> str:
        """获取单图模式下的标题。"""
        if header_info_override and header_info_override.get("live_preview"):
            live_stage = self._normalize_plot_stage_label(
                header_info_override.get("display_title")
            )
            if live_stage:
                return f"B-scan / {live_stage}"

        stage = ""
        if self._is_single_view_mode():
            selected = getattr(self.page_advanced, "single_view_combo", None)
            selected_label = selected.currentText() if selected is not None else ""
            stage = self._normalize_plot_stage_label(selected_label)
        if not stage:
            stage = self._normalize_plot_stage_label(
                getattr(self.shared_data, "current_label", None)
            )
        if not stage:
            header = (
                header_info_override
                if header_info_override is not None
                else (self.header_info or {})
            )
            stage = self._normalize_plot_stage_label(
                header.get("display_title") if header else ""
            )
        stage = stage or "原始数据"
        return f"B-scan / {stage}"

    def _build_processing_lineage_steps(self) -> list[str]:
        return self.processing_lineage_controller.build_steps()

    def _build_processing_lineage_text(self) -> str:
        return self.processing_lineage_controller.build_text()

    def _build_processing_lineage_tooltip(self) -> str:
        return self.processing_lineage_controller.build_tooltip()

    def _update_processing_lineage_display(self):
        return self.processing_lineage_controller.update_display()

    def _resolve_method_params(self, method_key: str):
        """解析方法参数"""
        method = PROCESSING_METHODS[method_key]
        defaults = {p["name"]: p.get("default") for p in method.get("params", [])}
        overrides = self._method_param_overrides.get(method_key, {})
        defaults.update(overrides)
        return defaults

    def _build_tasks_from_order(self, order: list, out_dir: str):
        """从顺序构建任务列表"""
        tasks = []
        for key in order:
            if key not in self.page_basic.method_keys:
                continue
            tasks.append(self._build_single_task(key, out_dir))
        return tasks

    def _build_single_task(
        self, method_key: str, out_dir: str, param_source_mode: str = "manual"
    ) -> dict:
        """构建单个任务字典"""
        method = PROCESSING_METHODS[method_key]
        params = self._resolve_method_params(method_key)
        return {
            "method_key": method_key,
            "method": method,
            "params": params,
            "out_dir": out_dir,
            "param_source_mode": param_source_mode,
        }

    def _apply_preset_by_key(self, preset_key: str):
        """根据预设键应用预设参数"""
        if preset_key not in GUI_PRESETS_V1:
            QMessageBox.warning(self, "预设错误", f"未知预设：{preset_key}")
            return

        preset = GUI_PRESETS_V1[preset_key]
        self._selected_preset_key = preset_key
        self._log(f"应用预设: {preset_key} - {preset['label']}")
        self._apply_preset_ui_values(preset.get("ui"), preset_key=preset_key)
        self._apply_preset_method_params(preset.get("method_params"))
        if self.data is not None:
            self._refresh_plot()

    def backfill_current_method_params(self):
        """回填当前方法的参数"""
        if self.data is None:
            QMessageBox.information(self, "回填参数", "请先加载数据。")
            return

        current_method_key = self.page_basic.get_current_method_key()
        if not current_method_key:
            QMessageBox.warning(self, "回填参数", "请选择一个处理方法。")
            return

        # 从已应用的预设中获取参数，如果没有则从方法默认参数中获取
        method_params = self._method_param_overrides.get(current_method_key, {})
        if not method_params:
            # 如果没有覆盖参数，尝试从方法定义中获取默认参数
            method_definition = PROCESSING_METHODS.get(current_method_key)
            if method_definition and "params" in method_definition:
                method_params = {
                    p["name"]: p.get("default") for p in method_definition["params"]
                }

        if method_params:
            self.page_basic.apply_method_params(current_method_key, method_params)
            self._log(
                f"已回填当前方法({PROCESSING_METHODS[current_method_key]['name']})的参数。"
            )
        else:
            QMessageBox.information(self, "回填参数", "当前方法没有可回填的参数。")

    def import_tzt_as_migration_defaults(self):
        """导入 TZT 文件作为迁移默认值"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Kirchhoff 参数文件",
            "",
            "Kirchhoff 参数 (*.txt *.tzt);;所有文件 (*)",
        )
        if not path:
            return
        try:
            params = load_cagpr_kir_parameter_file(path)
            if not params:
                QMessageBox.information(
                    self, "参数导入", "未识别到可用的 Kirchhoff 参数。"
                )
                return
            self._method_param_overrides["kirchhoff_migration"] = dict(params)
            self.page_basic.apply_method_params("kirchhoff_migration", params)
            current_key = self.page_basic.get_current_method_key()
            if current_key != "kirchhoff_migration":
                idx = self.page_basic.method_keys.index("kirchhoff_migration")
                self.page_basic.method_combo.setCurrentIndex(idx)
            self._log(f"已从参数文件导入 Kirchhoff 默认配置: {path}")
        except Exception as e:
            QMessageBox.warning(
                self, "参数导入失败", f"无法导入 Kirchhoff 参数文件:\n{e}"
            )
            self._log(f"导入 Kirchhoff 参数文件失败: {e}")

    def _reset_crop(self):
        """重置裁剪设置"""
        self.page_advanced.crop_enable_var.setChecked(False)
        self.page_advanced.time_start_edit.setText("")
        self.page_advanced.time_end_edit.setText("")
        self.page_advanced.dist_start_edit.setText("")
        self.page_advanced.dist_end_edit.setText("")
        self._log("裁剪设置已重置。")
        self._refresh_plot()

    def cancel_processing(self):
        """取消处理"""
        if (
            (self._worker is None or self._worker_thread is None)
            and (self._auto_tune_worker is None or self._auto_tune_thread is None)
            and (
                self._auto_tune_stage_worker is None
                or self._auto_tune_stage_thread is None
            )
            and (
                self._auto_tune_comparison_worker is None
                or self._auto_tune_comparison_thread is None
            )
        ):
            self.status_label.setText("当前无可取消任务")
            return
        if self._cancel_in_flight:
            self.status_label.setText("正在取消...（等待当前步骤安全退出）")
            return
        try:
            self._cancel_in_flight = True
            if self._worker is not None:
                self._worker.request_cancel()
            if self._auto_tune_worker is not None:
                self._auto_tune_worker.request_cancel()
            if self._auto_tune_stage_worker is not None:
                self._auto_tune_stage_worker.request_cancel()
            if self._auto_tune_comparison_worker is not None:
                self._auto_tune_comparison_worker.request_cancel()
            self.page_basic.btn_cancel.setEnabled(False)
            self.status_label.setText("正在取消...（等待当前步骤安全退出）")
            self._log("收到取消请求：将于当前步骤完成后停止")
        except Exception as e:
            self._cancel_in_flight = False
            self._log(f"取消请求失败: {e}")

    def _build_plot_signature(self):
        """构建绘图签名用于缓存判断"""
        plot_data, plot_header_info, _ = self._get_active_plot_payload(self.data)
        if plot_data is None:
            return None
        header = plot_header_info or {}
        skip_preprocess = bool(header.get("display_skip_preprocess"))
        slider_compare = bool(
            hasattr(self.page_advanced, "slider_compare_var")
            and self.page_advanced.slider_compare_var.isChecked()
        )
        sig = {
            "shape": plot_data.shape,
            "revision": self._data_revision,
            "display_override": self._display_data_override is not None,
            "display_override_revision": self._display_override_revision,
            "lineage_view_index": self._lineage_view_index,
            "manual_roi": tuple(sorted((self._manual_roi_values or {}).items())),
            "view_limits": tuple(sorted((self._main_view_limits or {}).items())),
            "cmap": self._get_colormap(plot_header_info),
            "view_style": self.page_advanced.get_view_style(),
            "symmetric": self.page_advanced.symmetric_var.isChecked(),
            "auto_contrast": self.page_advanced.auto_contrast_var.isChecked(),
            "compare": self.page_advanced.compare_var.isChecked(),
            "slider_compare": slider_compare,
            "cmap_invert": self.page_advanced.cmap_invert_var.isChecked(),
            "show_cbar": self.page_advanced.show_cbar_var.isChecked(),
            "show_grid": self.page_advanced.show_grid_var.isChecked(),
            "show_physical_x_axis": self.page_advanced.show_physical_x_axis_var.isChecked(),
            "show_physical_y_axis": self.page_advanced.show_physical_y_axis_var.isChecked(),
            "percentile": self.page_advanced.percentile_var.isChecked(),
            "normalize": False
            if skip_preprocess
            else self.page_advanced.normalize_var.isChecked(),
            "demean": False
            if skip_preprocess
            else self.page_advanced.demean_var.isChecked(),
            "crop": self.page_advanced.crop_enable_var.isChecked(),
        }
        controller = getattr(self, "processing_lineage_controller", None)
        if controller is not None:
            try:
                sig["lineage_compare_mode"] = controller.get_active_compare_mode()
                sig["lineage_compare_indices"] = tuple(controller._selected_compare_indices_sorted())
            except Exception:
                pass
        if self.page_advanced.compare_var.isChecked() or slider_compare:
            sig["left"] = self.page_advanced.compare_left_combo.currentText()
            sig["right"] = self.page_advanced.compare_right_combo.currentText()
            sig["diff"] = self.page_advanced.diff_var.isChecked()
            sig["slider_ratio"] = round(float(self._main_slider_compare_ratio), 4)
        else:
            sig["single"] = self.page_advanced.single_view_combo.currentText()
        return tuple(sorted(sig.items()))

    def _limited_cache_set(self, cache: dict, key, value, max_entries: int = 8) -> None:
        """Store a small UI cache entry with FIFO eviction."""
        try:
            cache[key] = value
            while len(cache) > int(max_entries):
                oldest = next(iter(cache))
                cache.pop(oldest, None)
        except Exception:
            logger.debug("Failed to update display cache", exc_info=True)

    def _display_cache_header_key(self, header: dict | None) -> tuple:
        """Return only display-affecting header fields for render caches."""
        header = header or {}
        keys = (
            "display_skip_preprocess",
            "display_fixed_unit_range",
            "display_center_zero",
            "display_percentile_abs_high",
            "display_bad_color",
            "display_show_cbar",
            "display_hint",
            "is_elevation",
            "total_time_ns",
            "sampling_interval_ns",
            "trace_spacing",
        )
        return tuple((key, self._make_hashable(header.get(key))) for key in keys)

    def _make_hashable(self, value):
        """Convert simple nested values to a hashable representation for UI caches."""
        if isinstance(value, dict):
            return tuple(sorted((str(k), self._make_hashable(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(self._make_hashable(v) for v in value)
        if isinstance(value, np.ndarray):
            return ("ndarray", tuple(value.shape), str(value.dtype))
        try:
            hash(value)
            return value
        except Exception:
            return repr(value)

    def _build_prepare_view_cache_key(
        self,
        data: np.ndarray,
        header: dict | None,
        trace_metadata_override: dict | None,
    ) -> tuple:
        """Build a cache key for display-only prepared B-scan data."""
        arr = np.asarray(data)
        try:
            ptr = int(arr.__array_interface__.get("data", (0,))[0])
        except Exception:
            ptr = id(arr)
        return (
            "prepare_view",
            int(getattr(self, "_data_revision", 0)),
            int(getattr(self, "_display_override_revision", 0)),
            ptr,
            tuple(arr.shape),
            str(arr.dtype),
            self._display_cache_header_key(header),
            id(trace_metadata_override),
            bool(self.page_advanced.crop_enable_var.isChecked()),
            self._parse_float_edit(self.page_advanced.crop_time_start_edit, default=0.0)
            if hasattr(self.page_advanced, "crop_time_start_edit") else None,
            self._parse_float_edit(self.page_advanced.crop_time_end_edit, default=0.0)
            if hasattr(self.page_advanced, "crop_time_end_edit") else None,
            bool(self.page_advanced.normalize_var.isChecked())
            if not (header or {}).get("display_skip_preprocess") else False,
            bool(self.page_advanced.demean_var.isChecked())
            if not (header or {}).get("display_skip_preprocess") else False,
            bool(self.page_advanced.show_physical_x_axis_var.isChecked()),
            bool(self.page_advanced.show_physical_y_axis_var.isChecked()),
            bool(self.page_advanced.percentile_var.isChecked()),
            bool(self.page_advanced.auto_contrast_var.isChecked()),
            bool(self.page_advanced.symmetric_var.isChecked()),
        )

    def _prepare_view_data(
        self,
        data: np.ndarray,
        header_info_override: dict | None = None,
        trace_metadata_override: dict | None = None,
    ):
        """准备用于显示的数据（裁剪→预处理，保留全分辨率数据）。

        This path is display-only.  Cache by data revision and display settings so
        ROI redraws, view-limit changes and repeated tab refreshes do not repeat
        normalization/cropping work for the same B-scan array.
        """
        start_ts = time.perf_counter()
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        cache = getattr(self, "_view_cache", None)
        cache_key = None
        if cache is not None:
            try:
                cache_key = self._build_prepare_view_cache_key(
                    data, header, trace_metadata_override
                )
                cached = cache.get(cache_key)
                if cached is not None:
                    self._last_view_cache_hit = True
                    self._last_prepare_ms = (time.perf_counter() - start_ts) * 1000.0
                    monitor = getattr(self, "_perf_monitor", None)
                    if monitor is not None:
                        monitor.record("display.prepare_view_cache_hit_ms", self._last_prepare_ms)
                    return cached
            except Exception:
                cache_key = None
                logger.debug("Prepare-view cache lookup failed", exc_info=True)

        self._last_view_cache_hit = False
        display_data = np.array(data, copy=False)
        time_axis = self._build_time_axis(display_data.shape[0], header_info_override)
        trace_axis = self._build_trace_axis(
            display_data.shape[1], trace_metadata_override, header_info_override
        )
        trace_indices = np.arange(display_data.shape[1], dtype=np.int32)
        bounds = (
            self._get_crop_bounds(display_data, time_axis, trace_axis)
            if self.page_advanced.crop_enable_var.isChecked()
            else None
        )
        if bounds:
            t0, t1 = bounds["time_start_idx"], bounds["time_end_idx"]
            d0, d1 = bounds["dist_start_idx"], bounds["dist_end_idx"]
            display_data = display_data[t0:t1, d0:d1]
            time_axis = time_axis[t0:t1]
            trace_axis = trace_axis[d0:d1]
            trace_indices = trace_indices[d0:d1]
        display_data = self._apply_preprocess(
            display_data, header_info_override=header_info_override
        )
        display_data = self._apply_display_transform(
            display_data, header_info_override=header_info_override
        )
        result = (
            display_data,
            bounds,
            {
                "time_axis": time_axis,
                "trace_axis": trace_axis,
                "trace_indices": trace_indices,
            },
        )
        self._last_prepare_ms = (time.perf_counter() - start_ts) * 1000.0
        monitor = getattr(self, "_perf_monitor", None)
        if monitor is not None:
            monitor.record("display.prepare_view_ms", self._last_prepare_ms)
        if cache is not None and cache_key is not None:
            self._limited_cache_set(cache, cache_key, result, max_entries=8)
        return result

    def _build_time_axis(
        self, n_samples: int, header_info_override: dict | None = None
    ) -> np.ndarray:
        """构建时间轴（ns 或采样索引）。"""
        header = (
            header_info_override
            if header_info_override is not None
            else self.header_info
        )
        if header and header.get("is_elevation"):
            if not self._use_physical_vertical_axis():
                return np.arange(n_samples, dtype=np.float32)
            elevation_top = header.get("elevation_top_m")
            depth_step = header.get("depth_step_m")
            if (
                elevation_top is not None
                and depth_step is not None
                and float(depth_step) > 0
            ):
                return float(elevation_top) - np.arange(
                    n_samples, dtype=np.float32
                ) * float(depth_step)
            elevation_bottom = header.get("elevation_bottom_m")
            if elevation_top is not None and elevation_bottom is not None:
                return np.linspace(
                    float(elevation_top),
                    float(elevation_bottom),
                    n_samples,
                    dtype=np.float32,
                )
        if header and header.get("is_depth"):
            if not self._use_physical_vertical_axis():
                return np.arange(n_samples, dtype=np.float32)
            depth_step = header.get("depth_step_m")
            if depth_step is not None and float(depth_step) > 0:
                return np.arange(n_samples, dtype=np.float32) * float(depth_step)
            depth_max = header.get("depth_max_m")
            if depth_max is not None:
                return np.linspace(0.0, float(depth_max), n_samples, dtype=np.float32)
        if header and header.get("total_time_ns"):
            if not self._use_physical_vertical_axis():
                return np.arange(n_samples, dtype=np.float32)
            total_time_ns = float(header["total_time_ns"])
            return np.linspace(0.0, total_time_ns, n_samples, dtype=np.float32)
        return np.arange(n_samples, dtype=np.float32)

    def _use_physical_vertical_axis(self) -> bool:
        """Whether the display should convert sample rows to time/depth/elevation labels."""
        return bool(
            hasattr(self.page_advanced, "show_physical_y_axis_var")
            and self.page_advanced.show_physical_y_axis_var.isChecked()
        )

    def _use_physical_horizontal_axis(self) -> bool:
        """Whether the display should convert trace columns to distance labels."""
        return bool(
            hasattr(self.page_advanced, "show_physical_x_axis_var")
            and self.page_advanced.show_physical_x_axis_var.isChecked()
        )

    def _build_trace_axis(
        self,
        n_traces: int,
        trace_metadata_override: dict | None = None,
        header_info_override: dict | None = None,
    ) -> np.ndarray:
        """构建距离轴（真实距离或均匀道距）。"""
        if not self._use_physical_horizontal_axis():
            return np.arange(n_traces, dtype=np.float32)
        meta = (
            trace_metadata_override
            if trace_metadata_override is not None
            else (self.trace_metadata or {})
        )
        distance = meta.get("trace_distance_m")
        if distance is not None:
            distance = np.asarray(distance, dtype=np.float32)
            if distance.ndim == 1 and distance.size >= n_traces:
                return distance[:n_traces]
        header = (
            header_info_override
            if header_info_override is not None
            else self.header_info
        )
        if header and header.get("trace_interval_m") is not None:
            interval = float(header.get("trace_interval_m", 0.0))
            return np.arange(n_traces, dtype=np.float32) * interval
        return np.arange(n_traces, dtype=np.float32)

    def _apply_preprocess(
        self, data: np.ndarray, header_info_override: dict | None = None
    ) -> np.ndarray:
        """应用预处理（无预处理时跳过拷贝）"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        if header.get("display_skip_preprocess"):
            return data
        do_norm = self.page_advanced.normalize_var.isChecked()
        do_demean = self.page_advanced.demean_var.isChecked()
        if not do_norm and not do_demean:
            return data
        result = np.array(data, copy=True)
        if do_norm:
            max_val = np.max(np.abs(result))
            if max_val > 0:
                result /= max_val
        if self.page_advanced.demean_var.isChecked():
            result -= np.mean(result, axis=0, keepdims=True)
        return result

    def _apply_display_transform(
        self, data: np.ndarray, header_info_override: dict | None = None
    ) -> np.ndarray:
        """应用仅用于显示的变换，例如 CaGPR 的对比度拉伸。"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        contrast = header.get("display_cagpr_contrast")
        if contrast is None:
            return data
        return self._normalize_cagpr_display(data, float(contrast))

    def _normalize_cagpr_display(self, data: np.ndarray, contrast: float) -> np.ndarray:
        """Match CaGPR's display clipping and normalization path."""
        arr = np.asarray(data, dtype=np.float64)
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return np.zeros_like(arr, dtype=np.float64)
        data_min = float(np.nanmin(arr))
        data_max = float(np.nanmax(arr))
        center = float(np.nanmean(arr))
        scale = 1.0 / (1.0 + contrast) if contrast > 0 else 1.0 - contrast
        half_range = max((data_max - data_min) * 0.5 * scale, 1.0e-12)
        vmin = center - half_range
        vmax = center + half_range
        clipped = np.array(arr, copy=True)
        clipped[finite_mask] = np.clip(clipped[finite_mask], vmin, vmax)
        normalized = np.full_like(clipped, np.nan, dtype=np.float64)
        normalized[finite_mask] = (
            2.0 * (clipped[finite_mask] - vmin) / max(vmax - vmin, 1.0e-12) - 1.0
        )
        return normalized

    def _get_crop_bounds(
        self, data: np.ndarray, time_axis: np.ndarray, trace_axis: np.ndarray
    ):
        """获取裁剪边界"""
        n_samples, n_traces = data.shape[0], data.shape[1]
        t_start = self._parse_float_edit(
            self.page_advanced.time_start_edit, default=None
        )
        t_end = self._parse_float_edit(self.page_advanced.time_end_edit, default=None)
        d_start = self._parse_float_edit(
            self.page_advanced.dist_start_edit, default=None
        )
        d_end = self._parse_float_edit(self.page_advanced.dist_end_edit, default=None)
        use_physical_time = bool(
            self._use_physical_vertical_axis()
            and
            self.header_info
            and (
                self.header_info.get("total_time_ns")
                or self.header_info.get("is_depth")
                or self.header_info.get("is_elevation")
            )
        )
        use_physical_dist = bool(
            self._use_physical_horizontal_axis()
            and
            self.trace_metadata is not None
            and "trace_distance_m" in self.trace_metadata
        ) or bool(
            self._use_physical_horizontal_axis()
            and self.header_info
            and self.header_info.get("trace_interval_m")
        )

        time_start_idx = (
            self._axis_value_to_index(time_axis, t_start, n_samples, "left")
            if t_start is not None and use_physical_time
            else max(0, int(t_start))
            if t_start is not None
            else 0
        )
        time_end_idx = (
            self._axis_value_to_index(time_axis, t_end, n_samples, "right")
            if t_end is not None and use_physical_time
            else min(n_samples, int(t_end))
            if t_end is not None
            else n_samples
        )
        dist_start_idx = (
            self._axis_value_to_index(trace_axis, d_start, n_traces, "left")
            if d_start is not None and use_physical_dist
            else max(0, int(d_start))
            if d_start is not None
            else 0
        )
        dist_end_idx = (
            self._axis_value_to_index(trace_axis, d_end, n_traces, "right")
            if d_end is not None and use_physical_dist
            else min(n_traces, int(d_end))
            if d_end is not None
            else n_traces
        )

        time_start_idx = max(0, min(time_start_idx, n_samples))
        time_end_idx = max(time_start_idx + 1, min(time_end_idx, n_samples))
        dist_start_idx = max(0, min(dist_start_idx, n_traces))
        dist_end_idx = max(dist_start_idx + 1, min(dist_end_idx, n_traces))

        return {
            "time_start_idx": time_start_idx,
            "time_end_idx": time_end_idx,
            "dist_start_idx": dist_start_idx,
            "dist_end_idx": dist_end_idx,
            "time_start": t_start if t_start is not None else 0,
            "time_end": t_end if t_end is not None else n_samples,
            "dist_start": d_start if d_start is not None else 0,
            "dist_end": d_end if d_end is not None else n_traces,
        }

    def _axis_value_to_index(
        self, axis: np.ndarray, value: float, fallback_size: int, side: str
    ) -> int:
        if axis is None or len(axis) == 0:
            return 0 if side == "left" else fallback_size
        if len(axis) > 1 and axis[0] > axis[-1]:
            reversed_axis = axis[::-1]
            reversed_idx = int(np.searchsorted(reversed_axis, float(value), side=side))
            idx = len(axis) - reversed_idx
            return max(0, min(idx, fallback_size))
        idx = int(np.searchsorted(axis, float(value), side=side))
        return max(0, min(idx, fallback_size))

    def _parse_float_edit(self, edit, default: float = None):
        """解析浮点数输入"""
        text = (edit.text() or "").strip()
        if text == "":
            return default
        try:
            return float(text)
        except Exception:
            return default

    def _resolve_plot_extent_and_labels(
        self,
        display_data: np.ndarray,
        bounds: dict,
        axis_info: dict,
        header_info_override: dict | None = None,
    ):
        """解析绘图范围和标签"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        time_axis = np.asarray(axis_info.get("time_axis", []), dtype=np.float32)
        trace_axis = np.asarray(axis_info.get("trace_axis", []), dtype=np.float32)
        if time_axis.size > 1 and trace_axis.size > 1:
            extent = [trace_axis[0], trace_axis[-1], time_axis[-1], time_axis[0]]
            xlabel = "距离 (m)" if self._use_physical_horizontal_axis() else "道数"
            ylabel = (
                "高程 (m)"
                if self._use_physical_vertical_axis() and header.get("is_elevation")
                else "深度 (m)"
                if self._use_physical_vertical_axis() and header.get("is_depth")
                else (
                    "时间 (ns)"
                    if self._use_physical_vertical_axis() and header.get("total_time_ns")
                    else "采样点"
                )
            )
        else:
            n_samples, n_traces = display_data.shape[0], display_data.shape[1]
            extent = [0, n_traces, n_samples, 0]
            xlabel, ylabel = "距离（道索引）", "时间（采样索引）"
        xlabel = str(header.get("display_xlabel") or xlabel)
        ylabel = str(header.get("display_ylabel") or ylabel)
        return {"extent": extent, "xlabel": xlabel, "ylabel": ylabel}

    def _normalize_compare_visual_data(self, data: np.ndarray) -> np.ndarray:
        """Return independently balanced display-only data for visual step comparison.

        Processing-lineage comparison often juxtaposes raw, background-suppressed,
        AGC, or other amplitude-altered stages.  A shared absolute colour range can
        make low-energy stages look like a flat grey block.  For the stepper compare
        UI we therefore use per-panel robust symmetric normalization, explicitly as
        display-only visualization.  Numeric/Evidence metrics should use the stored
        arrays, not this normalized rendering.
        """
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            return arr
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        try:
            high = 99.5
            if getattr(self, "page_advanced", None) is not None and self.page_advanced.percentile_var.isChecked():
                high = float(self._parse_float_edit(self.page_advanced.p_high_edit, default=99.0))
            scale = float(np.percentile(np.abs(finite), max(50.0, min(99.99, high))))
        except Exception:
            scale = float(np.max(np.abs(finite))) if finite.size else 1.0
        if not np.isfinite(scale) or scale <= 1.0e-12:
            scale = float(np.max(np.abs(finite))) if finite.size else 1.0
        if not np.isfinite(scale) or scale <= 1.0e-12:
            return np.zeros_like(arr, dtype=np.float32)
        balanced = np.nan_to_num(arr / scale, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.clip(balanced, -1.0, 1.0).astype(np.float32, copy=False)

    def _get_lineage_compare_pairs_for_plot(
        self,
        display_data: np.ndarray,
        *,
        balanced_visual: bool = False,
    ):
        """Return display-only processing-lineage compare pairs when active."""
        controller = getattr(self, "processing_lineage_controller", None)
        if controller is None:
            return None
        mode = getattr(controller, "get_active_compare_mode", lambda: None)()
        if mode not in {"grid", "diff", "slider"}:
            return None
        snapshots = controller.get_selected_compare_snapshots()
        if len(snapshots) < 2:
            return None
        pairs = []
        for snap in snapshots:
            label = str(snap.get("label") or "链路步骤")
            data = snap.get("data")
            if data is None:
                continue
            try:
                prepared = self._prepare_view_data(
                    data,
                    header_info_override=snap.get("header_info"),
                    trace_metadata_override=snap.get("trace_metadata"),
                )[0]
            except Exception:
                prepared = np.asarray(data)
            if balanced_visual:
                prepared = self._normalize_compare_visual_data(prepared)
            pairs.append((label, prepared))
        return pairs or None

    def _build_compare_data_pairs(
        self, display_data: np.ndarray, header_info_override: dict | None = None
    ):
        """构建对比数据对（复用已处理的 display_data，避免重复 _prepare_view_data）。"""
        lineage_mode = None
        controller = getattr(self, "processing_lineage_controller", None)
        if controller is not None:
            lineage_mode = controller.get_active_compare_mode()
        if lineage_mode in {"grid", "diff"}:
            lineage_pairs = self._get_lineage_compare_pairs_for_plot(
                display_data, balanced_visual=(lineage_mode == "grid")
            )
            if lineage_pairs and lineage_mode == "diff" and len(lineage_pairs) >= 2:
                left_label, left_data = lineage_pairs[0]
                right_label, right_data = lineage_pairs[1]
                min_rows = min(left_data.shape[0], right_data.shape[0])
                min_cols = min(left_data.shape[1], right_data.shape[1])
                diff = np.abs(left_data[:min_rows, :min_cols] - right_data[:min_rows, :min_cols])
                return [(f"|{left_label} - {right_label}|", diff)]
            if lineage_pairs:
                return lineage_pairs

        if not self.page_advanced.compare_var.isChecked():
            return [(self._get_single_plot_title(header_info_override), display_data)]
        left_label = self.page_advanced.compare_left_combo.currentText()
        right_label = self.page_advanced.compare_right_combo.currentText()

        def _get_prepared(label):
            if label == "当前":
                return display_data
            snap = next(
                (s for s in self.compare_snapshots if s["label"] == label), None
            )
            if snap and snap["data"] is not None:
                return self._prepare_view_data(
                    snap["data"],
                    header_info_override=snap.get("header_info"),
                    trace_metadata_override=snap.get("trace_metadata"),
                )[0]
            return display_data

        left_data = _get_prepared(left_label)
        right_data = _get_prepared(right_label)

        if self.page_advanced.diff_var.isChecked():
            # 差异视图：对齐尺寸后取绝对差
            min_rows = min(left_data.shape[0], right_data.shape[0])
            min_cols = min(left_data.shape[1], right_data.shape[1])
            diff = np.abs(
                left_data[:min_rows, :min_cols] - right_data[:min_rows, :min_cols]
            )
            return [(f"|{left_label} - {right_label}|", diff)]

        if left_label == right_label:
            return [(left_label, left_data)]

        return [(left_label, left_data), (right_label, right_data)]

    def _build_slider_compare_pair(
        self, display_data: np.ndarray, header_info_override: dict | None = None
    ):
        """构建滑动对比所需的左右数据。"""
        controller = getattr(self, "processing_lineage_controller", None)
        if controller is not None and controller.get_active_compare_mode() == "slider":
            lineage_pairs = self._get_lineage_compare_pairs_for_plot(
                display_data, balanced_visual=True
            )
            if lineage_pairs and len(lineage_pairs) >= 2:
                left_label, left_data = lineage_pairs[0]
                right_label, right_data = lineage_pairs[1]
                return left_label, left_data, right_label, right_data

        left_label = self.page_advanced.compare_left_combo.currentText() or "原始"
        right_label = self.page_advanced.compare_right_combo.currentText() or "当前"

        def _get_prepared(label):
            if label == "当前":
                return display_data
            snap = next(
                (s for s in self.compare_snapshots if s["label"] == label), None
            )
            if snap and snap["data"] is not None:
                return self._prepare_view_data(
                    snap["data"],
                    header_info_override=snap.get("header_info"),
                    trace_metadata_override=snap.get("trace_metadata"),
                )[0]
            return display_data

        left_data = _get_prepared(left_label)
        right_data = _get_prepared(right_label)
        return left_label, left_data, right_label, right_data

    def _create_plot_axes(self, n_panels: int):
        """创建绘图坐标轴。2 图横排，3–4 图使用轻量网格。"""
        if n_panels == 1:
            return [self.fig.add_subplot(111)]
        if n_panels == 2:
            return [self.fig.add_subplot(1, 2, i + 1) for i in range(2)]
        rows = 2
        cols = int(np.ceil(n_panels / rows))
        return [self.fig.add_subplot(rows, cols, i + 1) for i in range(n_panels)]

    def _get_or_create_plot_axes(self, n_panels: int):
        """获取或创建绘图坐标轴（复用已有）"""
        existing = self.fig.axes
        if len(existing) == n_panels:
            return existing
        return self._create_plot_axes(n_panels)

    def _clear_axes_artists(self, axes):
        """清除坐标轴上的艺术家对象"""
        for ax in axes:
            ax.cla()
            ax.set_title("B-scan")

    def _render_data_pairs(
        self,
        axes,
        data_pairs,
        cmap,
        extent,
        plot_config,
        header_info_override: dict | None = None,
    ):
        """渲染数据对（对比模式下统一色标）"""
        is_compare = len(data_pairs) > 1
        lineage_mode = None
        controller = getattr(self, "processing_lineage_controller", None)
        if controller is not None:
            lineage_mode = controller.get_active_compare_mode()
        if is_compare and lineage_mode == "grid":
            # Grid compare uses per-panel balanced display-only data.  Keep a
            # fixed common [-1, 1] range so every selected step remains visible.
            shared_vmin, shared_vmax = -1.0, 1.0
        elif is_compare:
            shared_vmin, shared_vmax = self._compute_shared_vmin_vmax(
                [data for _, data in data_pairs],
                header_info_override=header_info_override,
            )
        else:
            shared_vmin, shared_vmax = None, None

        last_im = None
        for ax, (label, data) in zip(axes, data_pairs):
            im = self._render_single_panel(
                ax,
                data,
                cmap,
                extent,
                plot_config,
                label,
                vmin_override=shared_vmin,
                vmax_override=shared_vmax,
                header_info_override=header_info_override,
            )
            if im:
                last_im = im
        return last_im

    def _render_wiggle_pairs(self, axes, data_pairs, axis_info: dict, plot_config: dict):
        """以摆动图形式渲染数据对。"""
        for ax, (label, data) in zip(axes, data_pairs):
            self._render_wiggle_panel(ax, data, label, axis_info, plot_config)
        return None

    def _render_wiggle_panel(
        self, ax, data: np.ndarray, title: str, axis_info: dict, plot_config: dict
    ):
        """渲染单个摆动图面板。"""
        from core.theme_manager import get_theme_manager

        ax.clear()
        ax.set_axis_on()

        if data.ndim != 2 or data.size == 0:
            placeholder = "#b7bcc6" if get_theme_manager().get_current_theme() == "dark" else "#888"
            ax.text(
                0.5,
                0.5,
                "摆动图需要二维数据",
                ha="center",
                va="center",
                fontsize=12,
                color=placeholder,
            )
            return

        x_axis = np.asarray(axis_info.get("trace_axis", []), dtype=np.float32)
        y_axis = np.asarray(axis_info.get("time_axis", []), dtype=np.float32)
        if x_axis.size != data.shape[1]:
            x_axis = np.arange(data.shape[1], dtype=np.float32)
        if y_axis.size != data.shape[0]:
            y_axis = np.arange(data.shape[0], dtype=np.float32)

        n_samples, n_traces = data.shape
        max_traces = 80
        step = max(1, int(np.ceil(n_traces / max_traces)))
        trace_indices = np.arange(0, n_traces, step, dtype=int)
        if hasattr(self, "page_advanced") and self.page_advanced is not None:
            try:
                self.page_advanced.update_wiggle_sampling_hint(
                    shown_traces=int(trace_indices.size),
                    total_traces=int(n_traces),
                )
            except Exception:
                pass

        finite_values = np.asarray(data[np.isfinite(data)], dtype=float)
        amp_ref = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
        if amp_ref <= 0:
            amp_ref = 1.0

        spacing = float(np.median(np.diff(x_axis))) if x_axis.size > 1 else 1.0
        spacing = spacing if spacing > 0 else 1.0
        wiggle_scale = spacing * 0.45

        theme = get_theme_manager().get_current_theme()
        line_color = "#f5f5f5" if theme == "dark" else "#111111"
        fill_color = "#8fb7ff" if theme == "dark" else "#4a4a4a"

        for trace_idx in trace_indices:
            trace = np.asarray(data[:, trace_idx], dtype=float)
            trace = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0)
            wiggle = x_axis[trace_idx] + (trace / amp_ref) * wiggle_scale
            ax.plot(wiggle, y_axis, color=line_color, linewidth=0.8)
            ax.fill_betweenx(
                y_axis,
                x_axis[trace_idx],
                wiggle,
                where=wiggle >= x_axis[trace_idx],
                color=fill_color,
                alpha=0.25,
                interpolate=True,
            )

        ax.set_title(f"{title} - 摆动图")
        ax.set_xlabel(plot_config["xlabel"])
        ax.set_ylabel(plot_config["ylabel"])
        if x_axis.size > 0:
            ax.set_xlim(x_axis[0] - spacing * 0.5, x_axis[-1] + spacing * 0.5)
        if y_axis.size > 0:
            ax.set_ylim(y_axis[-1], y_axis[0])
        ax.grid(False)

    def _render_slider_compare_panel(
        self,
        ax,
        display_data: np.ndarray,
        axis_info: dict,
        plot_config: dict,
        cmap,
        header_info_override: dict | None = None,
    ):
        """渲染主界面的滑动对比图。"""
        from core.theme_manager import get_theme_manager

        left_label, left_data, right_label, right_data = self._build_slider_compare_pair(
            display_data, header_info_override=header_info_override
        )

        try:
            left_data = np.asarray(left_data, dtype=np.float32)
            right_data = np.asarray(right_data, dtype=np.float32)
        except Exception:
            return None

        if left_data.ndim != 2 or right_data.ndim != 2:
            return None

        min_rows = min(left_data.shape[0], right_data.shape[0])
        min_cols = min(left_data.shape[1], right_data.shape[1])
        left_data = left_data[:min_rows, :min_cols]
        right_data = right_data[:min_rows, :min_cols]

        split_idx = int(round(self._main_slider_compare_ratio * max(min_cols - 1, 1)))
        split_idx = max(0, min(split_idx, min_cols - 1))

        controller = getattr(self, "processing_lineage_controller", None)
        lineage_slider = bool(controller is not None and controller.get_active_compare_mode() == "slider")
        if lineage_slider:
            # The lineage slider receives already-normalized display-only panels.
            # Fixed limits prevent one high-energy processing stage from flattening
            # the other side into an unreadable grey block.
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = self._compute_shared_vmin_vmax(
                [left_data, right_data],
                header_info_override=header_info_override,
            )

        # Render slider compare as two stacked images and reveal the left image
        # with a lightweight clip rectangle.  Dragging then only moves the clip
        # boundary and divider artists instead of rewriting a 2-D merged image
        # on every mouse-move event.
        im_right = ax.imshow(
            right_data,
            cmap=cmap,
            aspect="auto",
            extent=plot_config["extent"],
            vmin=vmin,
            vmax=vmax,
        )
        im_left = ax.imshow(
            left_data,
            cmap=cmap,
            aspect="auto",
            extent=plot_config["extent"],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(plot_config["xlabel"])
        ax.set_ylabel(plot_config["ylabel"])
        if lineage_slider:
            ax.set_title(f"滑动对比 · 视觉均衡 ({left_label} | {right_label})")
        else:
            ax.set_title(f"滑动对比 ({left_label} | {right_label})")

        trace_axis = np.asarray(axis_info.get("trace_axis", []), dtype=np.float32)
        if trace_axis.size == min_cols:
            split_x = float(trace_axis[split_idx])
        else:
            extent = plot_config["extent"]
            split_x = float(extent[0] + (extent[1] - extent[0]) * self._main_slider_compare_ratio)

        x0, x1, y0, y1 = plot_config["extent"]
        clip_x0 = min(float(x0), float(x1))
        clip_x1 = max(float(x0), float(x1))
        clip_y0 = min(float(y0), float(y1))
        clip_y1 = max(float(y0), float(y1))
        clip_width = max(0.0, min(float(split_x), clip_x1) - clip_x0)
        clip_patch = Rectangle(
            (clip_x0, clip_y0),
            clip_width,
            max(1.0e-9, clip_y1 - clip_y0),
            transform=ax.transData,
            visible=False,
        )
        ax.add_patch(clip_patch)
        im_left.set_clip_path(clip_patch)

        theme = get_theme_manager().get_current_theme()
        is_dark = theme == "dark"
        divider_color = "#d9e6ff" if is_dark else "#ffffff"
        label_text_color = "#f5f5f5" if is_dark else "#ffffff"
        label_bg_color = "#111318" if is_dark else "#000000"
        label_y = min(y0, y1) + abs(y1 - y0) * 0.08
        left_x = x0 + abs(x1 - x0) * 0.15
        right_x = x0 + abs(x1 - x0) * 0.85

        handle_line = ax.axvline(x=split_x, color=label_bg_color, linewidth=4.0, alpha=0.18)
        divider_line = ax.axvline(x=split_x, color=divider_color, linewidth=1.7, alpha=0.90)
        left_text = ax.text(
            left_x,
            label_y,
            left_label,
            color=label_text_color,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=label_bg_color, edgecolor="none", alpha=0.52),
        )
        right_text = ax.text(
            right_x,
            label_y,
            right_label,
            color=label_text_color,
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=label_bg_color, edgecolor="none", alpha=0.52),
        )
        self._slider_compare_render_cache = {
            "image": im_right,
            "left_image": im_left,
            "right_image": im_right,
            "ax": ax,
            "left_data": left_data,
            "right_data": right_data,
            "left_clip_patch": clip_patch,
            "clip_x0": clip_x0,
            "clip_x1": clip_x1,
            "clip_y0": clip_y0,
            "clip_y1": clip_y1,
            "split_idx": split_idx,
            "min_rows": min_rows,
            "min_cols": min_cols,
            "trace_axis": trace_axis.copy() if trace_axis.size else np.array([], dtype=np.float32),
            "extent": list(plot_config["extent"]),
            "divider_line": divider_line,
            "handle_line": handle_line,
            "left_text": left_text,
            "right_text": right_text,
            "vmin": vmin,
            "vmax": vmax,
            "lineage_slider": lineage_slider,
        }
        ax.grid(False)
        return im_right

    def _slider_compare_split_x_from_cache(self, cache: dict, split_idx: int) -> float:
        """Return the divider x-position for a cached slider-compare render."""
        trace_axis = np.asarray(cache.get("trace_axis", []), dtype=np.float32)
        if trace_axis.size and 0 <= int(split_idx) < trace_axis.size:
            return float(trace_axis[int(split_idx)])
        extent = cache.get("extent") or self._last_plot_extent or [0, 1, 1, 0]
        try:
            x0, x1 = float(extent[0]), float(extent[1])
            return float(x0 + (x1 - x0) * float(self._main_slider_compare_ratio))
        except Exception:
            return float(split_idx)

    def _try_update_slider_compare_lightweight(self, ratio: float, *, force: bool = False) -> bool:
        """Move the main slider-compare divider without rebuilding the whole figure.

        Dragging previously called ``_refresh_plot()``, which rebuilt axes,
        labels, colorbar, compare data and the processing-lineage bar.  That is
        correct for a mode switch but too heavy for mouse-motion events.  This
        method updates only the cached image buffer and divider artist; if the
        cache is stale or missing, callers can fall back to the normal refresh.
        """
        if not self._is_main_slider_compare_active():
            return False
        cache = getattr(self, "_slider_compare_render_cache", None) or {}
        image = cache.get("image")
        left_data = cache.get("left_data")
        right_data = cache.get("right_data")
        if image is None or left_data is None or right_data is None:
            return False
        try:
            ratio = max(0.0, min(1.0, float(ratio)))
            min_cols = int(cache.get("min_cols") or min(left_data.shape[1], right_data.shape[1]))
            if min_cols <= 0:
                return False
            split_idx = int(round(ratio * max(min_cols - 1, 1)))
            split_idx = max(0, min(split_idx, min_cols - 1))
            if (not force) and split_idx == int(cache.get("split_idx", -1)):
                self._main_slider_compare_ratio = ratio
                return True

            self._main_slider_compare_ratio = ratio
            cache["split_idx"] = split_idx

            split_x = self._slider_compare_split_x_from_cache(cache, split_idx)
            clip_patch = cache.get("left_clip_patch")
            if clip_patch is not None:
                clip_x0 = float(cache.get("clip_x0", min((cache.get("extent") or [0, 1])[0:2])))
                clip_x1 = float(cache.get("clip_x1", max((cache.get("extent") or [0, 1])[0:2])))
                clip_y0 = float(cache.get("clip_y0", min((cache.get("extent") or [0, 1, 1, 0])[2:4])))
                clip_y1 = float(cache.get("clip_y1", max((cache.get("extent") or [0, 1, 1, 0])[2:4])))
                clip_patch.set_bounds(
                    clip_x0,
                    clip_y0,
                    max(0.0, min(float(split_x), clip_x1) - clip_x0),
                    max(1.0e-9, clip_y1 - clip_y0),
                )
            else:
                # Backward-compatible fallback for stale caches from older renders.
                merged = cache.get("merged_data")
                if merged is None or getattr(merged, "shape", None) != getattr(right_data, "shape", None):
                    merged = np.array(right_data, copy=True)
                    previous_split = -1
                else:
                    previous_split = int(cache.get("split_idx", -1))
                if previous_split < 0:
                    merged[:, : split_idx + 1] = left_data[:, : split_idx + 1]
                elif split_idx > previous_split:
                    merged[:, previous_split + 1 : split_idx + 1] = left_data[:, previous_split + 1 : split_idx + 1]
                elif split_idx < previous_split:
                    merged[:, split_idx + 1 : previous_split + 1] = right_data[:, split_idx + 1 : previous_split + 1]
                cache["merged_data"] = merged
                image.set_data(merged)
            divider_line = cache.get("divider_line")
            if divider_line is not None:
                divider_line.set_xdata([split_x, split_x])
            handle_line = cache.get("handle_line")
            if handle_line is not None:
                handle_line.set_xdata([split_x, split_x])
            self._slider_compare_render_cache = cache
            self._request_main_canvas_draw("slider_compare")
            return True
        except Exception:
            logger.debug("Lightweight slider-compare update failed", exc_info=True)
            return False

    def _clear_hover_crosshair_artists(self, draw: bool = True):
        return self.bscan_interaction_controller.clear_hover_crosshair_artists(draw=draw)

    def _update_hover_crosshair(self, event):
        return self.bscan_interaction_controller.update_hover_crosshair(event)

    def _clear_selected_trace_marker_artists(self):
        return self.bscan_interaction_controller.clear_selected_trace_marker_artists()

    def _selected_trace_x_position(self) -> float | None:
        return self.bscan_interaction_controller.selected_trace_x_position()

    def _refresh_selected_trace_marker_lightweight(self) -> bool:
        return self.bscan_interaction_controller.refresh_selected_trace_marker_lightweight()

    def _draw_selected_trace_marker(self, axes, axis_info: dict):
        return self.bscan_interaction_controller.draw_selected_trace_marker(axes, axis_info)

    def _draw_manual_roi_marker(self, axes, axis_info: dict):
        return self.bscan_interaction_controller.draw_manual_roi_marker(axes, axis_info)

    def _render_single_panel(
        self,
        ax,
        data,
        cmap,
        extent,
        plot_config,
        title,
        vmin_override=None,
        vmax_override=None,
        header_info_override: dict | None = None,
    ):
        """渲染单个面板"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        if vmin_override is not None and vmax_override is not None:
            vmin, vmax = vmin_override, vmax_override
        else:
            vmin, vmax = self._compute_vmin_vmax(
                data, header_info_override=header_info_override
            )
        render_data = data
        render_cmap = cmap
        bad_color = header.get("display_bad_color")
        if bad_color:
            render_data = np.ma.masked_invalid(np.asarray(data, dtype=np.float64))
            render_cmap = (
                plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
            )
            render_cmap.set_bad(str(bad_color))
            ax.set_facecolor(str(bad_color))
        im = ax.imshow(
            render_data,
            cmap=render_cmap,
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel(plot_config["xlabel"])
        ax.set_ylabel(plot_config["ylabel"])
        ax.set_title(title, pad=4, fontsize=12, fontweight="semibold")
        ax.grid(False)
        return im

    def _build_vmin_vmax_cache_key(self, data: np.ndarray, header: dict | None) -> tuple:
        arr = np.asarray(data)
        try:
            ptr = int(arr.__array_interface__.get("data", (0,))[0])
        except Exception:
            ptr = id(arr)
        return (
            "vmin_vmax",
            int(getattr(self, "_data_revision", 0)),
            int(getattr(self, "_display_override_revision", 0)),
            ptr,
            tuple(arr.shape),
            str(arr.dtype),
            self._display_cache_header_key(header),
            bool(self.page_advanced.percentile_var.isChecked()),
            self._parse_float_edit(self.page_advanced.p_low_edit, default=1.0),
            self._parse_float_edit(self.page_advanced.p_high_edit, default=99.0),
            bool(self.page_advanced.auto_contrast_var.isChecked()),
            bool(self.page_advanced.symmetric_var.isChecked()),
        )

    def _compute_shared_vmin_vmax(
        self, data_list: list[np.ndarray], header_info_override: dict | None = None
    ) -> tuple[float, float]:
        """Compute display-only shared colour limits for compare panels.

        The previous compare path concatenated all finite pixels before calling
        _compute_vmin_vmax.  For normal symmetric GPR display this creates a
        large temporary copy although the exact answer is just the maximum
        absolute finite value over all panels.  Percentile/auto-contrast modes
        keep the exact concatenate path because their exact quantiles depend on
        the joint distribution.
        """
        start_ts = time.perf_counter()
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        monitor = getattr(self, "_perf_monitor", None)
        try:
            use_percentile = bool(self.page_advanced.percentile_var.isChecked())
            use_auto = bool(self.page_advanced.auto_contrast_var.isChecked())
            if use_percentile or use_auto:
                finite_parts = [
                    np.asarray(part)[np.isfinite(part)]
                    for part in data_list
                    if np.asarray(part).size and np.isfinite(part).any()
                ]
                if finite_parts:
                    result = self._compute_vmin_vmax(
                        np.concatenate(finite_parts),
                        header_info_override=header_info_override,
                    )
                else:
                    result = (-1.0, 1.0)
                if monitor is not None:
                    monitor.record("display.compare_shared_vmin_vmax_exact_ms", (time.perf_counter() - start_ts) * 1000.0)
                return result

            if header.get("display_fixed_unit_range"):
                result = (-1.0, 1.0)
            else:
                max_abs = 0.0
                for part in data_list:
                    arr = np.asarray(part, dtype=np.float32)
                    if arr.size == 0:
                        continue
                    finite = arr[np.isfinite(arr)]
                    if finite.size:
                        value = float(np.max(np.abs(finite)))
                        if np.isfinite(value):
                            max_abs = max(max_abs, value)
                if not np.isfinite(max_abs) or max_abs <= 0:
                    max_abs = 1.0
                result = (-float(max_abs), float(max_abs))
            if monitor is not None:
                monitor.record("display.compare_shared_vmin_vmax_fast_ms", (time.perf_counter() - start_ts) * 1000.0)
            return result
        except Exception:
            logger.debug("Shared compare vmin/vmax failed", exc_info=True)
            return (-1.0, 1.0)

    def _compute_vmin_vmax(
        self, data: np.ndarray, header_info_override: dict | None = None
    ):
        """计算vmin和vmax with a small display-only cache."""
        start_ts = time.perf_counter()
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        cache = getattr(self, "_vmin_vmax_cache", None)
        cache_key = None
        if cache is not None:
            try:
                cache_key = self._build_vmin_vmax_cache_key(data, header)
                cached = cache.get(cache_key)
                if cached is not None:
                    self._last_vmin_vmax_cache_hit = True
                    monitor = getattr(self, "_perf_monitor", None)
                    if monitor is not None:
                        monitor.record("display.vmin_vmax_cache_hit_ms", (time.perf_counter() - start_ts) * 1000.0)
                    return cached
            except Exception:
                cache_key = None
                logger.debug("vmin/vmax cache lookup failed", exc_info=True)

        self._last_vmin_vmax_cache_hit = False
        finite_data = np.asarray(data, dtype=np.float32)
        finite_data = finite_data[np.isfinite(finite_data)]
        if finite_data.size == 0:
            result = (-1.0, 1.0)
        elif header.get("display_fixed_unit_range"):
            result = (-1.0, 1.0)
        elif self.page_advanced.percentile_var.isChecked():
            p_low = self._parse_float_edit(self.page_advanced.p_low_edit, default=1.0)
            p_high = self._parse_float_edit(
                self.page_advanced.p_high_edit, default=99.0
            )
            vmin, vmax = np.percentile(finite_data, [p_low, p_high])
            result = (float(vmin), float(vmax))
        elif self.page_advanced.auto_contrast_var.isChecked():
            vmin, vmax = np.percentile(finite_data, [0.5, 99.5])
            result = (float(vmin), float(vmax))
        elif header.get("display_center_zero"):
            percentile_high = header.get("display_percentile_abs_high")
            if percentile_high is not None:
                vmax = float(np.percentile(np.abs(finite_data), float(percentile_high)))
            else:
                vmax = float(np.max(np.abs(finite_data))) if finite_data.size else 1.0
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            result = (-float(vmax), float(vmax))
        else:
            vmax = float(np.max(np.abs(finite_data)))
            vmin = (
                -vmax
                if self.page_advanced.symmetric_var.isChecked()
                else float(np.min(finite_data))
            )
            result = (float(vmin), float(vmax))
        monitor = getattr(self, "_perf_monitor", None)
        if monitor is not None:
            monitor.record("display.vmin_vmax_ms", (time.perf_counter() - start_ts) * 1000.0)
        if cache is not None and cache_key is not None:
            self._limited_cache_set(cache, cache_key, result, max_entries=16)
        return result

    def _draw_colorbar_if_needed(
        self, im, axes, header_info_override: dict | None = None
    ):
        """根据需要绘制色标"""
        header = (
            header_info_override
            if header_info_override is not None
            else (self.header_info or {})
        )
        if not self.page_advanced.show_cbar_var.isChecked() and not header.get(
            "display_show_cbar"
        ):
            return
        if len(axes) == 1:
            self.cbar = self.fig.colorbar(im, ax=axes[0])
        else:
            self.cbar = self.fig.colorbar(im, ax=axes)
        label = header.get("display_colorbar_label")
        if label:
            self.cbar.set_label(str(label))
