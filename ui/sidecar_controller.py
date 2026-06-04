# -*- coding: utf-8 -*-
"""Optional UAV sidecar selection and loading helpers.

This controller keeps RTK/IMU/altimeter sidecar discovery, manifest reading, and
optional fallback behaviour outside the main window shell.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ui.gui_base import detect_csv_header, _detect_skiprows
from core.gpr_io import extract_airborne_csv_payload

BASE_DIR = str(Path(__file__).resolve().parents[1])
logger = logging.getLogger(__name__)


class SidecarController:
    """Manage optional RTK/IMU/altimeter sidecars for a host window."""

    def __init__(self, host):
        self.host = host

    def __getattr__(self, name):
        return getattr(self.host, name)

    def _pick_sidecar_file(self, kind: str):
        """选择可选 RTK/IMU/高度计 sidecar 文件；取消选择时保留原状态。"""
        if kind not in {"rtk", "imu", "altimeter"}:
            raise ValueError(f"不支持的 sidecar 类型: {kind}")

        labels = {"rtk": "RTK", "imu": "IMU", "altimeter": "高度计"}
        label = labels[kind]
        current_path = self._sidecar_files.get(kind)
        initial_dir = os.path.dirname(current_path) if current_path else BASE_DIR
        path, _ = QFileDialog.getOpenFileName(
            self.host,
            f"选择 {label} sidecar 文件",
            initial_dir,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        self._set_sidecar_file(kind, path)

    def _set_sidecar_file(self, kind: str, path=None) -> None:
        """更新单个 sidecar 路径，并同步高级设置页标签。"""
        if kind not in {"rtk", "imu", "altimeter"}:
            raise ValueError(f"不支持的 sidecar 类型: {kind}")

        normalized = str(path) if path else None
        self._sidecar_files[kind] = normalized
        display = os.path.basename(normalized) if normalized else "未选择"
        # V0.8.42: visible sidecar controls live on the Space page.
        # The display page keeps hidden compatibility labels only.
        for page_name in ("page_terrain3d", "page_advanced"):
            page = getattr(self, page_name, None)
            label_widget = getattr(page, f"{kind}_sidecar_label", None) if page is not None else None
            if label_widget is not None:
                label_widget.setText(display)
                label_widget.setToolTip(normalized or "未选择")
        self._log(f"{kind.upper()} sidecar：{display}")

    def _clear_sidecar_file(self, kind: str) -> None:
        """仅清除指定 RTK/IMU/高度计 sidecar 选择。"""
        self._set_sidecar_file(kind, None)

    def _warn_sidecar_ignored(self, kind: str, reason: str) -> None:
        """提示用户已忽略可选 sidecar，同时保持 CSV 正常加载。"""
        message = f"已忽略可选 {kind} 辅助文件，CSV 将按普通数据继续加载。\n原因：{reason}"
        self._log(message.replace("\n", " "))
        if hasattr(self, "status_label"):
            self.status_label.setText(f"已忽略可选 {kind} 辅助文件")
        QMessageBox.warning(self.host, "可选辅助文件已忽略", message)

    def _store_trace_timestamps_from_metadata(self, trace_metadata) -> None:
        """从既有每道元数据同步 trace 时间戳缓存；不在 GUI 层推导。"""
        timestamps = None
        if isinstance(trace_metadata, dict):
            timestamps = trace_metadata.get("trace_timestamp_s")
        if timestamps is None:
            self._trace_timestamps_s = None
            return
        self._trace_timestamps_s = np.asarray(timestamps, dtype=np.float64).copy()

    def _get_trace_timestamps_for_sidecars(self):
        """仅返回当前会话已存在的 trace 时间戳，不在 GUI 层推导。"""
        self._store_trace_timestamps_from_metadata(self.trace_metadata)
        return getattr(self, "_trace_timestamps_s", None)

    def _read_csv_sidecar_manifest(self, path: str) -> dict | None:
        """Read a same-folder manifest when it describes the selected CSV."""
        try:
            csv_path = Path(path).resolve()
            manifest_path = csv_path.parent / "manifest.json"
            if not manifest_path.exists():
                return None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidate_names = [
                manifest.get("primary_data_file"),
                manifest.get("data_file"),
                manifest.get("main_csv"),
            ]
            sidecars = manifest.get("sidecars") if isinstance(manifest.get("sidecars"), dict) else {}
            candidate_names.extend(
                value for value in sidecars.values() if isinstance(value, str) and value.endswith(".csv")
            )
            explicit_data = [name for name in candidate_names if isinstance(name, str)]
            if explicit_data:
                data_matches = any((csv_path.parent / name).resolve() == csv_path for name in explicit_data)
                if not data_matches:
                    return None
            return manifest
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _merge_manifest_header_info(self, header_info, manifest: dict | None):
        """Use manifest shape/time hints for matrix CSV packages without headers."""
        merged = dict(header_info or {})
        if not isinstance(manifest, dict):
            return header_info
        if "sample_count" in manifest:
            merged.setdefault("a_scan_length", int(manifest["sample_count"]))
        if "trace_count" in manifest:
            merged.setdefault("num_traces", int(manifest["trace_count"]))
        if "total_time_ns" in manifest:
            merged.setdefault("total_time_ns", float(manifest["total_time_ns"]))
        if "trace_interval_m" in manifest:
            merged.setdefault("trace_interval_m", float(manifest["trace_interval_m"]))
        if "distance_start_m" in manifest and "distance_end_m" in manifest and "trace_count" in manifest:
            trace_count = max(int(manifest.get("trace_count") or 0), 1)
            if trace_count > 1:
                span = float(manifest["distance_end_m"]) - float(manifest["distance_start_m"])
                merged.setdefault("trace_interval_m", span / float(trace_count - 1))
        return merged or header_info

    def _manifest_sidecar_path(self, path: str, manifest: dict | None, key: str) -> str | None:
        """Resolve a sidecar path from same-folder manifest keys."""
        if not isinstance(manifest, dict):
            return None
        base = Path(path).resolve().parent
        sidecars = manifest.get("sidecars") if isinstance(manifest.get("sidecars"), dict) else {}
        candidates = [
            sidecars.get(key),
            sidecars.get(f"{key}_file"),
            manifest.get(f"{key}_file"),
        ]
        if key == "trace_timestamps":
            candidates.extend([manifest.get("trace_timestamps_file"), manifest.get("trace_timestamps")])
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate:
                continue
            resolved = (base / candidate).resolve()
            if resolved.exists():
                return str(resolved)
        return None

    def _read_trace_timestamps_sidecar(self, path: str, manifest: dict | None = None):
        """Read trace_timestamps.csv from a manifest or same CSV directory."""
        candidate = self._manifest_sidecar_path(path, manifest, "trace_timestamps")
        if candidate is None:
            default_path = Path(path).resolve().parent / "trace_timestamps.csv"
            if default_path.exists():
                candidate = str(default_path)
        if candidate is None:
            return None
        try:
            frame = pd.read_csv(candidate)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            return None
        if frame.empty:
            return None
        column = None
        for name in ("timestamp_s", "trace_timestamp_s", "time_s", "timestamp"):
            if name in frame.columns:
                column = name
                break
        if column is None:
            numeric_cols = [
                col for col in frame.columns if pd.to_numeric(frame[col], errors="coerce").notna().any()
            ]
            if not numeric_cols:
                return None
            column = numeric_cols[-1]
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        if values.size == 0 or not np.isfinite(values).all():
            return None
        return values.copy()

    def _discover_sidecar_files_for_csv(self, path: str, manifest: dict | None = None) -> dict[str, str]:
        """Discover same-folder sidecars when the dataset package includes a manifest."""
        discovered: dict[str, str] = {}
        for key in ("rtk", "imu", "altimeter"):
            candidate = self._manifest_sidecar_path(path, manifest, key)
            if candidate is None:
                default_path = Path(path).resolve().parent / f"{key}.csv"
                if default_path.exists():
                    candidate = str(default_path)
            if candidate is not None:
                discovered[key] = candidate
        return discovered

    def _read_explicit_trace_timestamps_from_csv(self, path: str):
        """从主 CSV 第 6 列读取显式 trace 时间戳；不根据采样参数推导。"""
        try:
            header_info = detect_csv_header(path)
            if not header_info:
                return None
            samples = int(header_info["a_scan_length"])
            traces = int(header_info["num_traces"])
            required_rows = samples * traces
            skip_lines = _detect_skiprows(path)
            timestamp_col = pd.read_csv(
                path,
                header=None,
                skiprows=skip_lines,
                usecols=[5],
                nrows=required_rows,
                na_filter=False,
                low_memory=False,
            ).iloc[:, 0]
        except (OSError, ValueError, KeyError, IndexError, pd.errors.ParserError):
            return None

        values = pd.to_numeric(timestamp_col, errors="coerce").to_numpy(dtype=np.float64)
        if values.size < required_rows:
            return None
        trace_timestamps_s = values.reshape((traces, samples))[:, 0]
        if not np.isfinite(trace_timestamps_s).all():
            return None
        return trace_timestamps_s.copy()

    def _is_current_data_path(self, path: str) -> bool:
        """Return True only when `path` is the already loaded data source."""
        current_path = getattr(self, "data_path", None)
        if not current_path:
            return False
        try:
            return os.path.abspath(str(current_path)) == os.path.abspath(str(path))
        except (OSError, TypeError, ValueError):
            return False

    def _build_sidecar_loader_kwargs(self, path: str) -> dict:
        """根据当前选择构造 CSV 加载 sidecar kwargs。"""
        manifest = self._read_csv_sidecar_manifest(path)
        discovered = self._discover_sidecar_files_for_csv(path, manifest)
        rtk_path = self._sidecar_files.get("rtk") or discovered.get("rtk")
        imu_path = self._sidecar_files.get("imu") or discovered.get("imu")
        altimeter_path = self._sidecar_files.get("altimeter") or discovered.get("altimeter")
        if not rtk_path and not imu_path and not altimeter_path:
            return {}

        trace_timestamps_s = self._read_explicit_trace_timestamps_from_csv(path)
        if trace_timestamps_s is None:
            trace_timestamps_s = self._read_trace_timestamps_sidecar(path, manifest)
        if trace_timestamps_s is None and self._is_current_data_path(path):
            trace_timestamps_s = self._get_trace_timestamps_for_sidecars()
        if trace_timestamps_s is None:
            selected_sidecars = self._describe_selected_sidecars(
                {
                    "rtk_path": rtk_path,
                    "imu_path": imu_path,
                    "altimeter_path": altimeter_path,
                }
            )
            self._warn_sidecar_ignored(
                selected_sidecars,
                "缺少可用于对齐的 trace_timestamps_s；本次不会接入辅助文件。",
            )
            return {}

        kwargs = {"trace_timestamps_s": trace_timestamps_s}
        if rtk_path:
            kwargs["rtk_path"] = rtk_path
        if imu_path:
            kwargs["imu_path"] = imu_path
        if altimeter_path:
            kwargs["altimeter_path"] = altimeter_path
        return kwargs

    def _extract_airborne_payload_with_optional_sidecars(
        self, raw_data: np.ndarray, header_info, sidecar_kwargs: dict
    ):
        """提取 CSV payload；可选 sidecar 出错时警告后回退为普通 CSV。"""
        try:
            return extract_airborne_csv_payload(
                raw_data, header_info, **sidecar_kwargs
            )
        except ValueError as exc:
            if not sidecar_kwargs or not self._is_optional_sidecar_error(exc):
                raise
            self._warn_sidecar_ignored(
                self._describe_selected_sidecars(sidecar_kwargs), str(exc)
            )
            return extract_airborne_csv_payload(raw_data, header_info)

    def _describe_selected_sidecars(self, sidecar_kwargs: dict) -> str:
        """返回用于告警文案的已选 sidecar 类型。"""
        selected = []
        if sidecar_kwargs.get("rtk_path"):
            selected.append("RTK")
        if sidecar_kwargs.get("imu_path"):
            selected.append("IMU")
        if sidecar_kwargs.get("altimeter_path"):
            selected.append("高度计")
        return "/".join(selected) or "RTK/IMU/高度计"

    def _is_optional_sidecar_error(self, exc: ValueError) -> bool:
        """判断 ValueError 是否来自可选 sidecar 解析/对齐链。"""
        text = str(exc).lower()
        markers = (
            "sidecar",
            "trace_timestamps_s",
            "rtk",
            "imu",
            "altimeter",
            "height_agl_m",
            "height_source",
            "distance_m",
            "target_count",
            "timestamp_s",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "parser",
            "parse",
            "optional sidecar integration",
            "unsupported sidecar",
        )
        if any(marker in text for marker in markers):
            return True
        if ("longitude" in text or "latitude" in text) and (
            "sidecar" in text or "rtk" in text or "timestamp" in text
        ):
            return True
        return False
