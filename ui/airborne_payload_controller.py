# -*- coding: utf-8 -*-
"""Airborne/QC/spatial payload builders for the main MyGPR window.

The methods here used to live directly in ``GPRGuiQt``.  They still operate on
the host window state, but moving them here keeps the main window focused on UI
wiring and command orchestration.
"""

from __future__ import annotations

import logging

import numpy as np

from core.uav_georeference_3d import build_airborne_georeference_3d_payload

logger = logging.getLogger(__name__)


class AirbornePayloadController:
    """Build lightweight airborne QC, trajectory, and 3D-preview payloads."""

    def __init__(self, host):
        self.host = host

    def __getattr__(self, name):
        return getattr(self.host, name)

    def _build_airborne_metadata_summary(self) -> list[str]:
        """构建航空元数据摘要文本。"""
        info = self.header_info or {}
        if not info.get("has_airborne_metadata"):
            return []

        meta = self.trace_metadata or {}
        lines = []
        track_length = info.get("track_length_m")
        if track_length is not None:
            lines.append(f"测线长度: {float(track_length):.2f} m")
        if meta.get("longitude") is not None and meta.get("latitude") is not None:
            lon = np.asarray(meta.get("longitude"), dtype=np.float64)
            lat = np.asarray(meta.get("latitude"), dtype=np.float64)
            if lon.size > 0 and lat.size > 0:
                lines.append(
                    f"起止经纬度: ({lon[0]:.7f}, {lat[0]:.7f}) -> ({lon[-1]:.7f}, {lat[-1]:.7f})"
                )
        if info.get("ground_elevation_min_m") is not None:
            lines.append(
                "地表高程: {:.2f} ~ {:.2f} m".format(
                    float(info.get("ground_elevation_min_m", 0.0)),
                    float(info.get("ground_elevation_max_m", 0.0)),
                )
            )
        if info.get("flight_height_min_m") is not None:
            lines.append(
                "飞行高度: {:.2f} ~ {:.2f} m".format(
                    float(info.get("flight_height_min_m", 0.0)),
                    float(info.get("flight_height_max_m", 0.0)),
                )
            )
        if info.get("trace_timestamp_min_s") is not None:
            lines.append(
                "轨迹时间戳: {:.3f} ~ {:.3f} s".format(
                    float(info.get("trace_timestamp_min_s", 0.0)),
                    float(info.get("trace_timestamp_max_s", 0.0)),
                )
            )
        if info.get("alignment_extrapolated_trace_count") is not None:
            total_traces = int(info.get("num_traces", 0) or 0)
            extrapolated = int(info.get("alignment_extrapolated_trace_count", 0) or 0)
            if total_traces > 0:
                lines.append(
                    "辅助文件对齐: {:.1%} 道在覆盖范围内".format(
                        max(0.0, 1.0 - float(info.get("alignment_extrapolated_fraction", 0.0)))
                    )
                )
            else:
                lines.append(
                    f"辅助文件对齐: 外推 {extrapolated} 道"
                )
        if info.get("height_confidence_mean") is not None:
            lines.append(
                "高度置信度: 均值 {:.2f}, 最小 {:.2f}".format(
                    float(info.get("height_confidence_mean", 0.0)),
                    float(info.get("height_confidence_min", 0.0)),
                )
            )
        if info.get("trace_interval_min_m") is not None:
            lines.append(
                "道间距: {:.3f} ~ {:.3f} m (均值 {:.3f} m)".format(
                    float(info.get("trace_interval_min_m", 0.0)),
                    float(info.get("trace_interval_max_m", 0.0)),
                    float(info.get("trace_interval_m", 0.0)),
                )
            )
        return lines

    def _build_airborne_line_summary_text(self) -> str:
        """构建测线结果卡片文本。"""
        if self.data is None:
            return "暂无测线信息"

        header = self.header_info or {}
        meta = self.trace_metadata or {}
        lines = [
            f"数据文件: {self.data_path or '--'}",
            f"矩阵尺寸: {self.data.shape[0]} × {self.data.shape[1]}",
        ]

        if header:
            lines.append(
                f"采样点数: {header.get('a_scan_length', self.data.shape[0])} | 道数: {header.get('num_traces', self.data.shape[1])}"
            )
        if header.get("has_airborne_metadata"):
            lines.append(f"测线长度: {float(header.get('track_length_m', 0.0)):.2f} m")
            lines.append(
                "道间距: {:.3f} ~ {:.3f} m (均值 {:.3f} m)".format(
                    float(header.get("trace_interval_min_m", 0.0)),
                    float(header.get("trace_interval_max_m", 0.0)),
                    float(header.get("trace_interval_m", 0.0)),
                )
            )
            lines.append(
                "地表高程: {:.2f} ~ {:.2f} m".format(
                    float(header.get("ground_elevation_min_m", 0.0)),
                    float(header.get("ground_elevation_max_m", 0.0)),
                )
            )
            lines.append(
                "飞行高度: {:.2f} ~ {:.2f} m".format(
                    float(header.get("flight_height_min_m", 0.0)),
                    float(header.get("flight_height_max_m", 0.0)),
                )
            )
            if meta.get("longitude") is not None and meta.get("latitude") is not None:
                lon = np.asarray(meta.get("longitude"), dtype=np.float64)
                lat = np.asarray(meta.get("latitude"), dtype=np.float64)
                if lon.size and lat.size:
                    lines.append(f"起点经纬度: {lon[0]:.7f}, {lat[0]:.7f}")
                    lines.append(f"终点经纬度: {lon[-1]:.7f}, {lat[-1]:.7f}")
        else:
            lines.append("航空元数据: 未提供")

        return "\n".join(lines)

    def _build_airborne_qc_summary_text(self) -> str:
        """构建航空质控摘要文本。"""
        qc = self._compute_airborne_qc_metrics()
        lines = []
        if qc:
            lines.extend(
                [
                    f"测线长度: {qc['track_length_m']:.2f} m",
                    f"道间距变异系数: {qc['trace_spacing_cv']:.3f}",
                    f"飞行高度跨度: {qc['flight_height_span_m']:.2f} m",
                    f"道间距离群点: {qc['spacing_outliers']}",
                    f"飞行高度离群点: {qc['height_outliers']}",
                ]
            )
            if qc.get("alignment_status_available"):
                lines.append(f"辅助文件外推道数: {qc['alignment_extrapolated_traces']}")
            if qc.get("height_confidence_available"):
                lines.append(f"低置信度高度计道数: {qc['height_confidence_low_traces']}")
            alerts = qc.get("alerts") or []
            lines.append("异常状态: " + (", ".join(alerts) if alerts else "正常"))

        no_prior_policy = self._build_no_prior_qc_policy(
            metrics=self._last_quality_metrics,
            airborne_qc=qc,
        )
        lines.append("")
        lines.append("无先验安全策略:")
        lines.append(f"- 等级: {no_prior_policy.get('no_prior_level', '--')}")
        lines.append(
            "- 初始策略: "
            + str(no_prior_policy.get("recommended_initial_policy", "--"))
        )
        lines.append(
            "- 人工复核: "
            + (
                "required"
                if no_prior_policy.get("manual_review_required")
                else "not_required"
            )
        )
        blocked_actions = ", ".join(no_prior_policy.get("blocked_actions", [])) or "--"
        lines.append(f"- 阻断动作: {blocked_actions}")
        claim_boundary = str(no_prior_policy.get("claim_boundary") or "--")
        lines.append(f"- Claim boundary: {claim_boundary}")
        return "\n".join(lines)

    def _build_airborne_qc_plot_payload(self) -> dict | None:
        """构建航空 QC 可视化所需数据。"""
        qc = self._compute_airborne_qc_metrics()
        meta = self.trace_metadata or {}
        if not qc or "trace_distance_m" not in meta or "flight_height_m" not in meta:
            return None

        distance = np.asarray(meta.get("trace_distance_m"), dtype=np.float64)
        flight = np.asarray(meta.get("flight_height_m"), dtype=np.float64)
        if distance.size <= 1:
            return None

        spacing = np.diff(distance)
        spacing_x = 0.5 * (distance[:-1] + distance[1:])
        spacing_mask = np.zeros_like(spacing, dtype=bool)
        for idx in qc.get("spacing_outlier_indices", []):
            if 0 <= idx < spacing_mask.size:
                spacing_mask[idx] = True

        flight_mask = np.zeros_like(flight, dtype=bool)
        for idx in qc.get("height_outlier_indices", []):
            if 0 <= idx < flight_mask.size:
                flight_mask[idx] = True

        return {
            "spacing_x": spacing_x.tolist(),
            "spacing": spacing.tolist(),
            "spacing_mask": spacing_mask.tolist(),
            "distance": distance.tolist(),
            "flight": flight.tolist(),
            "flight_mask": flight_mask.tolist(),
        }

    def _build_airborne_trajectory_plot_payload(self) -> dict | None:
        """构建航空航迹图所需数据。"""
        header = self.header_info or {}
        meta = self.trace_metadata or {}
        if not header.get("has_airborne_metadata"):
            return None

        longitude = np.asarray(meta.get("longitude", []), dtype=np.float64)
        latitude = np.asarray(meta.get("latitude", []), dtype=np.float64)
        n = min(longitude.size, latitude.size)
        if n == 0:
            return None

        longitude = longitude[:n]
        latitude = latitude[:n]
        trace_indices = np.asarray(
            meta.get("trace_index", np.arange(n)), dtype=np.int32
        )
        if trace_indices.size < n:
            trace_indices = np.arange(n, dtype=np.int32)
        else:
            trace_indices = trace_indices[:n]
        anomaly_mask = np.zeros(n, dtype=bool)
        qc = self._compute_airborne_qc_metrics() or {}
        for idx in qc.get("spacing_outlier_indices", []):
            mapped_idx = int(idx) + 1
            if 0 <= mapped_idx < n:
                anomaly_mask[mapped_idx] = True
        for idx in qc.get("height_outlier_indices", []):
            mapped_idx = int(idx)
            if 0 <= mapped_idx < n:
                anomaly_mask[mapped_idx] = True

        finite_mask = np.isfinite(longitude) & np.isfinite(latitude)
        if not np.any(finite_mask):
            return None

        flight_height = np.asarray(meta.get("flight_height_m", []), dtype=np.float64)[:n]

        payload = {
            "longitude": longitude[finite_mask].tolist(),
            "latitude": latitude[finite_mask].tolist(),
            "trace_indices": trace_indices[finite_mask].tolist(),
            "anomaly_mask": anomaly_mask[finite_mask].tolist(),
            "selected_trace_index": self._selected_trace_index,
        }
        if flight_height.size >= n:
            payload["flight_height_m"] = flight_height[finite_mask].tolist()
        return payload

    def _build_airborne_georeference_3d_plot_payload(self) -> dict | None:
        """构建三维地理参考预览所需的原始/当前/差异数据。"""
        current = self._build_airborne_georeference_3d_payload_for(
            self.data,
            self.header_info,
            self.trace_metadata,
        )
        raw = self._build_airborne_georeference_3d_payload_for(
            self.shared_data.original_data,
            self.shared_data.original_header_info,
            self.shared_data.original_trace_metadata,
        )
        diff_data = self._build_georeference_difference_data(
            self.shared_data.original_data,
            self.data,
        )
        diff = self._build_airborne_georeference_3d_payload_for(
            diff_data,
            self.header_info,
            self.trace_metadata,
        )
        if not any([raw, current, diff]):
            return None
        return {"raw": raw, "current": current, "diff": diff}

    def _build_airborne_georeference_3d_payload_for(
        self,
        data,
        header_info,
        trace_metadata,
    ) -> dict | None:
        """Build one lightweight 3D preview payload for a layer."""
        if data is None:
            return None
        try:
            return build_airborne_georeference_3d_payload(
                data,
                header_info,
                trace_metadata,
                selected_trace_index=self._selected_trace_index,
                max_preview_traces=240,
                max_preview_samples=160,
            )
        except Exception as exc:
            logger.warning("Failed to build airborne 3D georeference payload: %s", exc)
            return None

    def _build_georeference_difference_data(self, raw, current) -> np.ndarray | None:
        """Build current-raw data for 3D preview only."""
        if raw is None or current is None:
            return None
        raw_arr = np.asarray(raw, dtype=np.float32)
        current_arr = np.asarray(current, dtype=np.float32)
        if raw_arr.ndim != 2 or current_arr.ndim != 2:
            return None
        rows = min(raw_arr.shape[0], current_arr.shape[0])
        if rows <= 0 or current_arr.shape[1] <= 0:
            return None
        raw_trimmed = raw_arr[:rows, :]
        current_trimmed = current_arr[:rows, :]
        if raw_trimmed.shape[1] != current_trimmed.shape[1]:
            source_x = np.linspace(0.0, 1.0, raw_trimmed.shape[1], dtype=np.float32)
            target_x = np.linspace(0.0, 1.0, current_trimmed.shape[1], dtype=np.float32)
            resampled = np.empty((rows, current_trimmed.shape[1]), dtype=np.float32)
            for row_idx in range(rows):
                resampled[row_idx, :] = np.interp(
                    target_x,
                    source_x,
                    raw_trimmed[row_idx, :],
                ).astype(np.float32)
            raw_trimmed = resampled
        return (current_trimmed - raw_trimmed).astype(np.float32, copy=False)

    def _build_airborne_anomaly_details(self) -> list[dict]:
        """构建航空异常明细行。"""
        qc = self._compute_airborne_qc_metrics()
        meta = self.trace_metadata or {}
        if not qc or not meta:
            return []

        distance = np.asarray(meta.get("trace_distance_m", []), dtype=np.float64)
        longitude = np.asarray(meta.get("longitude", []), dtype=np.float64)
        latitude = np.asarray(meta.get("latitude", []), dtype=np.float64)
        flight = np.asarray(meta.get("flight_height_m", []), dtype=np.float64)
        height_confidence = np.asarray(meta.get("height_confidence", []), dtype=np.float64)
        spacing = (
            np.diff(distance) if distance.size > 1 else np.array([], dtype=np.float64)
        )
        spacing_x = (
            0.5 * (distance[:-1] + distance[1:])
            if spacing.size
            else np.array([], dtype=np.float64)
        )

        details: list[dict] = []
        for idx in qc.get("spacing_outlier_indices", []):
            if 0 <= idx < spacing.size:
                lon = (
                    float(longitude[min(idx + 1, longitude.size - 1)])
                    if longitude.size
                    else None
                )
                lat = (
                    float(latitude[min(idx + 1, latitude.size - 1)])
                    if latitude.size
                    else None
                )
                details.append(
                    {
                        "type": "trace_spacing",
                        "index": int(idx),
                        "distance_m": float(spacing_x[idx]),
                        "value": float(spacing[idx]),
                        "longitude": lon,
                        "latitude": lat,
                    }
                )

        for idx in qc.get("height_outlier_indices", []):
            if 0 <= idx < flight.size:
                lon = float(longitude[idx]) if longitude.size > idx else None
                lat = float(latitude[idx]) if latitude.size > idx else None
                details.append(
                    {
                        "type": "flight_height",
                        "index": int(idx),
                        "distance_m": float(distance[idx])
                        if distance.size > idx
                        else None,
                        "value": float(flight[idx]),
                        "longitude": lon,
                        "latitude": lat,
                    }
                )

        for idx in qc.get("alignment_extrapolated_indices", []):
            if 0 <= idx < distance.size:
                lon = float(longitude[idx]) if longitude.size > idx else None
                lat = float(latitude[idx]) if latitude.size > idx else None
                details.append(
                    {
                        "type": "sidecar_alignment",
                        "index": int(idx),
                        "distance_m": float(distance[idx])
                        if distance.size > idx
                        else None,
                        "value": None,
                        "longitude": lon,
                        "latitude": lat,
                    }
                )

        for idx in qc.get("height_confidence_low_indices", []):
            if 0 <= idx < height_confidence.size:
                lon = float(longitude[idx]) if longitude.size > idx else None
                lat = float(latitude[idx]) if latitude.size > idx else None
                details.append(
                    {
                        "type": "height_confidence",
                        "index": int(idx),
                        "distance_m": float(distance[idx])
                        if distance.size > idx
                        else None,
                        "value": float(height_confidence[idx]),
                        "longitude": lon,
                        "latitude": lat,
                    }
                )

        return details

    def _build_airborne_anomaly_text(self) -> str:
        """构建航空异常明细文本。"""
        details = self._build_airborne_anomaly_details()
        if not details:
            return "暂无异常明细"

        lines = []
        for item in details:
            if item["type"] == "trace_spacing":
                lines.append(
                    "道间距异常 | idx={} | 距离={:.2f} m | 间距={:.3f} m | lon={:.7f} | lat={:.7f}".format(
                        item["index"],
                        item["distance_m"],
                        item["value"],
                        item["longitude"] if item["longitude"] is not None else 0.0,
                        item["latitude"] if item["latitude"] is not None else 0.0,
                    )
                )
            elif item["type"] == "sidecar_alignment":
                lines.append(
                    "辅助文件外推 | idx={} | 距离={:.2f} m | lon={:.7f} | lat={:.7f}".format(
                        item["index"],
                        item["distance_m"] if item["distance_m"] is not None else 0.0,
                        item["longitude"] if item["longitude"] is not None else 0.0,
                        item["latitude"] if item["latitude"] is not None else 0.0,
                    )
                )
            elif item["type"] == "height_confidence":
                lines.append(
                    "高度置信度低 | idx={} | 距离={:.2f} m | 值={:.2f} | lon={:.7f} | lat={:.7f}".format(
                        item["index"],
                        item["distance_m"] if item["distance_m"] is not None else 0.0,
                        item["value"],
                        item["longitude"] if item["longitude"] is not None else 0.0,
                        item["latitude"] if item["latitude"] is not None else 0.0,
                    )
                )
            else:
                lines.append(
                    "飞行高度异常 | idx={} | 距离={:.2f} m | 高度={:.2f} m | lon={:.7f} | lat={:.7f}".format(
                        item["index"],
                        item["distance_m"] if item["distance_m"] is not None else 0.0,
                        item["value"],
                        item["longitude"] if item["longitude"] is not None else 0.0,
                        item["latitude"] if item["latitude"] is not None else 0.0,
                    )
                )
        return "\n".join(lines)
