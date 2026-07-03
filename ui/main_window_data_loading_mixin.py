# -*- coding: utf-8 -*-
"""Data loading helpers for app_qt.GPRGuiQt."""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMessageBox

from core.app_errors import InputDataError
from core.app_runtime import _save_last_data_path
from core.gpr_io import auto_load_data, extract_airborne_csv_payload, read_ascans_folder
from core.io_performance import (
    choose_csv_read_dtype,
    csv_import_context,
    sanitize_float32_matrix,
    summarize_array_memory,
)
from core.runtime_warnings import build_runtime_warning
from ui.gui_base import (
    _detect_skiprows,
    build_csv_load_error_message,
    detect_csv_header,
)

logger = logging.getLogger(__name__)


class MainWindowDataLoadingMixin:

    def _load_common_gpr_file_with_progress(self, path, progress_callback=None):
        """Load a common GPR profile file through the central format registry."""
        load_t0 = time.perf_counter()
        try:
            if progress_callback:
                progress_callback(10, "正在识别 GPR 数据格式...")
            result = auto_load_data(path)
            if "data" not in result:
                raise ValueError("该文件只包含配置/头信息，未返回可显示的 B-scan 数据。")
            if progress_callback:
                progress_callback(70, "正在标准化数据矩阵...")
            data, memory_summary = sanitize_float32_matrix(result.get("data"))
            header_info = dict(result.get("header_info") or {})
            header_info.setdefault("a_scan_length", int(data.shape[0]) if data.ndim >= 1 else 0)
            header_info.setdefault("num_traces", int(data.shape[1]) if data.ndim >= 2 else 1)
            header_info.setdefault("total_time_ns", 0.0)
            header_info.setdefault("trace_interval_m", 0.0)
            header_info.setdefault("source", result.get("format") or result.get("type") or "common_gpr")
            header_info["import_memory_summary"] = dict(memory_summary)
            header_info["import_elapsed_ms"] = float((time.perf_counter() - load_t0) * 1000.0)
            trace_metadata = result.get("trace_metadata")
            runtime_warnings = list(result.get("runtime_warnings", []) or [])
            if memory_summary.get("nonfinite_replaced", 0):
                runtime_warnings.append(
                    build_runtime_warning(
                        "data_sanitized",
                        "导入 GPR 数据包含 NaN/Inf，已使用均值填充。",
                        fill_value=memory_summary.get("fill_value", 0.0),
                        path=path,
                    )
                )
            if progress_callback:
                progress_callback(100, "加载完成！")
            return {
                "data": data,
                "header_info": header_info,
                "trace_metadata": trace_metadata,
                "runtime_warnings": runtime_warnings,
                "path": path,
                "import_memory_summary": memory_summary,
            }
        except Exception as e:
            raise InputDataError(
                "GPR 数据加载失败",
                technical_detail=str(e),
                context={"path": path, "loader": "common_gpr"},
            ) from e


    def _read_csv_raw_array_for_import(
        self,
        path: str,
        *,
        skip_lines: int = 0,
        header_info: dict | None = None,
        trace_timestamps_s=None,
        rtk_path=None,
        imu_path=None,
        altimeter_path=None,
        progress_callback=None,
    ) -> tuple[np.ndarray, dict]:
        """Read CSV to a numeric ndarray with conservative dtype selection.

        Matrix-only CSV is read as float32 to avoid a transient float64 copy.
        Header/sidecar airborne data stays on pandas inference so longitude,
        latitude and timestamps keep their existing precision before extraction.
        """
        has_sidecars = any(
            value is not None
            for value in (trace_timestamps_s, rtk_path, imu_path, altimeter_path)
        )
        read_dtype = choose_csv_read_dtype(
            header_info=header_info,
            has_sidecars=has_sidecars,
        )
        context = csv_import_context(
            path,
            header_info=header_info,
            trace_timestamps_s=trace_timestamps_s,
            rtk_path=rtk_path,
            imu_path=imu_path,
            altimeter_path=altimeter_path,
        )
        rows = []
        total_chunks = 0
        read_kwargs = {
            "header": None,
            "skiprows": skip_lines,
            "chunksize": 50000,
            "na_filter": False,
            "low_memory": False,
        }
        if read_dtype is not None:
            read_kwargs["dtype"] = read_dtype

        for chunk in pd.read_csv(path, **read_kwargs):
            rows.append(chunk)
            total_chunks += 1
            if progress_callback:
                percent = min(20 + total_chunks * 5, 80)
                progress_callback(percent, f"已读取 {total_chunks} 个数据块...")

        df = pd.concat(rows, ignore_index=True, copy=False) if rows else pd.DataFrame()
        if read_dtype is not None:
            raw_data = df.to_numpy(dtype=np.float32, copy=False)
        else:
            raw_data = df.to_numpy(copy=False)
        context.update(
            {
                "chunks": int(total_chunks),
                "raw_shape": tuple(int(v) for v in raw_data.shape),
                "raw_dtype": str(getattr(raw_data, "dtype", "unknown")),
                "raw_nbytes_mb": float(getattr(raw_data, "nbytes", 0) / (1024.0 * 1024.0)),
            }
        )
        return raw_data, context

    def _load_single_csv_with_progress(
        self,
        path,
        progress_callback=None,
        *,
        trace_timestamps_s=None,
        rtk_path=None,
        imu_path=None,
        altimeter_path=None,
    ):
        """带进度回调的CSV加载"""
        try:
            if progress_callback:
                progress_callback(10, "正在检测文件格式...")

            header_info = detect_csv_header(path)
            read_manifest = getattr(self, "_read_csv_sidecar_manifest", None)
            merge_manifest = getattr(self, "_merge_manifest_header_info", None)
            manifest_info = read_manifest(path) if callable(read_manifest) else {}
            if callable(merge_manifest):
                header_info = merge_manifest(header_info, manifest_info)
            skip_lines = _detect_skiprows(path)

            if progress_callback:
                progress_callback(20, "正在读取数据...")

            raw_data, import_context = MainWindowDataLoadingMixin._read_csv_raw_array_for_import(self,
                path,
                skip_lines=skip_lines,
                header_info=header_info,
                trace_timestamps_s=trace_timestamps_s,
                rtk_path=rtk_path,
                imu_path=imu_path,
                altimeter_path=altimeter_path,
                progress_callback=progress_callback,
            )

            if progress_callback:
                progress_callback(85, "正在合并数据...")

            if raw_data.size == 0:
                raise ValueError("CSV 未读取到有效数据")

            if progress_callback:
                progress_callback(90, "正在处理数据...")

            sidecar_kwargs = {}
            if trace_timestamps_s is not None:
                sidecar_kwargs["trace_timestamps_s"] = trace_timestamps_s
            if rtk_path is not None:
                sidecar_kwargs["rtk_path"] = rtk_path
            if imu_path is not None:
                sidecar_kwargs["imu_path"] = imu_path
            if altimeter_path is not None:
                sidecar_kwargs["altimeter_path"] = altimeter_path

            payload_extractor = getattr(
                self, "_extract_airborne_payload_with_optional_sidecars", None
            )
            if payload_extractor is None:
                data, trace_metadata, header_info = extract_airborne_csv_payload(
                    raw_data, header_info, **sidecar_kwargs
                )
            else:
                data, trace_metadata, header_info = payload_extractor(
                    raw_data,
                    header_info,
                    sidecar_kwargs,
                )

            data, memory_summary = sanitize_float32_matrix(data)
            if header_info is None:
                header_info = {}
            header_info = dict(header_info or {})
            header_info["import_context"] = dict(import_context)
            header_info["import_memory_summary"] = dict(memory_summary)

            runtime_warnings = []
            if memory_summary.get("nonfinite_replaced", 0):
                runtime_warnings.append(
                    build_runtime_warning(
                        "data_sanitized",
                        "导入 CSV 数据包含 NaN/Inf，已使用均值填充。",
                        fill_value=memory_summary.get("fill_value", 0.0),
                        path=path,
                    )
                )

            if progress_callback:
                progress_callback(100, "加载完成！")

            return {
                "data": data,
                "header_info": header_info,
                "trace_metadata": trace_metadata,
                "runtime_warnings": runtime_warnings,
                "path": path,
                "import_context": import_context,
                "import_memory_summary": memory_summary,
            }

        except Exception as e:
            raise InputDataError(
                "CSV 加载失败",
                technical_detail=str(e),
                context={"path": path, "loader": "csv"},
            ) from e

    def _load_ascans_folder_with_progress(self, folder, progress_callback=None):
        """带进度回调的文件夹加载"""
        try:
            if progress_callback:
                progress_callback(10, "正在扫描文件夹...")

            def _progress(current, total, msg):
                if progress_callback:
                    percent = int(10 + (current / max(total, 1)) * 80)
                    progress_callback(percent, msg)

            result = read_ascans_folder(
                folder, max_files=0, progress_cb=_progress
            )

            if progress_callback:
                progress_callback(95, "正在处理数据...")

            data = result["data"]
            samples = result["samples_per_trace"]
            traces = result["num_traces"]
            time_step_s = result.get("time_step_s")

            total_time_ns = None
            if time_step_s and samples > 0:
                total_time_ns = time_step_s * samples * 1e9

            header_info = {
                "a_scan_length": samples,
                "total_time_ns": total_time_ns if total_time_ns else 0.0,
                "num_traces": traces,
                "trace_interval_m": 0.01,
                "source": "folder",
                "folder_path": folder,
            }

            data, memory_summary = sanitize_float32_matrix(data)
            header_info["import_memory_summary"] = dict(memory_summary)
            runtime_warnings = []
            if memory_summary.get("nonfinite_replaced", 0):
                runtime_warnings.append(
                    build_runtime_warning(
                        "data_sanitized",
                        "导入的 A-scan 数据包含 NaN/Inf，已使用均值填充。",
                        fill_value=memory_summary.get("fill_value", 0.0),
                        path=folder,
                    )
                )

            if progress_callback:
                progress_callback(100, "加载完成！")

            return {
                "data": data,
                "header_info": header_info,
                "runtime_warnings": runtime_warnings,
                "path": folder,
                "import_memory_summary": memory_summary,
            }

        except Exception as e:
            raise InputDataError(
                "A-scan 文件夹加载失败",
                technical_detail=str(e),
                context={"path": folder, "loader": "ascans_folder"},
            ) from e

    def _load_gprmax_out(self, path, progress_callback=None):
        """加载 gprMax .out 文件"""
        try:
            from core.gpr_io import read_gprmax_out

            if progress_callback:
                progress_callback(10, "正在读取 .out 文件...")

            # 读取 .out 文件
            result = read_gprmax_out(path)

            if progress_callback:
                progress_callback(50, "正在构建头信息...")

            data = result["data"]
            samples = result["samples_per_trace"]
            traces = result["num_traces"]
            time_step_s = result.get("time_step_s")
            total_time_ns = result.get("total_time_ns")
            header_info = dict(result.get("header_info") or {})
            trace_metadata = result.get("trace_metadata")

            # 尝试从同目录的 .in 文件读取 src_steps，得到正确的道间距
            trace_interval_m = float(header_info.get("trace_interval_m") or 0.1)
            try:
                folder = os.path.dirname(path)
                in_files = [f for f in os.listdir(folder) if f.endswith(".in")]
                if in_files:
                    from core.gpr_io import read_gprmax_in

                    in_cfg = read_gprmax_in(os.path.join(folder, in_files[0]))
                    src_steps = in_cfg.get("src_steps")
                    if src_steps and len(src_steps) >= 1 and src_steps[0] > 0:
                        trace_interval_m = src_steps[0]
            except Exception as e:
                logger.debug("Failed to read src_steps from gprMax .in file: %s", e)

            # 构建头信息
            header_info.update(
                {
                    "a_scan_length": samples,
                    "total_time_ns": total_time_ns if total_time_ns else 0.0,
                    "num_traces": traces,
                    "trace_interval_m": trace_interval_m,
                    "source": "gprmax_out",
                    "out_path": path,
                }
            )

            if progress_callback:
                progress_callback(100, "加载完成！")

            return {
                "data": data,
                "header_info": header_info,
                "trace_metadata": trace_metadata,
                "path": path,
            }

        except Exception as e:
            raise InputDataError(
                "gprMax .out 加载失败",
                technical_detail=str(e),
                context={"path": path, "loader": "gprmax_out"},
            ) from e

    def _on_data_load_failed(self, error_msg: str) -> None:
        """数据加载失败回调，写入结构化错误与用户日志。"""
        payload = self._record_structured_error(
            str(error_msg),
            category="input_data",
            context={"operation": "load_data"},
            log=False,
        )
        self._log(
            f"数据加载失败: {payload.get('user_message', error_msg)}",
            event_type="ERR",
            source="input_data",
            context=payload,
        )
        try:
            self._set_runtime_summary("状态：数据加载失败", "danger")
        except Exception:
            pass

    def _on_data_loaded(self, result):
        """数据加载完成回调"""
        if result is None:
            return

        data = result.get("data")
        header_info = result.get("header_info")
        trace_metadata = result.get("trace_metadata")
        path = result.get("path", "")
        import_warnings = list(result.get("runtime_warnings", []) or [])

        if data is None:
            return

        self._clear_runtime_warnings()
        if not np.isfinite(data).all():
            finite_mask = np.isfinite(data)
            fill_value = float(np.mean(data[finite_mask])) if finite_mask.any() else 0.0
            data = np.nan_to_num(
                data, nan=fill_value, posinf=fill_value, neginf=fill_value
            )
            import_warnings.append(
                build_runtime_warning(
                    "data_sanitized",
                    "导入数据包含 NaN/Inf，已使用均值填充。",
                    fill_value=fill_value,
                    path=path,
                )
            )

        # 更新共享数据
        self.shared_data.load_data(
            data,
            path=path,
            header_info=header_info,
            trace_metadata=trace_metadata,
            source="async_load",
        )

        # 更新UI
        self._mark_data_changed()
        self._clear_transient_compare_snapshots()
        self._set_quality_metrics(None)
        self._append_runtime_warnings(import_warnings, source="async_load")

        self._log(f"已加载数据: {data.shape}")
        try:
            summary = summarize_array_memory(data).to_dict()
            self._log(
                "导入内存摘要："
                f"dtype={summary.get('dtype')} "
                f"size={summary.get('nbytes_mb', 0.0):.2f} MB "
                f"contiguous={summary.get('is_c_contiguous')}"
            )
        except Exception:
            pass
        if header_info:
            self.status_label.setText(self._build_status_text())
            for line in self._build_airborne_metadata_summary():
                self._log(line)
        else:
            self.status_label.setText(os.path.basename(path))

        self._update_empty_state_and_brief()
        self.plot_data(data)

    def _load_ascans_folder(self, folder: str):
        """从文件夹加载 A-scan 数据"""
        self._log(f"正在从文件夹加载 A-scan: {folder}")
        self._set_busy(True, text="读取 A-scan 文件...")
        self._clear_runtime_warnings()
        QApplication.processEvents()
        try:
            import_warnings = []

            def _progress(current, total, msg):
                self.status_label.setText(f"{msg} ({current}/{total})")
                QApplication.processEvents()

            result = read_ascans_folder(
                folder, max_files=0, progress_cb=_progress
            )
            data = result["data"]
            samples = result["samples_per_trace"]
            traces = result["num_traces"]
            time_step_s = result.get("time_step_s")

            # 构造 header_info
            total_time_ns = None
            if time_step_s and samples > 0:
                total_time_ns = time_step_s * samples * 1e9

            header_info = {
                "a_scan_length": samples,
                "total_time_ns": total_time_ns if total_time_ns else 0.0,
                "num_traces": traces,
                "trace_interval_m": 0.01,
                "source": "folder",
                "folder_path": folder,
            }

            if not np.isfinite(data).all():
                finite_mask = np.isfinite(data)
                fill = float(np.mean(data[finite_mask])) if finite_mask.any() else 0.0
                data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)
                import_warnings.append(
                    build_runtime_warning(
                        "data_sanitized",
                        "导入的 A-scan 数据包含 NaN/Inf，已使用均值填充。",
                        fill_value=fill,
                        path=folder,
                    )
                )

            if data.ndim == 1:
                data = data.reshape(-1, 1)

            self.shared_data.load_data(
                data,
                path=folder,
                header_info=header_info,
                trace_metadata=None,
                source="ascans_folder",
            )
            self._mark_data_changed()
            self._clear_transient_compare_snapshots()
            self._set_quality_metrics(None)

            self._log(f"已加载文件夹 A-scan: {traces} 道 x {samples} 采样点")
            self.status_label.setText(
                f"{os.path.basename(folder)} | 采样:{samples} 道数:{traces}"
            )

            self._update_empty_state_and_brief()
            self.plot_data(data)
            self._append_runtime_warnings(import_warnings, source="ascans_folder")

        except Exception as e:
            friendly_msg = f"文件夹加载失败:\n{e}"
            QMessageBox.critical(self, "错误", friendly_msg)
            self._log(friendly_msg)
        finally:
            self._set_busy(False, text="就绪")

    def _load_single_csv(
        self,
        path: str,
        *,
        trace_timestamps_s=None,
        rtk_path=None,
        imu_path=None,
        altimeter_path=None,
    ):
        """加载单个CSV矩阵文件"""

        try:
            self._clear_runtime_warnings()
            import_warnings = []
            header_info = detect_csv_header(path)
            manifest_info = self._read_csv_sidecar_manifest(path)
            header_info = self._merge_manifest_header_info(header_info, manifest_info)
            skip_lines = _detect_skiprows(path)

            df = pd.read_csv(
                path,
                header=None,
                skiprows=skip_lines,
                na_filter=False,
                low_memory=False,
            )
            raw_data = df.values

            if raw_data.size == 0:
                raise ValueError("CSV 未读取到有效数据（空文件或分隔符不匹配）")

            sidecar_kwargs = {}
            if trace_timestamps_s is not None:
                sidecar_kwargs["trace_timestamps_s"] = trace_timestamps_s
            if rtk_path is not None:
                sidecar_kwargs["rtk_path"] = rtk_path
            if imu_path is not None:
                sidecar_kwargs["imu_path"] = imu_path
            if altimeter_path is not None:
                sidecar_kwargs["altimeter_path"] = altimeter_path

            payload_extractor = getattr(
                self, "_extract_airborne_payload_with_optional_sidecars", None
            )
            if payload_extractor is None:
                data, trace_metadata, header_info = extract_airborne_csv_payload(
                    raw_data, header_info, **sidecar_kwargs
                )
            else:
                data, trace_metadata, header_info = payload_extractor(
                    raw_data,
                    header_info,
                    sidecar_kwargs,
                )

            try:
                data = np.asarray(data, dtype=np.float32)
            except Exception as conv_err:
                raise ValueError(f"CSV 包含非数值内容，无法转换为数值矩阵: {conv_err}")

            if data.size == 0:
                raise ValueError("CSV 数据矩阵为空")

            if not np.isfinite(data).all():
                finite_mask = np.isfinite(data)
                if finite_mask.any():
                    fill_value = float(np.mean(data[finite_mask]))
                else:
                    fill_value = 0.0
                data = np.nan_to_num(
                    data, nan=fill_value, posinf=fill_value, neginf=fill_value
                )
                self._log(f"检测到 NaN/Inf，已使用 {fill_value:.6g} 填充。")
                import_warnings.append(
                    build_runtime_warning(
                        "data_sanitized",
                        "导入 CSV 数据包含 NaN/Inf，已使用均值填充。",
                        fill_value=fill_value,
                        path=path,
                    )
                )

            if data.ndim == 1:
                data = data.reshape(-1, 1)

            self.shared_data.load_data(
                data,
                path=path,
                header_info=header_info,
                trace_metadata=trace_metadata,
                source="csv_import",
            )
            self._mark_data_changed()
            self._clear_transient_compare_snapshots()
            self._set_quality_metrics(None)

            self._log(f"已加载 CSV： {path}  shape={data.shape}")
            if header_info:
                self.status_label.setText(
                    f"{os.path.basename(path)} | 采样:{header_info['a_scan_length']} 道数:{header_info['num_traces']}"
                )
            else:
                self.status_label.setText(os.path.basename(path))

            if header_info:
                self._log(
                    "检测到头信息： "
                    f"A-scan length={header_info['a_scan_length']} samples; "
                    f"Total time={header_info['total_time_ns']} ns; "
                    f"A-scan count={header_info['num_traces']}; "
                    f"Trace interval={header_info['trace_interval_m']} m"
                )
                for line in self._build_airborne_metadata_summary():
                    self._log(line)
            else:
                self._log("未检测到头信息；使用索引坐标。")

            self._update_empty_state_and_brief()
            self.plot_data(data)
            self._append_runtime_warnings(import_warnings, source="csv_import")

            # 保存数据路径以便下次自动加载
            _save_last_data_path(path)

        except Exception as e:
            friendly_msg = build_csv_load_error_message(e)
            QMessageBox.critical(self, "错误", friendly_msg)
            self._log(f"加载 CSV 失败：\n{friendly_msg}")
