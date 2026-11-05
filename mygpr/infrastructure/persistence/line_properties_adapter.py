"""Project-local line scale and display preference overrides."""
import json

import numpy as np

class LinePropertiesMixin:
    def _legacy_matrix_unscaled(self, record):
        if str(record.data_format or '').split('+')[0] not in {
                'csv-matrix-chunked', 'npy-matrix-chunked', 'hdf5-matrix-chunked'}:
            return False
        try:
            payload = json.loads(self._store.gpr_metadata_path(record.line_id).read_text(encoding='utf-8'))
            metadata = payload.get('metadata') or {}
            if not isinstance(metadata, dict):
                metadata = {}
        except (OSError, ValueError, AttributeError):
            metadata = {}
        return not (metadata.get('physical_axes_source') == 'user_input'
                    or metadata.get('time_axis_status') == 'known')

    def _mark_legacy_matrix_axes(self, dataset, record):
        if self._legacy_matrix_unscaled(record):
            dataset.metadata.setdefault('time_axis_status', 'unknown')
            if record.distance_override != 'measured':
                dataset.metadata.setdefault('distance_axis_status', 'unknown')

    def _normalized_artifact_dataset(self, line_id, artifact_id):
        dataset = self._store.storage.load_processing_artifact(line_id, artifact_id)
        self._mark_legacy_matrix_axes(dataset, self._store.get_line(line_id))
        return dataset

    def update_line_properties(self, line_id, length_m, view_reversed, time_window_ns=None):
        safe = self._validated_line(line_id)
        if length_m is not None and (not np.isfinite(length_m) or length_m <= 0):
            raise ValueError("测线长度必须为正数")
        if time_window_ns is not None and (not np.isfinite(time_window_ns) or time_window_ns < 0):
            raise ValueError("时间窗必须为有限正数或 0（取消补录）")
        with self._lock:
            self._store.assert_writable()
            record = self._store.get_line(safe)
            if length_m is not None and record.trace_count < 2:
                raise ValueError("至少需要两道才能设置实测长度")
            if time_window_ns is not None and time_window_ns > 0:
                dataset = self._store.load_gpr_dataset(safe)
                if dataset.sample_count < 2 or dataset.metadata.get('is_depth') or dataset.metadata.get('is_elevation'):
                    raise ValueError("深度、高程或不足两个采样点的数据不能补录时间窗")
            record.length_m = float(length_m) if length_m is not None else 0.0
            record.distance_override = "measured" if length_m is not None else "unknown"
            record.view_reversed = bool(view_reversed)
            if time_window_ns is not None:
                record.time_window_override_ns = float(time_window_ns)
            self._store.upsert_line(record)
            return self.get_line(safe)

    def _raw_dataset_with_properties(self, line_id):
        dataset = self._store.load_gpr_dataset(line_id)
        record = self._store.get_line(line_id)
        self._mark_legacy_matrix_axes(dataset, record)
        if record.time_window_override_ns > 0:
            self._apply_time_window_override(dataset, record.time_window_override_ns)
        if record.distance_override in ("measured", "unknown"):
            dataset.distance_axis_m = np.linspace(0, record.length_m, dataset.trace_count, dtype=np.float32)
            dataset.metadata = {**dataset.metadata, "distance_axis_status": record.distance_override}
        dataset.metadata = {**dataset.metadata, "view_reversed": record.view_reversed}
        return dataset
