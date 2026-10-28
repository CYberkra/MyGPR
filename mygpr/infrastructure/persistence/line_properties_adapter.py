"""Project-local line scale and display preference overrides."""
import numpy as np

class LinePropertiesMixin:
    def update_line_properties(self, line_id, length_m, view_reversed):
        safe = self._validated_line(line_id)
        if length_m is not None and (not np.isfinite(length_m) or length_m <= 0):
            raise ValueError("测线长度必须为正数")
        with self._lock:
            self._store.assert_writable()
            record = self._store.get_line(safe)
            if length_m is not None and record.trace_count < 2:
                raise ValueError("至少需要两道才能设置实测长度")
            record.length_m = float(length_m) if length_m is not None else 0.0
            record.distance_override = "measured" if length_m is not None else "unknown"
            record.view_reversed = bool(view_reversed)
            self._store.upsert_line(record)
            return self.get_line(safe)

    def _raw_dataset_with_properties(self, line_id):
        dataset = self._store.load_gpr_dataset(line_id)
        record = self._store.get_line(line_id)
        if record.distance_override in ("measured", "unknown"):
            dataset.distance_axis_m = np.linspace(0, record.length_m, dataset.trace_count, dtype=np.float32)
            dataset.metadata = {**dataset.metadata, "distance_axis_status": record.distance_override}
        dataset.metadata = {**dataset.metadata, "view_reversed": record.view_reversed}
        return dataset
