"""Convert an active B-scan selection into processing steps."""


class ProcessingSelection:
    def _selection_step_allowed(self, group_id, key):
        group = self._workspace._groups.get(group_id)
        return bool(group and self._project_id and self._line_id and not self._running
                    and not self._project_read_only and key == self._selected_source_key
                    and group.current_key() == key and key in self._sources
                    and not self._sources[key].get('comparison')
                    and group.rendered_bundle is self._sources[key].get('bundle')
                    and group.view._view_selection is not None
                    and not self._pipeline_list.steps())

    def _selection_to_step(self, group_id, key, method):
        import math
        import numpy as np
        if method not in ('trace_cut', 'trace_qc', 'trace_cut_time') or not self._selection_step_allowed(group_id, key):
            return False
        view = self._workspace._groups[group_id].view
        rect = view._view_selection.normalized()
        first = max(0, math.floor(rect.left()))
        stop = min(view._matrix.shape[1], math.ceil(rect.right()))
        if stop <= first:
            return False
        time_params = None
        if method == 'trace_cut_time':
            axis = view._sample_axis
            if view._showing_elevation or 'ns' not in view._sample_axis_label.lower() or axis is None:
                return False
            axis = np.asarray(axis, dtype=float)
            if len(axis) < 2 or not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
                return False
            start = max(0, math.floor(rect.top()))
            end = min(len(axis), math.ceil(rect.bottom()))
            if end <= start:
                return False
            time_end = float(axis[end]) if end < len(axis) else float(axis[-1] + axis[-1] - axis[-2])
            time_params = {'mode': 'keep_range', 'time_start_ns': float(axis[start]), 'time_end_ns': time_end}
        if method in ('trace_cut', 'trace_cut_time'):
            params = {'trace_start': first + 1, 'trace_end': stop}
            label = '测道裁剪'
        else:
            params = {'mode': 'mark', 'empty_rms_threshold': 0.0, 'spike_zscore': 0.0,
                      'manual_trace_indices': f'{first}-{stop - 1}'}
            label = '坏道标记'
        self._pipeline_list.add_step('trace_cut' if method == 'trace_cut_time' else method, label, params)
        if time_params is not None:
            self._pipeline_list.add_step('time_cut', '时间窗裁剪', time_params)
        return True
