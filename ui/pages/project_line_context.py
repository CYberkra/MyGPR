"""Apply asynchronous line-source and sensor context to the project page."""
from ui.sensor_summary import SensorCompletenessReport


class ProjectLineContextMixin:
    def set_line_source_path(self, line_id: str, path) -> None:
        """主窗口/controller 异步回包：缓存测线源文件路径（右键菜单素材）。

        由 page_coordinator 把 ProjectController.line_source_path_ready
        接到本方法；查询在 worker 线程完成，右键菜单构建时直接读缓存。
        """
        line_id = str(line_id or '')
        if line_id:
            self._source_path_cache[line_id] = str(path) if path else None
            if line_id == self._current_line_id:
                self._update_action_state()

    def set_line_sensor_summary(self, line_id: str, text: str) -> None:
        """Compatibility for text-only summary clients."""
        self._cache_sensor_summary(line_id, str(text))

    def set_line_sensor_report(self, line_id: str, report: SensorCompletenessReport) -> None:
        self._cache_sensor_summary(line_id, report)

    def set_sensor_sync_result(self, line_id: str, result) -> None:
        self._sensor_sync_results[str(line_id)] = result
        if str(line_id) == self._current_line_id:
            self._update_action_state()

    def _cache_sensor_summary(self, line_id: str, report: SensorCompletenessReport | str) -> None:
        if str(line_id) not in {str(getattr(line, 'line_id', '')) for line in self._lines}:
            return
        self._sensor_summary_cache[str(line_id)] = report
        if str(line_id) == self._current_line_id:
            self._update_action_state()

    def _sensor_template_key(self):
        return f"sensor_sync_template:{getattr(self._summary, 'project_id', '')}"

    def _save_sensor_template(self):
        if self._sm is None or not getattr(self._summary, 'project_id', ''):
            return
        self._sm.set(self._sensor_template_key(), self._sync_form.settings())
        self._sm.save()

    def _apply_sensor_template(self):
        if self._sm is None:
            return
        values = self._sm.get(self._sensor_template_key(), {})
        if isinstance(values, dict):
            self._sync_form.set_settings(values)
