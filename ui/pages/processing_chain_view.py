"""Visible chain presentation; the page's existing step model owns each draft."""
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel

from ui.file_tree import artifact_direct_parent, artifact_parameters
from ui.widgets.chain_strip import ChainStrip
from ui.widgets.saved_source_path import SavedSourcePath


class ProcessingChainView:
    def _build_chain_view(self, layout):
        self._saved_path = SavedSourcePath(self)
        self._saved_path.source_requested.connect(self._workspace_source_open)
        layout.addWidget(self._saved_path)
        self._chain_panel = QWidget(self)
        row = QHBoxLayout(self._chain_panel)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(CaptionLabel('待执行', self._chain_panel))
        self._chain_strip = ChainStrip(self._chain_panel)
        self._chain_input = CaptionLabel('输入：未选择 →', self._chain_panel)
        self._chain_input.hide()  # The compact header already identifies the active input.
        self._chain_strip.set_run_button_visible(False)
        self._chain_strip._empty_hint.setText('点击 ＋ 添加处理步骤')
        row.addWidget(self._chain_strip, 1)
        row.addWidget(self._draft_menu_btn)
        layout.addWidget(self._chain_panel)

    def _connect_chain_view(self):
        strip = self._chain_strip
        strip.sig_step_selected.connect(self._pipeline_list.select_step)
        strip.sig_step_clicked.connect(self._on_chain_chip_clicked)
        strip.sig_step_editor_requested.connect(self._open_step_editor)
        strip.sig_step_toggled.connect(self._on_chain_step_toggled)
        strip.sig_step_removed.connect(self._on_chain_step_removed)
        strip.sig_step_moved.connect(self._on_chain_step_moved)
        strip.sig_add_requested.connect(self._open_method_picker)

    def _workspace_source_open(self, key):
        self._workspace.open_source(key)

    def _on_chain_chip_clicked(self, index):
        if not self._draft_locked():
            self._pipeline_list.select_step(index)
            self._open_step_editor(index)

    def _sync_chain_view(self):
        steps = self._pipeline_list.steps()
        rendered = [{**step, 'param_summary': self._step_summary(step)} for step in steps]
        strip = self._chain_strip
        strip.set_context((self._draft_key(), self._draft_edit_epoch))
        strip.blockSignals(True)
        try:
            strip.set_steps(rendered)
            strip.select_step(self._selected_step)
        finally:
            strip.blockSignals(False)
        strip.setEnabled(not self._draft_locked())
        source = self._sources.get(self._selected_source_key) or {}
        title = '输入：' + str(source.get('title') or '未选择') + ' →'
        self._chain_input.setToolTip(title)
        self._chain_input.setText(self._chain_input.fontMetrics().elidedText(
            title, Qt.TextElideMode.ElideMiddle, 185))
        path, seen, aid = [], set(), self._input_artifact_id()
        while aid and aid in self._artifacts_by_id and aid not in seen:
            seen.add(aid)
            art = self._artifacts_by_id[aid]
            meta = artifact_parameters(art)
            title = self._display_name(str(getattr(art, 'method_id', '') or ''))
            run, index = str(meta.get('run_group_id') or ''), meta.get('run_step_index')
            if run and index:
                title += f' · {run[:6]}.{index}'
            path.insert(0, ('artifact:' + aid, title or aid, str(getattr(art, 'sha256', '') or '')))
            parent = artifact_direct_parent(art)
            if parent is None or (parent and parent not in self._artifacts_by_id):
                path.insert(0, ('', '来源待核对', ''))
                break
            aid = parent
        if aid and aid in seen and path and path[0][0]:
            path.insert(0, ('', '来源待核对', ''))
        if self._selected_source_key and (not path or path[0][0]):
            raw = self._sources.get('original')
            path.insert(0, ('original' if raw else '', '原始输入', self._raw_revision))
        self._saved_path.set_path(path, (self._project_id, self._line_id))
        self._source_path_btn.setEnabled(len(path) > 1)
        self._saved_path.setVisible(len(path) > 1 and self._source_path_btn.isChecked())

    def _close_chain_popups(self):
        for popup in (self._step_editor, self._method_picker):
            if popup is not None:
                popup.close()

    def _step_summary(self, step):
        schema = self._methods_by_id.get(step.get('method_id'), {}).get('parameter_schema') or []
        names = {item.get('name'): item.get('label') or item.get('name') for item in schema}
        values = []
        for key, value in (step.get('params') or {}).items():
            if isinstance(value, (str, int, float, bool)) and value != '':
                values.append(f'{names.get(key, key)}={str(value)[:16]}')
            if len(values) == 2:
                break
        return ' · '.join(values)

    def _position_step_editor(self, popup, index):
        anchor = self._chain_strip.step_anchor(index)
        if anchor is None or not anchor.isVisibleTo(self):
            return
        popup.adjustSize()
        point = anchor.mapToGlobal(QPoint(0, anchor.height() + 5))
        available = anchor.screen().availableGeometry()
        x = max(available.left(), min(point.x(), available.right() - popup.width() + 1))
        y = point.y()
        if y + popup.height() > available.bottom():
            y = anchor.mapToGlobal(QPoint()).y() - popup.height() - 5
        popup.move(x, max(available.top(), y))
