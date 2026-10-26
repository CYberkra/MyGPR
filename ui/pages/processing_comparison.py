"""Project-scoped, read-only line previews, separate from processing drafts."""
import json
from uuid import uuid4

from PyQt6.QtCore import QMimeData, QObject
from ui.widgets.bscan_line_drop import BScanLineDrop
from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.fluent_menu_keyboard import FluentMenuKeyboard
from ui.result_presentation import describe_artifact

MIME = 'application/x-mygpr-line-reference'


class LineComparison(QObject):
    def __init__(self, page):
        super().__init__(page)
        self.page = page
        self.sources, self.pending, self.lines = {}, {}, {}
        self.line_groups = {}
        self.active = ''
        self.epoch = uuid4().hex
        self.drop = BScanLineDrop(page._workspace, self)
        page._line_combo.mime_factory = self.mime_for_line
        page._line_combo.drag_finished.connect(self.drop.hide_hints)
        page._workspace.source_closed.connect(self.closed)
        page._workspace.source_input_requested.connect(self.make_input)

    def label(self, line_id):
        return self.lines.get(str(line_id), str(line_id))

    def reset(self):
        self.epoch = uuid4().hex
        self.sources.clear()
        self.pending.clear()
        self.line_groups.clear()
        self.active = ''
        self.drop.hide_hints()

    def set_lines(self, lines):
        self.lines = {str(getattr(line, 'line_id', '')): self._line_label(line) for line in lines}
        for key in list(self.sources):
            if self.sources[key]['line_id'] not in self.lines:
                self.sources.pop(key)
                self.pending.pop(key, None)
            elif key.endswith(':raw'):
                self.sources[key]['title'] = f'{self.label(self.sources[key]["line_id"])} · 原始数据 · 对比'
        self.page._sync_sources()

    def restart_pending(self):
        workspace=self.page._workspace
        workspace._pending.difference_update(self.pending)
        self.pending.clear()
        workspace._sync()

    @staticmethod
    def _line_label(line):
        line_id, name = str(getattr(line, 'line_id', '')), str(getattr(line, 'name', '') or '')
        return line_id if not name or name == line_id else f'{line_id} {name}'

    def mime_for_line(self, line_id):
        mime = QMimeData()
        if self.page._project_id and line_id in self.lines:
            mime.setData(MIME, json.dumps({'scope': self.epoch, 'project': self.page._project_id,
                                        'line': line_id}).encode('utf-8'))
        return mime

    def read_mime(self, mime):
        if not mime.hasFormat(MIME) or len(mime.data(MIME)) > 4096:
            return ''
        try:
            data = json.loads(bytes(mime.data(MIME)))
        except (ValueError, UnicodeError):
            return ''
        if (isinstance(data, dict) and data.get('scope') == self.epoch
                and data.get('project') == self.page._project_id
                and isinstance(data.get('line'), str) and data['line'] in self.lines):
            return data['line']
        return ''

    def owns(self, key):
        return key in self.sources

    def has_open(self):
        workspace = self.page._workspace
        return any(key in self.sources for gid in workspace.group_ids() for key in workspace.tab_keys(gid))

    def open_line(self, line_id, group_id=None, edge='center', index=None):
        page, workspace = self.page, self.page._workspace
        if not page._project_id or line_id not in self.lines:
            return False
        key = f'compare:{self.epoch}:{line_id}:raw'
        if key not in self.sources:
            self.sources[key] = dict(key=key, title=f'{self.label(line_id)} · 原始数据 · 对比',
                line_id=line_id, bundle=None, comparison=True, artifact_id='')
            page._sync_sources()
        if key not in workspace._sources:
            return False
        target = workspace._group(group_id)
        if target is None:
            return False
        for gid in workspace.group_ids():
            if key in workspace.tab_keys(gid):
                target = workspace._group(gid)
                break
        with workspace._change():
            if key not in workspace.tab_keys(target.group_id):
                if edge in ('left', 'right', 'top', 'bottom'):
                    new = workspace._new_group()
                    workspace._split_group(target, new, edge)
                    target = new
                workspace.open_source(key, target.group_id)
                if index is not None:
                    workspace.move_source(key, target.group_id, target.group_id, index)
        self.line_groups[line_id] = target.group_id
        self.open_line_artifacts(line_id, target.group_id)
        page.comparison_artifacts_requested.emit(page._project_id, str(line_id))
        return True

    def open_line_artifacts(self, line_id, group_id):
        for key, source in self.sources.items():
            if source['line_id'] == line_id and source.get('artifact_id'):
                self.page._workspace.open_source(key, group_id)

    def set_artifacts(self, line_id, artifacts):
        page = self.page
        line_id = str(line_id)
        artifacts = list(artifacts or [])
        by_id = {str(getattr(item, 'artifact_id', '')): item for item in artifacts}
        for artifact in by_id.values():
            artifact_id = str(artifact.artifact_id)
            key = f'compare:{self.epoch}:{line_id}:artifact:{artifact_id}'
            if key in self.sources:
                continue
            presentation = describe_artifact(artifact, by_id)
            self.sources[key] = dict(
                key=key,
                title=f'{self.label(line_id)} · {presentation.tab_title} · 对比',
                line_id=line_id,
                artifact_id=artifact_id,
                presentation=presentation,
                bundle=None,
                comparison=True,
            )
        page._sync_sources()
        group_id = self.line_groups.get(line_id)
        if group_id:
            self.open_line_artifacts(line_id, group_id)

    def request(self, key):
        source = self.sources.get(key)
        if not source or key in self.pending:
            return
        token = uuid4().hex
        self.pending[key] = token
        if source.get('artifact_id'):
            self.page.comparison_artifact_preview_requested.emit(
                self.page._project_id, source['line_id'], source['artifact_id'], token)
        else:
            self.page.comparison_preview_requested.emit(self.page._project_id, source['line_id'], token)

    def deliver(self, project_id, line_id, token, bundle, error):
        if str(project_id) != self.page._project_id:
            return
        key = f'compare:{self.epoch}:{line_id}:raw'
        if key not in self.sources or self.pending.get(key) != token:
            return
        self.pending.pop(key)
        self.sources[key].update(bundle=bundle, error=str(error or ''))
        self.page._sync_sources()
        self.page._refresh_ui()

    def deliver_artifact(self, project_id, line_id, artifact_id, token, bundle, error):
        if str(project_id) != self.page._project_id:
            return
        key = f'compare:{self.epoch}:{line_id}:artifact:{artifact_id}'
        if key not in self.sources or self.pending.get(key) != token:
            return
        self.pending.pop(key)
        self.sources[key].update(bundle=bundle, error=str(error or ''))
        self.page._sync_sources()
        self.page._refresh_ui()

    def closed(self, key):
        if key in self.sources:
            self.pending.pop(key, None)
            self.sources[key].update(bundle=None, error='')
            if self.active == key:
                self.active = ''
            self.page._sync_sources()

    def retry(self, key):
        if key not in self.sources:
            return False
        self.pending.pop(key, None)
        self.sources[key].update(bundle=None, error='')
        self.page._sync_sources()
        self.request(key)
        return True

    def make_input(self, key):
        source = self.sources.get(key)
        if not source or source['line_id'] not in self.lines:
            return False
        line_id = source['line_id']
        if line_id != self.page._line_id:
            self.page.line_changed.emit(line_id)
        if self.page._line_id != line_id:
            return False
        self.page.select_original()
        return True

    def show_menu(self, key, position):
        epoch = self.epoch
        def current(callback):
            if self.epoch == epoch and key in self.sources:
                callback(key)
        menu = make_menu(self.page)
        add_action(menu, None, '设为处理输入', lambda: current(self.make_input))
        add_action(menu, None, '重新加载预览', lambda: current(self.retry))
        menu._object_keyboard = FluentMenuKeyboard(menu)
        menu.exec(position)
        menu._object_keyboard.focus_menu(menu)
