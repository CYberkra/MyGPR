"""Internal processing-page source, run and draft object menus."""
import json
from copy import deepcopy

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from ui.file_tree import artifact_direct_parent, artifact_parameters
from ui.widgets.context_menus import make_menu
from ui.widgets.fluent_menu_keyboard import FluentMenuKeyboard
from ui.widgets.object_context_menu import object_action, show_details


class ProcessingMenus:
    """Shared page state remains owned by ProcessingPage; public methods stay inherited."""

    def _tree_context(self, tree, pos):
        binding = self._history_menu_binding if tree is self._history_tree else self._lineage_menu_binding
        binding.show(pos)

    def _tree_menu(self, tree, pos):
        item = tree.itemAt(pos)
        if item is None:
            menu = make_menu(self)
            object_action(menu, 'no-target', '请先选择数据来源或运行', lambda: None, enabled=False)
            return menu
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key:
            return self._source_menu(str(key))
        group = item.data(0, Qt.ItemDataRole.UserRole + 1)
        return self._run_menu(str(group or ''))

    def _run_group_params(self, art):
        return artifact_parameters(art)

    def _source_menu(self, key):
        source = self._sources.get(key)
        menu = make_menu(self)
        if not source:
            return menu
        target = self._draft_key(key)
        def current(callback):
            if key in self._sources and target == self._draft_key(key):
                callback()
        object_action(menu, 'continue-source', '从此结果继续处理',
                      lambda: current(lambda: self._workspace.open_source(key)),
                      enabled=not source.get('error'), reason='数据读取失败，请重试预览')
        aid = source.get('artifact_id')
        if aid:
            parent = artifact_direct_parent(self._artifacts_by_id.get(aid))
            parent_source = 'artifact:' + parent if parent else 'original'
            parent_identity = self._draft_key(parent_source)
            def reconfigure():
                if parent_source in self._sources and parent_identity == self._draft_key(parent_source):
                    self.reconfigure_artifact(aid)
            object_action(menu, 'reconfigure-step', '修改此步参数并另建分支…',
                          lambda: current(reconfigure),
                          enabled=self.can_reconfigure_artifact(aid), reason='缺少已保存的直接步骤输入')
        object_action(menu, 'source-info', '参数与来源…', lambda: current(lambda: self._show_source_info(key)))
        if source.get('error'):
            object_action(menu, 'retry-preview', '重试预览', lambda: current(lambda: self.retry_preview(key)))
        if aid:
            object_action(menu, 'delete-source', '删除此成果及依赖成果…',
                lambda: current(lambda: self.artifact_delete_requested.emit(target[1], [aid]) if not self._running else None),
                enabled=not self._running, reason='任务运行中')
        return menu

    def _show_source_menu(self, key, pos):
        menu = self._source_menu(key)
        menu._object_keyboard = FluentMenuKeyboard(menu)
        menu.exec(pos)
        menu._object_keyboard.focus_menu(menu)

    def _run_signature(self, group):
        return tuple(sorted((aid, str(getattr(art, 'sha256', '') or ''),
                            json.dumps(artifact_parameters(art), sort_keys=True, default=str))
                           for aid, art in self._artifacts_by_id.items()
                           if artifact_parameters(art).get('run_group_id') == group))

    def _run_menu(self, group):
        menu = make_menu(self)
        members = [art for art in self._artifacts_by_id.values()
                   if artifact_parameters(art).get('run_group_id') == group]
        scope, signature = (self._project_id, self._line_id), self._run_signature(group)
        def current(callback):
            if scope == (self._project_id, self._line_id) and signature and signature == self._run_signature(group):
                callback()
        if not members:
            object_action(menu, 'unknown-run', '来源记录不完整，无法重建运行', lambda: None, enabled=False)
            return menu
        members.sort(key=lambda art: int(artifact_parameters(art).get('run_step_index') or 0))
        final_id = str(members[-1].artifact_id)
        saved = self._saved_run(group)
        input_identity = self._draft_key(saved[0]) if saved else None
        def input_current(callback):
            valid = self._saved_run(group)
            if valid and input_identity == self._draft_key(valid[0]):
                callback()
        object_action(menu, 'open-final', '打开最终成果', lambda: current(lambda: self.select_artifact(final_id)))
        object_action(menu, 'open-saved-steps', '打开本次已保存步骤',
                      lambda: current(lambda: [self.select_artifact(str(art.artifact_id)) for art in members]))
        object_action(menu, 'locate-run-input', '定位冻结输入',
                      lambda: current(lambda: input_current(lambda: self._workspace.open_source(saved[0]))),
                      enabled=saved is not None, reason='输入缺失、修订变化或来源待核对')
        object_action(menu, 'reconfigure-run', '按原输入重新配置本次运行…',
                      lambda: current(lambda: input_current(lambda: self.reconfigure_run(group))),
                      enabled=saved is not None, reason='缺少完整配置、有效原输入或输入草稿被锁定')
        record = {'run_group_id': group, 'project_id': scope[0], 'line_id': scope[1],
                  'outputs': [{'artifact_id': str(art.artifact_id), 'status': str(getattr(art, 'status', '')),
                               'sha256': str(getattr(art, 'sha256', '') or ''), 'params': artifact_parameters(art)} for art in members]}
        text = json.dumps(record, ensure_ascii=False, indent=2, default=str)
        object_action(menu, 'run-info', '查看配置、状态与来源…',
                      lambda: current(lambda: self._show_details('运行 ' + group[:12], text)))
        object_action(menu, 'copy-run-record', '复制运行记录',
                      lambda: current(lambda: QApplication.clipboard().setText(text)))
        return menu

    def _show_details(self, title, text):
        self._details_dialog = show_details(self, title, text)

    def _show_source_info(self, key):
        source = self._sources[key]
        art = self._artifacts_by_id.get(source.get('artifact_id'))
        metadata = artifact_parameters(art) if art else {'source': '原始输入', 'line_id': self._line_id}
        self._show_details(source['title'], json.dumps(metadata, ensure_ascii=False, indent=2, default=str))

    def _draft_menu(self):
        menu = make_menu(self)
        key = self._draft_key()
        def current(callback):
            if key == self._draft_key() and not self._draft_locked():
                callback()
        def restore():
            if key != self._draft_key() or self._draft_locked() or not self._draft_backups.get(key):
                return
            backup = self._draft_backups[key].pop()
            self._suspend = True
            self._pipeline_list.set_steps(deepcopy(backup.get('steps') or []))
            self._result_name_edit.setText(backup.get('result_name') or '')
            self._suspend = False
            if self._pipeline_list.count():
                self._pipeline_list.select_step(0)
            self._draft_changed()
        object_action(menu, 'add-step', '添加处理步骤…', lambda: current(self._open_method_picker), enabled=not self._draft_locked())
        object_action(menu, 'edit-step', '编辑选中步骤…', lambda: current(lambda: self._open_step_editor(self._selected_step)),
                   enabled=not self._draft_locked() and self._selected_step >= 0)
        object_action(menu, 'restore-draft', '恢复此输入的上一个草稿', restore,
                   enabled=not self._draft_locked() and bool(self._draft_backups.get(key)))
        object_action(menu, 'copy-draft', '复制待执行配置', lambda: current(lambda: QApplication.clipboard().setText(
                   json.dumps(self.current_pipeline(), ensure_ascii=False, indent=2))))
        stale = [{'revision': k[3], 'draft': value} for k, value in self._drafts.items()
                 if k[:3] == key[:3] and k != key and value.get('steps')]
        if stale:
            object_action(menu, 'stale-drafts', '查看旧修订草稿（已失效）…', lambda: current(lambda:
                self._show_details('旧修订草稿 · 已失效，不能用于当前数据',
                                   json.dumps(stale, ensure_ascii=False, indent=2, default=str))))
        return menu

    def _show_draft_menu(self):
        menu = self._draft_menu()
        menu._object_keyboard = FluentMenuKeyboard(menu)
        menu.exec(self._draft_menu_btn.mapToGlobal(self._draft_menu_btn.rect().bottomLeft()))
        menu._object_keyboard.focus_menu(menu)
