"""Internal processing-page step editing and parameter presentation."""
from copy import deepcopy
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidgetItem

from ui.file_tree import artifact_parameters
from ui.result_presentation import describe_artifact


@dataclass(frozen=True)
class _TuneTarget:
    draft_key: tuple
    uid: str
    method_id: str
    params: dict
    title: str


class ProcessingSteps:
    """Shared page state remains owned by ProcessingPage; public methods stay inherited."""

    def _display_name(self, method_id, fallback=''):
        method = self._methods_by_id.get(method_id, {})
        return str(method.get('display_name') or method.get('name') or fallback or method_id)

    @staticmethod
    def _editor_parameters(step):
        params = dict(step.get('params') or {})
        if step.get('method_id') == 'short_trace_repair' and 'window_mode' not in params:
            params['window_mode'] = 'manual' if str(params.get('time_windows_ns') or '').strip() else 'auto'
        return params

    def set_methods(self, methods):
        self._methods = [dict(m) for m in methods or []]
        self._methods_by_id = {m.get('method_id', ''): m for m in self._methods}
        self._method_browser.set_methods(self._methods)
        self._sync_chain_view()
        if self._picker_context is None:
            self._show_selected_step()
        else:
            self._on_library_selection(self._method_browser.current_method_id())
        self._sync_autotune()

    def _on_chain_step_toggled(self, index: int, enabled: bool) -> None:
        """启用/禁用：与当前状态不同才翻转（PipelineList 内置翻转语义）。"""
        if not self._can_edit_chain():
            return
        steps = self._pipeline_list.steps()
        if 0 <= index < len(steps) and steps[index]['enabled'] != enabled:
            self._pipeline_list._toggle_enabled(index)

    def _on_chain_step_removed(self, index: int) -> None:
        if not self._can_edit_chain():
            return
        self._step_edits.pop(self._pipeline_list.step_uid(index), None)
        self._pipeline_list._remove_step(index)

    def _on_chain_step_moved(self, source: int, target: int) -> None:
        """chip 拖拽落点 → 步骤换位（target 为插入位语义）。"""
        if not self._can_edit_chain():
            return
        self._pipeline_list._move_step_to(source, target)

    def _on_step_editor_replace(self, index: int) -> None:
        self._open_method_picker(purpose='replace', index=index)

    def _duplicate_step(self, index: int) -> None:
        if not self._can_edit_chain() or not self._pipeline_list.step_uid(index):
            return
        steps = self._pipeline_list.draft_steps()
        duplicate = deepcopy(steps[index])
        duplicate.pop('_step_uid')
        steps.insert(index + 1, duplicate)
        self._pipeline_list.set_steps(steps)
        self._pipeline_list.select_step(index + 1)
        self._pipeline_list.sig_changed.emit()

    def _replace_step_method(self, index: int, method_id: str, params=None) -> None:
        if not self._can_edit_chain() or not self._pipeline_list.step_uid(index):
            return
        steps = self._pipeline_list.draft_steps()
        self._step_edits.pop(self._pipeline_list.step_uid(index), None)
        steps[index] = {
            'method_id': method_id, 'label': self._display_name(method_id),
            'params': self._method_defaults(method_id) if params is None else params,
            'enabled': True,
        }
        self._pipeline_list.set_steps(steps)
        self._pipeline_list.select_step(index)
        self._pipeline_list.sig_changed.emit()

    def _method_defaults(self, method_id):
        schema = self._methods_by_id.get(method_id, {}).get('parameter_schema') or []
        return {item['name']: item.get('default') for item in schema if item.get('name')}

    def _add_method_to_pipeline(self, method_id: str, params=None) -> None:
        if self._draft_locked():
            return
        self._pipeline_list.add_step(
            method_id, self._display_name(method_id),
            self._method_defaults(method_id) if params is None else params)

    def _on_step_selected(self, index: int) -> None:
        self._selected_step = index if self._pipeline_list.step_uid(index) else -1
        self._chain_strip.select_step(self._selected_step)
        if self._picker_context is None:
            self._show_selected_step()
        self._sync_autotune()

    def set_autotune_running(self, running: bool) -> None:
        self._autotune_running = bool(running)
        self._sync_autotune()

    def set_autotune_failed(self, method_id: str) -> None:
        request = self._autotune_pending_request
        if request is not None and request.method_id == method_id:
            self._autotune_pending_request = None
            self.set_autotune_running(False)

    def _on_autotune_clicked(self) -> None:
        index = self._selected_step
        uid = self._pipeline_list.step_uid(index)
        if (self._autotune_running or self._autotune_pending_request is not None
                or not self._can_edit_chain() or not uid or uid in self._step_edits):
            return
        step = self._pipeline_list.steps()[index]
        source = self._sources[self._selected_source_key]
        target = _TuneTarget(
            self._draft_key(), uid, step['method_id'], deepcopy(step['params']),
            f'{self._line_id} · {source["title"]} · 第 {index + 1} 步 {self._display_name(step["method_id"], step["label"])}')
        self._autotune_pending_request = target
        self._autotune_result = None
        self._sync_autotune()
        self.autotune_requested.emit(target.method_id, deepcopy(target.params), self._input_artifact_id())

    def set_autotune_result(self, method_id, result):
        target = self._autotune_pending_request
        if target is None or target.method_id != method_id:
            return
        self._autotune_pending_request = None
        self._autotune_result = (target, dict((result or {}).get('best_params') or {}))
        self._autotune_card.set_collapsed(False, animate=False)
        self._sync_autotune()

    def _tune_target_index(self, target):
        if target.draft_key != self._draft_key():
            return -1
        index = self._pipeline_list.index_of_step(target.uid)
        if index < 0:
            return -1
        step = self._pipeline_list.steps()[index]
        if step['method_id'] != target.method_id or step['params'] != target.params:
            return -1
        return index

    def _can_adopt_autotune(self):
        if not self._autotune_result or self._autotune_running or not self._can_edit_chain():
            return False
        target, params = self._autotune_result
        return bool(params and self._tune_target_index(target) >= 0 and target.uid not in self._step_edits)

    def _sync_autotune(self):
        uid = self._pipeline_list.step_uid(self._selected_step)
        step = self._pipeline_list.steps()[self._selected_step] if uid else None
        self._autotune_btn.setEnabled(bool(
            self._can_edit_chain() and step and step['method_id'] and uid not in self._step_edits
            and not self._autotune_running and self._autotune_pending_request is None))
        label = (f'第 {self._selected_step + 1} 步 · {self._display_name(step["method_id"], step["label"])}'
                 if step else '')
        self._autotune_method_label.setText(label)
        self._autotune_method_label.setVisible(bool(label))
        self._adopt_params_btn.setEnabled(self._can_adopt_autotune())
        self._autotune_result_label.setText(self._autotune_summary())

    def _autotune_summary(self):
        if self._autotune_result is None:
            target = self._autotune_pending_request
            return f'正在评估 {target.title}' if target else ''
        target, params = self._autotune_result
        index = self._tune_target_index(target)
        lines = [f'调参输入 · {target.title}']
        if target.draft_key != self._draft_key():
            lines.append('切回调参时的输入可查看并应用建议')
        elif index < 0:
            lines.append('原步骤或参数已变化，请重新调参')
        elif target.uid in self._step_edits:
            lines.append('目标步骤有未应用的修改')
        elif params:
            lines.append(f'应用到当前第 {index + 1} 步')
        names = {item['name']: item.get('label') or item['name']
                 for item in self._methods_by_id.get(target.method_id, {}).get('parameter_schema') or []
                 if item.get('name')}
        changes = [f'{names.get(key, key)}  {str(target.params.get(key, "默认"))[:32]} → {str(value)[:32]}'
                   for key, value in params.items() if target.params.get(key) != value]
        lines.extend(changes or ['建议参数与提交值一致' if params else '本次没有可应用的参数建议'])
        return '\n'.join(lines)

    def _on_adopt_params(self) -> None:
        if not self._can_adopt_autotune():
            return
        target, params = self._autotune_result
        index = self._tune_target_index(target)
        self._autotune_result = None
        self._open_step_editor(index)
        self._pipeline_list.update_step_params(index, {**target.params, **params})
        self._step_feedback.setText(f'已将建议参数应用到第 {index + 1} 步')
        self._sync_step_tools()

    def set_artifacts(self, artifacts):
        self._stash_draft()
        self._artifacts_by_id = {str(a.artifact_id): a for a in artifacts or [] if getattr(a, 'artifact_id', '')}
        raw = self._sources.get('original')
        old = self._sources
        self._sources = {'original': raw} if raw else {}
        for aid, art in self._artifacts_by_id.items():
            key = 'artifact:' + aid
            presentation = describe_artifact(art, self._artifacts_by_id)
            revision = str(getattr(art, 'sha256', '') or '')
            previous = old.get(key, {})
            changed = bool(previous and previous.get('revision', '') != revision)
            self._sources[key] = {'key': key, 'title': presentation.tab_title,
                                  'tooltip': presentation.tooltip, 'artifact_id': aid,
                                  'presentation': presentation,
                                  'revision': revision,
                                  'bundle': None if changed else previous.get('bundle')}
            if changed:
                self._preview_requested.discard(aid)
                self._sources[key]['error'] = '数据修订已变化，重新加载预览'
            elif previous.get('error'):
                self._sources[key]['error'] = previous['error']
        self._sync_sources()
        changed_sources = [key for key in self._sources
                           if old.get(key) and old[key].get('revision', '') != self._sources[key].get('revision', '')]
        for key in changed_sources:
            self._sources[key].pop('error', None)
        if changed_sources:
            self._sync_sources()
        self._activate_source(self._workspace.active_key())
        self._render_trees()
        self._restore_workspace()
        pending = self._completion_pending
        if pending and pending['key'][:2] == (self._project_id, self._line_id):
            expected = str(pending.get('result_id') or '')
            candidates = [a for aid, a in self._artifacts_by_id.items()
                          if (aid == expected if expected else aid not in pending['artifacts']
                          and artifact_parameters(a).get('run_input_artifact_id') == pending['input_artifact_id'])]
            if candidates:
                final = max(candidates, key=lambda a: (str(getattr(a, 'created_at', '')),
                            int(artifact_parameters(a).get('run_step_index') or 0)))
                self._completed_artifact = str(final.artifact_id)
                self._completion_pending = None
                if pending['activate'] and pending['epoch'] == self._activation_epoch and self._selected_source_key:
                    if pending.get('pipeline', {}).get('save_intermediates'):
                        run_group = str(artifact_parameters(final).get('run_group_id') or '')
                        members = [a for a in self._artifacts_by_id.values()
                                   if artifact_parameters(a).get('run_group_id') == run_group]
                        members.sort(key=lambda a: int(artifact_parameters(a).get('run_step_index') or 0))
                        for artifact in members:
                            self._workspace.open_source(f'artifact:{artifact.artifact_id}')
                    self.select_artifact(self._completed_artifact)
                else:
                    self._completion_btn.show()
        self._refresh_ui()

    def _render_trees(self):
        from ui.pages import processing_page

        self._lineage_tree.clear()
        pending = [(None, node) for node in reversed(processing_page.build_lineage_model(
            list(self._artifacts_by_id.values()), self._line_id))]
        while pending:
            parent, node = pending.pop()
            text = node.text + (f' · {node.suffix}' if node.suffix else '')
            item = QTreeWidgetItem([text])
            key = 'original' if node.kind == 'raw' else 'artifact:' + node.payload if node.kind == 'artifact' else ''
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            item.setToolTip(0, node.tooltip)
            (parent.addChild if parent else self._lineage_tree.addTopLevelItem)(item)
            pending.extend((item, child) for child in reversed(node.children))
            item.setExpanded(node.kind in ('raw', 'group') or bool(key and key == self._selected_source_key))
        self._history_tree.clear()
        groups = {}
        for aid, art in self._artifacts_by_id.items():
            meta = artifact_parameters(art)
            group = str(meta.get('run_group_id') or '历史成果 · 来源待核对')
            groups.setdefault(group, []).append((aid, art))
        for group, members in groups.items():
            parent = QTreeWidgetItem([group[:12]])
            parent.setData(0, Qt.ItemDataRole.UserRole + 1, group)
            self._history_tree.addTopLevelItem(parent)
            for aid, art in sorted(members, key=lambda pair: int(artifact_parameters(pair[1]).get('run_step_index') or 0)):
                item = QTreeWidgetItem([self._sources['artifact:' + aid]['title']])
                item.setData(0, Qt.ItemDataRole.UserRole, 'artifact:' + aid)
                parent.addChild(item)

    def _open_tree_item(self, item, _column):
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key:
            self._workspace.open_source(str(key))

    def select_artifact(self, artifact_id):
        key = 'artifact:' + str(artifact_id)
        if key not in self._sources:
            return False
        self._workspace.open_source(key)
        return True

    def has_artifact(self, artifact_id):
        return str(artifact_id) in self._artifacts_by_id

    def select_original(self):
        if not self._line_id and 'original' not in self._sources:
            return False
        # An explicit open overrides a saved empty layout, even before I/O finishes.
        self._pending_layout = None
        self._raw_opened = True
        if 'original' not in self._sources:
            self._sources['original'] = dict(key='original', title='原始输入',
                bundle=None, artifact_id='', revision=self._raw_revision)
            self._sync_sources()
        return self._workspace.open_source('original')
