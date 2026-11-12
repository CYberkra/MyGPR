"""Inline algorithm library and parameter editor for the active input draft."""
from copy import deepcopy

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, PrimaryPushButton, PushButton, TransparentPushButton, TransparentToolButton
from qfluentwidgets import FluentIcon as FIF

from ui.widgets import ParamForm
from ui.widgets.elided_caption import ElidedCaptionLabel
from ui.widgets.method_browser import MethodBrowser


class ProcessingSidebar:
    def _build_step_tools(self, parent, layout):
        self._picker_context = None
        self._editor_uid = ''
        self._step_edits = {}
        self._step_tools = QWidget(parent)
        self._step_tools.setObjectName('processingStepTools')
        tools = QVBoxLayout(self._step_tools)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(8)
        header = QHBoxLayout()
        self._step_tools_title = ElidedCaptionLabel(self._step_tools)
        title_font = self._step_tools_title.font()
        title_font.setBold(True)
        self._step_tools_title.setFont(title_font)
        self._reset_params_btn = TransparentPushButton('恢复默认', self._step_tools)
        self._reset_params_btn.clicked.connect(self._restore_default_params)
        self._close_library_btn = TransparentToolButton(FIF.CLOSE, self._step_tools)
        self._close_library_btn.setAccessibleName('关闭算法库')
        self._close_library_btn.setToolTip('关闭算法库（Esc）')
        self._close_library_btn.clicked.connect(self._leave_method_library)
        header.addWidget(self._step_tools_title, 1)
        header.addWidget(self._reset_params_btn)
        header.addWidget(self._close_library_btn)
        tools.addLayout(header)
        self._method_browser = MethodBrowser(self._step_tools)
        self._method_browser.setFixedHeight(208)
        self._method_browser.setToolTip('双击或按 Enter 使用默认参数添加')
        self._method_browser.sig_method_selected.connect(self._on_library_selection)
        self._method_browser.sig_add_requested.connect(self._quick_add_method)
        self._library_escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), self._step_tools,
                                         context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._library_escape.activated.connect(self._leave_method_library)
        self._library_escape.setEnabled(False)
        tools.addWidget(self._method_browser)
        self._step_detail = CaptionLabel('', self._step_tools)
        self._step_detail.setWordWrap(True)
        self._step_detail.setTextFormat(Qt.TextFormat.PlainText)
        tools.addWidget(self._step_detail)
        self._param_empty_hint = CaptionLabel('未选择处理步骤', self._step_tools)
        tools.addWidget(self._param_empty_hint)
        self._param_form = ParamForm(self._step_tools)
        self._param_form.sig_changed.connect(self._stage_step_params)
        self._param_form.gain_preview_requested.connect(self._request_gain_preview)
        tools.addWidget(self._param_form)
        actions = QHBoxLayout()
        self._discard_step_btn = PushButton('放弃修改', self._step_tools)
        self._discard_step_btn.clicked.connect(self._load_selected_step)
        self._commit_step_btn = PrimaryPushButton('应用到草稿', self._step_tools)
        self._commit_step_btn.clicked.connect(self._commit_step_form)
        actions.addWidget(self._discard_step_btn)
        actions.addWidget(self._commit_step_btn, 1)
        tools.addLayout(actions)
        self._step_feedback = CaptionLabel('', self._step_tools)
        self._step_feedback.setWordWrap(True)
        tools.addWidget(self._step_feedback)
        layout.addWidget(self._step_tools)

    def _can_edit_chain(self):
        source = self._sources.get(self._selected_source_key)
        return bool(source and not source.get('error')
                    and (self._selected_source_key != 'original' or self._selected_revision == self._raw_revision)
                    and not self._draft_locked() and not self._project_read_only
                    and not self._comparison.active)

    def _reset_step_tools(self):
        self._close_gain_previews()
        self._picker_context = None
        self._step_edits.clear()
        self._on_step_selected(-1)

    def _reveal_step_tools(self):
        self._tools_btn.setChecked(True)
        self._tools_panel.setCurrentWidget(self._chain_scroll)
        self._chain_scroll.verticalScrollBar().setValue(0)

    def _open_method_picker(self, purpose='append', index=-1):
        if not self._can_edit_chain():
            return
        replacing = purpose == 'replace'
        if replacing and not self._pipeline_list.step_uid(index):
            return
        if not replacing and self._picker_context == (self._draft_key(), None):
            self._reveal_step_tools()
            self._method_browser.focus_search()
            return
        self._picker_context = (self._draft_key(), (index, self._draft_edit_epoch) if replacing else None)
        self._close_gain_previews()
        self._step_tools_title.setText(f'替换第 {index + 1} 步' if replacing else '算法库')
        self._method_browser.add_label = '选择此算法' if replacing else '使用默认参数添加'
        self._method_browser.setToolTip('选定算法后确认替换' if replacing else '双击或按 Enter 使用默认参数添加')
        self._method_browser.reset_selection()
        self._on_library_selection('')
        self._reveal_step_tools()
        self._method_browser.focus_search()

    def _leave_method_library(self):
        if self._picker_context is None:
            return
        self._close_gain_previews()
        self._picker_context = None
        self._show_selected_step()

    def _on_library_selection(self, method_id):
        if self._picker_context is None:
            return
        self._close_gain_previews()
        method = self._methods_by_id.get(method_id, {})
        self._step_detail.setText(str(method.get('description') or ''))
        self._load_parameter_values(method_id, {})
        self._step_feedback.clear()
        self._sync_step_tools()

    def _quick_add_method(self, method_id):
        if not self._valid_library_choice(method_id):
            return
        if self._picker_context[1] is not None:
            self._method_browser.select_method(method_id)
            return
        self._append_library_method(method_id, self._method_defaults(method_id))

    def _valid_library_choice(self, method_id):
        return bool(self._picker_context and self._picker_context[0] == self._draft_key()
                    and self._can_edit_chain() and method_id in self._methods_by_id
                    and self._method_browser.is_method_visible(method_id))

    def _append_library_method(self, method_id, params):
        self._add_method_to_pipeline(method_id, params)
        self._step_feedback.setText(f'已添加第 {self._pipeline_list.count()} 步')
        self._sync_step_tools()

    def _open_step_editor(self, index):
        if not self._can_edit_chain() or not self._pipeline_list.step_uid(index):
            return
        self._leave_method_library()
        self._pipeline_list.select_step(index)
        self._show_selected_step()
        self._reveal_step_tools()

    def _show_selected_step(self):
        index = self._selected_step
        uid = self._pipeline_list.step_uid(index)
        if uid != self._editor_uid:
            self._close_gain_previews()
        self._editor_uid = uid
        self._step_feedback.clear()
        if uid:
            step = self._pipeline_list.steps()[index]
            name = self._display_name(step['method_id'], step['label'])
            self._step_tools_title.setText(f'{index + 1}. {name}')
            pending = self._step_edits.get(uid)
            if pending is not None and pending[0] != step['params']:
                self._step_edits.pop(uid)
                pending = None
            self._load_parameter_values(
                step['method_id'], pending[1] if pending else self._editor_parameters(step))
        else:
            self._param_form.clear()
        self._sync_step_tools()

    def _load_parameter_values(self, method_id, params):
        schema = self._methods_by_id.get(method_id, {}).get('parameter_schema') or []
        self._param_form.set_schema(schema)
        values = {}
        for field in schema:
            key = field.get('name')
            value = params.get(key, field.get('default'))
            if value is None:
                value = '' if field.get('type') == 'str' else 0
            values[key] = value
        self._param_form.set_values(values)

    def _restore_default_params(self):
        if not self._can_edit_chain():
            return
        if self._picker_context:
            method_id = self._method_browser.current_method_id()
        else:
            index = self._pipeline_list.index_of_step(self._editor_uid)
            method_id = self._pipeline_list.steps()[index]['method_id'] if index >= 0 else ''
        if method_id:
            self._load_parameter_values(method_id, {})
            self._stage_step_params()

    def _stage_step_params(self):
        uid = self._editor_uid
        index = self._pipeline_list.index_of_step(uid)
        if self._picker_context is None and self._can_edit_chain() and index >= 0:
            saved = self._pipeline_list.steps()[index]['params']
            self._step_edits[uid] = (deepcopy(saved), self._param_form.values())
        self._step_feedback.clear()
        self._sync_step_tools()
        self._sync_autotune()

    def _load_selected_step(self):
        self._step_edits.pop(self._editor_uid, None)
        self._show_selected_step()
        self._sync_autotune()

    def _commit_step_form(self):
        if not self._can_edit_chain():
            return
        if self._picker_context is not None:
            self._commit_library_choice()
            return
        index = self._pipeline_list.index_of_step(self._editor_uid)
        if index < 0:
            return
        saved = self._pipeline_list.steps()[index]['params']
        pending = self._step_edits.get(self._editor_uid)
        if pending is None or pending[0] != saved:
            return
        known = self._param_form._editors
        params = {name: value for name, value in saved.items() if name not in known}
        params.update(self._param_form.values())
        self._step_edits.pop(self._editor_uid)
        self._pipeline_list.update_step_params(index, params)
        self._step_feedback.setText('已应用到草稿')
        self._sync_step_tools()

    def _commit_library_choice(self):
        method_id = self._method_browser.current_method_id()
        if not self._valid_library_choice(method_id):
            return
        target = self._picker_context[1]
        params = self._param_form.values()
        if target is None:
            self._append_library_method(method_id, params)
            return
        index, epoch = target
        if epoch != self._draft_edit_epoch or not self._pipeline_list.step_uid(index):
            self._step_feedback.setText('处理链已变化，请重新选择要替换的步骤')
            self._sync_step_tools()
            return
        self._picker_context = None
        self._close_gain_previews()
        self._replace_step_method(index, method_id, params)

    def _sync_step_tools(self):
        browsing = self._picker_context is not None
        has_step = bool(self._editor_uid)
        can_edit = self._can_edit_chain()
        has_schema = bool(self._param_form._schema)
        pending = self._editor_uid in self._step_edits
        self._step_tools_title.setVisible(browsing or has_step)
        self._reset_params_btn.setVisible(has_schema)
        self._reset_params_btn.setEnabled(can_edit)
        self._close_library_btn.setVisible(browsing)
        self._library_escape.setEnabled(browsing)
        self._method_browser.setVisible(browsing)
        self._method_browser.setEnabled(can_edit)
        self._step_detail.setVisible(browsing and bool(self._step_detail.text()))
        self._param_form.setVisible(has_schema)
        self._param_form.setEnabled(can_edit)
        self._param_empty_hint.setVisible(not browsing and not has_schema)
        self._param_empty_hint.setText('此算法无可编辑参数' if has_step else '未选择处理步骤')
        self._discard_step_btn.setVisible(not browsing and pending)
        self._discard_step_btn.setEnabled(can_edit)
        self._commit_step_btn.setVisible(browsing or has_schema)
        if browsing:
            replacing = self._picker_context[1] is not None
            self._commit_step_btn.setText('确认替换' if replacing else '添加到处理链')
            self._commit_step_btn.setEnabled(self._valid_library_choice(self._method_browser.current_method_id()))
        else:
            self._commit_step_btn.setText('应用到草稿')
            self._commit_step_btn.setEnabled(can_edit and pending)
        self._step_feedback.setVisible(bool(self._step_feedback.text()))

    def _focus_pending_edit(self):
        for index in range(self._pipeline_list.count()):
            if self._pipeline_list.step_uid(index) in self._step_edits:
                self._open_step_editor(index)
                self._step_feedback.setText('请先应用或放弃参数修改')
                self._sync_step_tools()
                return True
        return False
