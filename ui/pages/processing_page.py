"""Processing workbench: saved result tabs, per-input drafts and frozen submissions."""
from collections.abc import Mapping
from copy import deepcopy
import json

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QScrollArea, QSplitter,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    CaptionLabel, CheckBox, ComboBox, LineEdit, MessageBox,
    PrimaryPushButton, ProgressBar, PushButton, TransparentToolButton,
    isDarkTheme,
)
from qfluentwidgets import FluentIcon as FIF

from ui import constants, design_tokens
from ui.file_tree import artifact_parameters, artifact_direct_parent, artifact_step, build_lineage_model
from ui.motion import animate_progress
from ui.page_scaffold import make_form_row, refill_combo
from ui.widgets import ParamForm, PipelineList, clear_invalid, make_separator
from ui.widgets.bscan_workspace import BScanWorkspace
from ui.widgets.collapsible_card import make_collapsible_card
from ui.widgets.object_context_menu import ObjectMenuBinding
from ui.widgets.elided_caption import ElidedCaptionLabel
from ui.pages.processing_steps import ProcessingSteps
from ui.pages.processing_menus import ProcessingMenus
from ui.pages.processing_chain_view import ProcessingChainView
from ui.pages.processing_comparison import LineComparison
from ui.widgets.line_drag_combo import LineDragComboBox
from ui.result_presentation import describe_artifact


class ProcessingPage(ProcessingChainView, ProcessingSteps, ProcessingMenus, QWidget):
    run_requested = pyqtSignal(dict)
    batch_run_requested = pyqtSignal(dict)
    estimate_requested = pyqtSignal(dict)
    cancel_requested = pyqtSignal()
    autotune_requested = pyqtSignal(str, dict, str)
    line_changed = pyqtSignal(str)
    artifact_selected = pyqtSignal(str)
    artifact_preview_requested = pyqtSignal(str)
    original_preview_requested = pyqtSignal()
    artifact_delete_requested = pyqtSignal(str, list)
    artifact_focus_requested = pyqtSignal(str, str)
    project_resources_toggled = pyqtSignal(bool)
    bscan_view_created = pyqtSignal(object)
    input_activated = pyqtSignal(str, str)
    comparison_preview_requested = pyqtSignal(str, str, str)
    comparison_artifacts_requested = pyqtSignal(str, str)
    comparison_artifact_preview_requested = pyqtSignal(str, str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._methods, self._methods_by_id, self._artifacts_by_id = [], {}, {}
        self._lines = []
        self._sources = {}
        self._drafts, self._draft_backups, self._layouts = {}, {}, {}
        self._project_id, self._line_id, self._selected_source_key = '', '', ''
        self._selected_revision, self._raw_revision = '', ''
        self._draft_edit_epoch = 0
        self._project_read_only = False
        self._running, self._autotune_running = False, False
        self._run_snapshot, self._completion_pending = None, None
        self._completed_artifact = ''
        self._activation_epoch, self._suspend, self._raw_opened = 0, False, False
        self._source_selected_this_line = False
        self._preview_requested = set()
        self._selected_step, self._selected_method_id = -1, ''
        self._method_picker, self._step_editor = None, None
        self._picker_purpose, self._picker_index = 'append', -1
        self._autotune_result, self._autotune_context = None, None
        self._autotune_generation = 0
        self._autotune_pending_request = None
        self._autotune_result_generation = None
        self._sm, self._job_id, self._pending_layout = None, '', None
        self._build_ui()
        self._comparison = LineComparison(self)
        self._connect_internal()
        self._refresh_ui()
        self.apply_theme(isDarkTheme())

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(8)
        self._resources_btn = TransparentToolButton(FIF.FOLDER, self)
        self._tools_btn = TransparentToolButton(FIF.DEVELOPER_TOOLS, self)
        for button, text in ((self._resources_btn, '项目资源'), (self._tools_btn, '处理工具')):
            button.setCheckable(True)
            button.setToolTip(text)
            button.setAccessibleName(text)
            button.setFixedSize(*constants.TOOL_BTN_SIZE)
            header.addWidget(button)
        self._tools_btn.setChecked(True)
        self._line_combo = LineDragComboBox(self)
        self._line_combo.setPlaceholderText('选择测线')
        self._line_combo.setMinimumWidth(100)
        self._line_combo.setMaximumWidth(160)
        header.addWidget(self._line_combo)
        self._input_label = ElidedCaptionLabel(self)
        self._input_label.setText('下次输入：未选择数据')
        header.addWidget(self._input_label, 1)
        self._layout_combo = ComboBox(self)
        self._layout_combo.addItems(['单组标签', '左右对照', '上下对照', '自定义布局'])
        self._layout_combo.setToolTip('手动设置视图布局；打开标签不会自动分屏')
        header.addWidget(self._layout_combo)
        self._source_path_btn = TransparentToolButton(FIF.HISTORY, self)
        self._source_path_btn.setCheckable(True)
        self._source_path_btn.setChecked(True)
        self._source_path_btn.setFixedSize(*constants.TOOL_BTN_SIZE)
        self._source_path_btn.setToolTip('已完成来源：展开 / 收起')
        self._source_path_btn.setAccessibleName('已完成来源')
        self._source_path_btn.toggled.connect(self._sync_chain_view)
        header.addWidget(self._source_path_btn)
        self._run_btn = PrimaryPushButton('从此结果运行', self, FIF.PLAY)
        self._run_btn.setToolTip('使用当前激活结果执行新增步骤（Ctrl+R）')
        self._cancel_btn = PushButton('取消任务', self, FIF.CANCEL)
        self._batch_btn = PushButton('批量应用…', self, FIF.PLAY)
        self._batch_btn.setToolTip('将当前处理链应用到多条测线的原始输入')
        header.addWidget(self._batch_btn)
        header.addWidget(self._run_btn)
        header.addWidget(self._cancel_btn)
        root.addLayout(header)
        self._body = QSplitter(Qt.Orientation.Horizontal, self)
        self._body.setChildrenCollapsible(False)
        self._tools_panel = QTabWidget(self._body)
        self._tools_panel.setObjectName('processingTools')
        self._tools_panel.setMinimumWidth(250)
        self._tools_panel.setMaximumWidth(360)
        self._tools_panel.setDocumentMode(True)
        chain_page = QWidget()
        chain_layout = QVBoxLayout(chain_page)
        chain_layout.setContentsMargins(8, 8, 8, 8)
        chain_layout.setSpacing(8)
        self._lineage_tree = QTreeWidget(chain_page)
        self._lineage_tree.setHeaderHidden(True)
        self._lineage_tree.setIndentation(12)
        self._lineage_tree.setMinimumHeight(100)
        self._lineage_tree.setMaximumHeight(160)
        self._lineage_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        sources_header = QHBoxLayout()
        sources_header.addWidget(CaptionLabel('数据来源与分支', chain_page), 1)
        self._lineage_more_btn = TransparentToolButton(FIF.MORE, chain_page)
        self._lineage_more_btn.setToolTip('来源操作')
        self._lineage_more_btn.setAccessibleName('来源操作')
        sources_header.addWidget(self._lineage_more_btn)
        chain_layout.addLayout(sources_header)
        chain_layout.addWidget(self._lineage_tree)
        self._draft_menu_btn = TransparentToolButton(FIF.MORE, chain_page)
        self._draft_menu_btn.setToolTip('草稿操作')
        self._draft_menu_btn.setAccessibleName('草稿操作')
        self._pipeline_list = PipelineList(self)
        self._pipeline_list.hide()  # Internal step model; the editable chain is its only visible list.
        self._add_method_btn = PushButton('添加处理步骤', chain_page, FIF.ADD)
        self._add_method_btn.hide()  # Compatibility command; the visible append action is in the chain.
        chain_layout.addWidget(make_separator())
        chain_layout.addWidget(CaptionLabel('参数与调参', chain_page))
        self._param_empty_hint = CaptionLabel('点击上方算法节点编辑参数', chain_page)
        self._param_empty_hint.setWordWrap(True)
        chain_layout.addWidget(self._param_empty_hint)
        self._param_form = ParamForm(chain_page)
        chain_layout.addWidget(self._param_form)
        self._result_name_edit = LineEdit(chain_page)
        self._result_name_edit.setPlaceholderText('可选，留空自动命名')
        chain_layout.addLayout(make_form_row('结果名称', self._result_name_edit, parent=chain_page))
        self._execution_card, execution = make_collapsible_card('执行与资源', collapsed=False)
        self._save_intermediates_check = CheckBox('保存每步中间成果', self._execution_card)
        self._save_intermediates_check.setChecked(True)
        self._save_intermediates_check.setToolTip('默认保存并逐步展示每个处理结果；大测线可关闭以优先分块和降低资源占用')
        execution.addWidget(self._save_intermediates_check)
        self._estimate_btn = PushButton('估算运行资源', self._execution_card, FIF.SEARCH)
        execution.addWidget(self._estimate_btn)
        self._estimate_label = CaptionLabel('默认逐步展示结果；关闭中间成果后可优先分块', self._execution_card)
        self._estimate_label.setWordWrap(True)
        execution.addWidget(self._estimate_label)
        chain_layout.addWidget(self._execution_card)
        self._autotune_card, tune = make_collapsible_card('自动调参', collapsed=True)
        self._autotune_method_label = CaptionLabel('--', self._autotune_card)
        tune.addWidget(self._autotune_method_label)
        self._autotune_btn = PushButton('开始调参', self._autotune_card)
        self._autotune_result_label = CaptionLabel('暂无调参结果', self._autotune_card)
        self._autotune_result_label.setWordWrap(True)
        self._adopt_params_btn = PushButton('采用最优参数', self._autotune_card)
        self._adopt_params_btn.setEnabled(False)
        tune.addWidget(self._autotune_btn)
        tune.addWidget(self._autotune_result_label)
        tune.addWidget(self._adopt_params_btn)
        chain_layout.addWidget(self._autotune_card)
        chain_layout.addStretch(1)
        scroll = QScrollArea(self._tools_panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(chain_page)
        self._tools_panel.addTab(scroll, '处理')
        display = QWidget()
        dl = QVBoxLayout(display)
        hint = CaptionLabel('仅改变当前视图；不修改数据或处理步骤', display)
        hint.setWordWrap(True)
        dl.addWidget(hint)
        self._display_cmap = ComboBox(display)
        self._display_cmap.addItems(list(constants.COLORMAPS))
        dl.addLayout(make_form_row('色图', self._display_cmap, parent=display))
        fit = PushButton('适应当前视图', display)
        fit.clicked.connect(lambda: self._workspace.primary_view().fit_to_data())
        dl.addWidget(fit)
        dl.addWidget(CaptionLabel('色阶、显示增益、坐标与导出可在画布右键调整', display))
        dl.addStretch(1)
        self._tools_panel.addTab(display, '显示')
        history_page = QWidget()
        history_layout = QVBoxLayout(history_page)
        history_layout.setContentsMargins(8, 8, 8, 8)
        history_header = QHBoxLayout()
        history_header.addWidget(CaptionLabel('已保存运行', history_page), 1)
        self._history_more_btn = TransparentToolButton(FIF.MORE, history_page)
        self._history_more_btn.setToolTip('历史运行操作')
        self._history_more_btn.setAccessibleName('历史运行操作')
        history_header.addWidget(self._history_more_btn)
        history_layout.addLayout(history_header)
        self._history_tree = QTreeWidget(history_page)
        self._history_tree.setHeaderHidden(True)
        self._history_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        history_layout.addWidget(self._history_tree)
        self._tools_panel.addTab(history_page, '历史')
        center = QWidget(self._body)
        main = QVBoxLayout(center)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(4)
        self._path_label = CaptionLabel('', center)
        self._path_label.setWordWrap(True)
        self._path_label.hide()
        self._build_chain_view(main)
        self._draft_revision_hint = CaptionLabel('', center)
        self._draft_revision_hint.setWordWrap(True)
        main.addWidget(self._draft_revision_hint)
        preview_status = QHBoxLayout()
        self._preview_error_label = CaptionLabel('', center)
        self._preview_error_label.setWordWrap(True)
        self._preview_retry_btn = PushButton('重试预览', center, FIF.SYNC)
        self._preview_retry_btn.clicked.connect(lambda: self.retry_preview(self._selected_source_key))
        preview_status.addWidget(self._preview_error_label, 1)
        preview_status.addWidget(self._preview_retry_btn)
        main.addLayout(preview_status)
        self._workspace = BScanWorkspace(center)
        self._bscan_container = self._workspace
        main.addWidget(self._workspace, 1)
        self._body.setStretchFactor(0, 0)
        self._body.setStretchFactor(1, 1)
        self._body.setSizes([300, 850])
        root.addWidget(self._body, 1)
        progress = QHBoxLayout()
        self._progress_bar = ProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_label = CaptionLabel('', self)
        self._progress_label.setWordWrap(True)
        progress.addWidget(self._progress_bar, 1)
        progress.addWidget(self._progress_label, 2)
        self._progress_row_widget = QWidget(self)
        self._progress_row_widget.setLayout(progress)
        self._progress_row_widget.hide()
        root.addWidget(self._progress_row_widget)
        self._completion_btn = PushButton('查看新结果', self)
        self._completion_btn.hide()
        root.addWidget(self._completion_btn)

    def _apply_polish_theme(self, dark):
        t = design_tokens.tokens(dark)
        self._tools_panel.setStyleSheet(
            f'QTabWidget#processingTools::pane {{ border: 1px solid {t["border_default"]};'
            f' background: {t["bg_surface"]}; border-radius: 6px; }}'
            f'QTabBar::tab {{ background: transparent; color: {t["text_secondary"]};'
            ' padding: 6px 12px; border: none; border-bottom: 2px solid transparent; }'
            f'QTabBar::tab:selected {{ color: {t["text_primary"]};'
            f' border-bottom: 2px solid {t["primary"]}; }}')
        self._input_label.setStyleSheet(f'color: {t["text_secondary"]};')
        self._param_empty_hint.setStyleSheet(f'color: {t["text_muted"]};')
        self._draft_revision_hint.setStyleSheet(f'color: {t["warning"]};')
        self._preview_error_label.setStyleSheet(f'color: {t["error"]};')
        self._chain_panel.setStyleSheet(
            f'QWidget#processingChainPanel {{ background: {t["bg_subtle"]};'
            f' border: 1px solid {t["border_default"]}; border-radius: 6px; }}')

    def _connect_internal(self):
        self._connect_chain_view()
        self._line_combo.currentIndexChanged.connect(self._on_line_combo_changed)
        self._pipeline_list.sig_step_selected.connect(self._on_step_selected)
        self._pipeline_list.sig_edit_requested.connect(self._open_step_editor)
        self._pipeline_list.sig_duplicate_requested.connect(self._duplicate_step)
        self._pipeline_list.sig_changed.connect(self._draft_changed)
        self._param_form.sig_changed.connect(self._auto_write_params_to_selected)
        self._result_name_edit.textChanged.connect(self._draft_changed)
        self._add_method_btn.clicked.connect(self._open_method_picker)
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._batch_btn.clicked.connect(self._on_batch_clicked)
        self._estimate_btn.clicked.connect(self._on_estimate_clicked)
        self._save_intermediates_check.toggled.connect(self._invalidate_estimate)
        self._cancel_btn.clicked.connect(self.cancel_requested)
        self._workspace.active_changed.connect(self._activate_source)
        self._workspace.source_activated_by_user.connect(self._activate_comparison_source)
        self._workspace.preview_requested.connect(self._request_preview)
        self._workspace.source_context_requested.connect(self._show_source_menu)
        self._workspace.layout_changed.connect(self._sync_layout_label)
        self._workspace.view_created.connect(self.bscan_view_created)
        self._tools_btn.toggled.connect(self._toggle_tools)
        self._resources_btn.toggled.connect(self._toggle_resources)
        self._layout_combo.currentIndexChanged.connect(self._change_layout)
        self._lineage_tree.itemClicked.connect(self._open_tree_item)
        self._history_tree.itemClicked.connect(self._open_tree_item)
        self._lineage_menu_binding = ObjectMenuBinding(self, self._lineage_tree,
            lambda pos: self._tree_menu(self._lineage_tree, pos))
        self._history_menu_binding = ObjectMenuBinding(self, self._history_tree,
            lambda pos: self._tree_menu(self._history_tree, pos))
        self._lineage_more_btn.clicked.connect(self._lineage_menu_binding.show_keyboard)
        self._history_more_btn.clicked.connect(self._history_menu_binding.show_keyboard)
        self._draft_menu_btn.clicked.connect(self._show_draft_menu)
        self._display_cmap.currentTextChanged.connect(self._set_display_cmap)
        self._completion_btn.clicked.connect(self.show_latest_result)
        self._autotune_btn.clicked.connect(self._on_autotune_clicked)
        self._adopt_params_btn.clicked.connect(self._on_adopt_params)
        self._run_shortcut = QShortcut(QKeySequence('Ctrl+R'), self)
        self._run_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._run_shortcut.activated.connect(self._on_run_clicked)

    def _draft_key(self, source=None):
        key = self._selected_source_key if source is None else source
        revision = (self._selected_revision if source is None
                    else str(self._sources.get(key, {}).get('revision') or ''))
        return (self._project_id, self._line_id, key, revision)

    def _stash_draft(self):
        if self._suspend or not self._selected_source_key:
            return
        self._drafts[self._draft_key()] = deepcopy(self.current_pipeline())

    def _draft_changed(self, *_):
        if self._suspend:
            return
        self._draft_edit_epoch += 1
        self._stash_draft()
        self._refresh_ui()
        self._save_session()

    def _draft_locked(self):
        return bool(self._running and self._run_snapshot and
                    self._run_snapshot['key'] == self._draft_key())

    def _activate_source(self, key):
        if self._suspend:
            return
        key = str(key or '')
        previous_comparison = self._comparison.active
        self._comparison.active = key if self._comparison.owns(key) else ''
        if self._comparison.active:
            if self._comparison.active != previous_comparison:
                self._activation_epoch += 1
            self._refresh_ui()
            return
        revision = str(self._sources.get(key, {}).get('revision') or '')
        if key != self._selected_source_key or revision != self._selected_revision:
            self._stash_draft()
            self._selected_source_key = key
            self._selected_revision = revision
            if key:
                self._source_selected_this_line = True
                self._pending_layout = None  # an explicit selection supersedes session restoration
            self._activation_epoch += 1
            self._close_chain_popups()
            draft = self._drafts.get(self._draft_key(), {})
            self._suspend = True
            self._pipeline_list.set_steps(deepcopy(draft.get('steps') or []))
            self._result_name_edit.setText(str(draft.get('result_name') or ''))
            self._suspend = False
            self._selected_step = -1
            self._param_form.clear()
            if self._pipeline_list.count():
                self._pipeline_list.select_step(0)
        self._refresh_ui()
        self._save_session()
        self.input_activated.emit(self._line_id if self._selected_source_key else '', self._input_artifact_id())

    def _activate_comparison_source(self, key):
        if self._comparison.owns(str(key or '')):
            self._comparison.make_input(str(key))

    def _refresh_ui(self):
        source = self._sources.get(self._selected_source_key)
        self._input_label.setText('下次输入：' + (source['title'] if source else '未选择数据'))
        steps = self._pipeline_list.steps()
        error = str(source.get('error') or '') if source else ''
        valid = bool(source and not error and not self._project_read_only
                     and any(s.get('enabled', True) for s in steps))
        self._run_btn.setEnabled(valid and not self._running)
        self._batch_btn.setEnabled(valid and not self._running and len(self._lines) > 0)
        self._estimate_btn.setEnabled(valid and not self._running)
        self._save_intermediates_check.setEnabled(not self._running and not self._project_read_only)
        self._run_btn.setText('任务运行中…' if self._running else '从此结果运行')
        self._run_btn.setToolTip('项目只读，无法提交处理任务' if self._project_read_only else
            ('输入读取失败，请重试预览：' + error) if error else
            '请先选择有效数据并添加启用步骤' if not valid else '使用所选结果执行新增步骤（Ctrl+R）')
        self._preview_error_label.setText(error)
        self._preview_error_label.setVisible(bool(error))
        self._preview_retry_btn.setVisible(bool(error))
        stale = any(k[:3] == self._draft_key()[:3] and k != self._draft_key()
                    and value.get('steps') for k, value in self._drafts.items())
        self._draft_revision_hint.setText('此数据已更换修订；旧修订草稿已失效并隔离保存。当前修订从空草稿开始。' if stale else '')
        self._draft_revision_hint.setVisible(stale)
        self._cancel_btn.setVisible(self._running)
        self._cancel_btn.setEnabled(self._running)
        locked = self._draft_locked()
        for widget in (self._pipeline_list, self._param_form, self._add_method_btn, self._result_name_edit):
            widget.setEnabled(not locked)
        self._sync_param_empty_hint()
        self._update_autotune_enabled()
        self._adopt_params_btn.setEnabled(self._can_adopt_autotune())
        path, seen, aid = [], set(), self._input_artifact_id()
        while aid and aid in self._artifacts_by_id and aid not in seen:
            seen.add(aid)
            art = self._artifacts_by_id[aid]
            path.insert(0, str(getattr(art, 'name', '') or aid))
            parent = artifact_direct_parent(art)
            if parent is None:
                path.insert(0, '来源待核对')
                break
            aid = parent
        if self._selected_source_key and (not path or path[0] != '来源待核对'):
            path.insert(0, '原始输入')
        self._path_label.setText(' › '.join(path))
        self._sync_chain_view()
        if self._comparison.active:
            source = self._comparison.sources[self._comparison.active]
            self._input_label.setText(f'仅对比：{source["line_id"]}；处理输入仍为 {self._line_id} · '
                                     + str(self._sources.get(self._selected_source_key, {}).get('title', '未选择')))
            self._run_btn.setEnabled(False)
            self._run_btn.setToolTip('此视图仅用于对比；请从标签菜单选择“设为处理输入”或回到处理输入标签')

    def _sync_sources(self):
        sources = [dict(source, title=f'{self._comparison.label(self._line_id)} · 原始输入')
                   if source['key'] == 'original' and self._line_id else source
                   for source in self._sources.values()]
        self._workspace.set_sources(sources + list(self._comparison.sources.values()))

    def set_comparison_preview(self, project_id, line_id, token, bundle, error):
        self._comparison.deliver(project_id, line_id, token, bundle, error)

    def set_original_bundle(self, bundle):
        if bundle is None:
            self._sources.pop('original', None)
            self._sync_sources()
            self._refresh_ui()
            return
        self._sources['original'] = {'key': 'original', 'title': '原始输入', 'bundle': bundle,
                                     'artifact_id': '', 'revision': self._raw_revision}
        self._preview_requested.discard('original')
        self._sync_sources()
        self._maybe_open_initial_raw()
        self._restore_workspace()
        self._activate_source(self._workspace.active_key())

    def set_artifact_bundle(self, artifact_id, bundle):
        key = 'artifact:' + str(artifact_id)
        if key not in self._sources:
            return  # late preview from another/deleted context
        self._sources[key]['bundle'] = bundle
        self._sources[key].pop('error', None)
        self._preview_requested.discard(str(artifact_id))
        self._sync_sources()  # update existing views only; never reopen/activate
        self._refresh_ui()

    def set_original_revision(self, project_id, line_id, revision):
        if (str(project_id), str(line_id)) != (self._project_id, self._line_id):
            return
        self._raw_revision = str(revision or '')

    def set_preview_failure(self, project_id, line_id, artifact_id, message):
        """Only the matching saved source owns an asynchronous read failure."""
        if (str(project_id), str(line_id)) != (self._project_id, self._line_id):
            return
        aid = str(artifact_id or '')
        key = 'artifact:' + aid if aid else 'original'
        if key not in self._sources:
            if aid:
                return
            self._sources[key] = {'key': key, 'title': '原始输入', 'artifact_id': '',
                                  'revision': self._raw_revision, 'bundle': None}
        self._sources[key]['error'] = str(message or '无法读取数据')
        self._sources[key]['bundle'] = None
        self._preview_requested.discard(aid or 'original')
        self._sync_sources()  # error clears the workspace's in-flight request marker
        if not aid:
            self._maybe_open_initial_raw()
        self._refresh_ui()

    def _maybe_open_initial_raw(self):
        if self._raw_opened:
            return
        self._raw_opened = True
        if (not self._source_selected_this_line and self._pending_layout is None
                and not self._workspace.tab_keys()):
            self._workspace.open_source('original')

    def retry_preview(self, key):
        source = self._sources.get(str(key))
        if not source or not source.get('error'):
            return False
        source.pop('error', None)
        self._preview_requested.discard(str(source.get('artifact_id') or 'original'))
        self._sync_sources()
        self._request_preview(key)  # also works for an inactive source, without activation
        self._refresh_ui()
        return True

    def _request_preview(self, key):
        if self._suspend:
            return
        if self._comparison.owns(key):
            self._comparison.request(key)
            return
        source = self._sources.get(str(key))
        aid = str(source.get('artifact_id') or '') if source else ''
        if aid and aid not in self._preview_requested:
            self._preview_requested.add(aid)
            self.artifact_preview_requested.emit(aid)
        elif source and not aid and 'original' not in self._preview_requested:
            self._preview_requested.add('original')
            self.original_preview_requested.emit()

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
        self._lineage_tree.clear()
        pending = [(None, node) for node in reversed(build_lineage_model(
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
        if 'original' in self._sources:
            self._workspace.open_source('original')

    def close_artifact_tab(self, artifact_id):
        self._workspace.close_source('artifact:' + str(artifact_id))

    def close_all_artifact_tabs(self):
        self._stash_draft()
        self._save_session()
        self._suspend = True
        for aid in self._artifacts_by_id:
            self._workspace.close_source('artifact:' + aid)
        self._suspend = False
        self._activate_source(self._workspace.active_key())

    def show_latest_result(self):
        if self._completed_artifact:
            self.select_artifact(self._completed_artifact)
            self._completion_btn.hide()

    def set_project_context(self, project_id):
        project_id = str(project_id or '')
        if project_id != self._project_id:
            self._stash_draft()
            self._save_session()
            self._comparison.reset()
            self._switch_line('')
            self._project_id = project_id
            self._project_read_only = False
            self._pending_layout = None

    def set_project_read_only(self, read_only):
        self._project_read_only = bool(read_only)
        self._refresh_ui()

    def set_line_label(self, text):
        line_id = str(text or '')
        self._set_line_combo_without_emit(line_id)
        if line_id != self._line_id:
            self._switch_line(line_id)

    def _switch_line(self, line_id):
        self._close_chain_popups()
        self._stash_draft()
        self._save_session()
        self._suspend = True
        self._sources, self._artifacts_by_id = {}, {}
        comparing = self._comparison.has_open()
        if comparing and line_id:
            self._sources['original'] = dict(key='original', title='原始输入',
                                              bundle=None, artifact_id='', revision='')
        self._line_id, self._selected_source_key = str(line_id), ''
        self._sync_sources()
        self._selected_revision, self._raw_revision = '', ''
        self._pipeline_list.set_steps([])
        self._result_name_edit.clear()
        self._param_form.clear()
        self._preview_requested.clear()
        self._raw_opened = False
        self._source_selected_this_line = False
        self._activation_epoch += 1
        self._pending_layout = None if comparing else self._layouts.get((self._project_id, self._line_id))
        self._suspend = False
        self._render_trees()
        if comparing and line_id:
            gid = next((gid for gid in self._workspace.group_ids()
                        if 'original' in self._workspace.tab_keys(gid)), None)
            self._workspace.open_source('original', gid)
            self._activate_source('original')
        self._refresh_ui()

    def _input_artifact_id(self):
        return self._selected_source_key.removeprefix('artifact:') if self._selected_source_key.startswith('artifact:') else ''

    def current_pipeline(self):
        return {'steps': self._pipeline_list.steps(), 'result_name': self._result_name_edit.text().strip(),
                'input_artifact_id': self._input_artifact_id(),
                'save_intermediates': self._save_intermediates_check.isChecked()}

    def set_running(self, running, job_id='', success=None):
        running = bool(running)
        if running and not self._running:
            self._stash_draft()
            self._run_snapshot = {'key': self._draft_key(), 'input_artifact_id': self._input_artifact_id(),
                'pipeline': deepcopy(self.current_pipeline()), 'epoch': self._activation_epoch,
                'artifacts': set(self._artifacts_by_id), 'activate': True}
            if self._step_editor is not None:
                self._step_editor.close()
            self._progress_bar.setValue(0)
            self._progress_label.setText('输入：' + self._input_label.text().removeprefix('下次输入：'))
        elif not running and self._running and self._run_snapshot:
            if success:
                snap = self._run_snapshot
                self._drafts[snap['key']] = {'steps': [], 'result_name': '', 'input_artifact_id': snap['input_artifact_id']}
                if snap['key'] == self._draft_key():
                    self._suspend = True
                    self._pipeline_list.set_steps([])
                    self._result_name_edit.clear()
                    self._suspend = False
                self._completion_pending = dict(snap)
            self._run_snapshot = None
        self._running, self._job_id = running, str(job_id or '')
        self._progress_row_widget.setVisible(running)
        self._refresh_ui()
        self._save_session()

    def set_run_result(self, job_id, artifact_id):
        """Use the job's actual output identity, rather than a newest-result guess."""
        if self._run_snapshot is not None and str(job_id) == self._job_id:
            self._run_snapshot['result_id'] = str(artifact_id or '')

    def hideEvent(self, event):
        if self._run_snapshot:
            self._run_snapshot['activate'] = False
        super().hideEvent(event)

    def _on_run_clicked(self):
        if self._running:
            self.cancel_requested.emit()
            return
        if self._run_btn.isEnabled():
            clear_invalid(self._result_name_edit)
            self.run_requested.emit(deepcopy(self.current_pipeline()))

    def _on_batch_clicked(self):
        if self._running or self._project_read_only:
            return
        steps = [step for step in self._pipeline_list.steps() if step.get('enabled', True)]
        if not steps:
            self._estimate_label.setText('请先添加至少一个启用步骤')
            return
        from ui.pages.processing_batch import BatchProcessingDialog
        dialog = BatchProcessingDialog(self._lines, steps, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.batch_run_requested.emit(dialog.payload())

    def _on_estimate_clicked(self):
        if not any(step.get('enabled', True) for step in self._pipeline_list.steps()):
            self._estimate_label.setText('请先添加至少一个启用步骤')
            return
        self._estimate_label.setText('正在估算…')
        self.estimate_requested.emit(deepcopy(self.current_pipeline()))

    def _invalidate_estimate(self, *_args):
        self._estimate_label.setText('配置已变化，请重新估算')

    def set_estimate_result(self, estimate):
        def size(value):
            value = float(value or 0)
            return f'{value / 1024 ** 3:.2f} GB' if value >= 1024 ** 3 else f'{value / 1024 ** 2:.1f} MB'
        mode = '分块执行' if getattr(estimate, 'supports_chunking', False) else '整线加载'
        self._estimate_label.setText(
            f'{mode}｜预计峰值内存 {size(getattr(estimate, "memory_bytes", 0))}'
            f'｜临时磁盘 {size(getattr(estimate, "temporary_disk_bytes", 0))}'
        )
        self._estimate_label.setToolTip('\n'.join(getattr(estimate, 'notes', ()) or ()))

    def set_estimate_failed(self, message):
        self._estimate_label.setText(f'估算失败：{message}')

    def reconfigure_artifact(self, artifact_id):
        art = self._artifacts_by_id.get(str(artifact_id))
        if art is None:
            return False
        parent, step = artifact_direct_parent(art), artifact_step(art)
        if parent is None or not step:
            return False
        if not self.can_reconfigure_artifact(artifact_id):
            return False
        source = 'artifact:' + parent if parent else 'original'
        if source not in self._sources:
            return False
        if self._running and self._run_snapshot and self._run_snapshot['key'] == self._draft_key(source):
            return False
        method = str(step.get('method_id') or getattr(art, 'method_id', ''))
        new = [{'method_id': method, 'label': self._display_name(method),
                'params': deepcopy(step.get('params') or {}), 'enabled': True}]
        return self._replace_draft(source, new)

    def can_reconfigure_artifact(self, artifact_id):
        art = self._artifacts_by_id.get(str(artifact_id))
        if art is None or artifact_direct_parent(art) is None or not artifact_step(art):
            return False
        meta = artifact_parameters(art)
        index = int(meta.get('run_step_index') or 0)
        if index <= 1:
            return True
        previous = self._artifacts_by_id.get(artifact_direct_parent(art))
        pmeta = artifact_parameters(previous) if previous else {}
        return (pmeta.get('run_group_id') == meta.get('run_group_id')
                and int(pmeta.get('run_step_index') or 0) == index - 1)

    def _saved_run(self, group):
        members = [art for art in self._artifacts_by_id.values()
                   if artifact_parameters(art).get('run_group_id') == group]
        if not members:
            return None
        final = max(members, key=lambda art: int(artifact_parameters(art).get('run_step_index') or 0))
        meta = artifact_parameters(final)
        pipeline = meta.get('pipeline')
        if (meta.get('lineage_version') != 2 or 'run_input_artifact_id' not in meta
                or not pipeline or meta.get('artifact_kind') == 'intermediate'):
            return None
        source = 'artifact:' + str(meta['run_input_artifact_id']) if meta['run_input_artifact_id'] else 'original'
        if source not in self._sources or self._sources[source].get('error'):
            return None
        if self._running and self._run_snapshot and self._run_snapshot['key'] == self._draft_key(source):
            return None
        expected = str(meta.get('run_input_sha256') or '')
        actual = str(self._sources[source].get('revision') or '')
        if expected and actual and expected != actual:
            return None
        if not isinstance(pipeline, (list, tuple)) or not all(
                isinstance(step, Mapping) and step.get('method_id') for step in pipeline):
            return None
        steps = [{'method_id': str(step['method_id']), 'label': self._display_name(str(step['method_id'])),
                  'params': deepcopy(dict(step.get('params') or {})), 'enabled': bool(step.get('enabled', True))}
                 for step in pipeline]
        return source, steps, members, final

    def reconfigure_run(self, group):
        """Load the persisted whole run at its frozen input, including final-only runs."""
        saved = self._saved_run(str(group))
        return self._replace_draft(saved[0], saved[1]) if saved else False

    def _replace_draft(self, source, steps):
        if source not in self._sources:
            return False
        self._stash_draft()
        key = self._draft_key(source)
        old = self._drafts.get(key, {})
        if old.get('steps'):
            summary = '\n'.join(f'{i + 1}. {self._display_name(str(step.get("method_id") or ""))} '
                                + json.dumps(step.get('params') or {}, ensure_ascii=False, default=str)
                                for i, step in enumerate(steps))
            box = MessageBox('替换此输入的待执行草稿？',
                f'输入：{self._sources[source]["title"]}\n已有 {len(old["steps"])} 步；新配置 {len(steps)} 步：\n{summary}\n旧草稿可从草稿菜单恢复。', self.window())
            box.yesButton.setText('替换并保留备份')
            if box.exec() != QDialog.DialogCode.Accepted:
                return False
            if source not in self._sources or key != self._draft_key(source):
                return False
            self._draft_backups.setdefault(key, []).append(deepcopy(old))
        self._drafts[key] = {'steps': deepcopy(steps), 'result_name': ''}
        self._workspace.open_source(source)
        self._suspend = True
        self._pipeline_list.set_steps(deepcopy(steps))
        self._result_name_edit.clear()
        self._suspend = False
        self._tools_btn.setChecked(True)
        self._tools_panel.setCurrentIndex(0)
        if steps:
            self._pipeline_list.select_step(0)
        self._draft_changed()
        return True


    def has_line_session(self, line_id):
        layout = self._layouts.get((self._project_id, str(line_id))) or {}
        nodes = [layout.get('root') or {}]
        while nodes:
            node = nodes.pop()
            if node.get('type') == 'group' and node.get('tabs'):
                return True
            nodes.extend(node.get('children') or [])
        return False


    def _toggle_tools(self, visible):
        self._tools_panel.setVisible(bool(visible))
        if visible and self._resources_btn.isChecked():
            self._resources_btn.setChecked(False)

    def _toggle_resources(self, visible):
        if visible:
            self._tools_btn.setChecked(False)
        self.project_resources_toggled.emit(bool(visible))

    def set_project_resources_visible(self, visible):
        """Mirror Ctrl+B and page-specific tree state without re-emitting the toggle."""
        self._resources_btn.blockSignals(True)
        self._resources_btn.setChecked(bool(visible))
        self._resources_btn.blockSignals(False)
        if visible:
            self._tools_btn.setChecked(False)

    def toggle_side_panels(self):
        self._tools_btn.setChecked(not self._tools_btn.isChecked())

    def _change_layout(self, index):
        key = self._workspace.active_key()
        if index == 0:
            self._workspace.reset_layout()
        elif key and index in (1, 2):
            self._workspace.split_source(key, 'right' if index == 1 else 'bottom', copy=True)

    def _sync_layout_label(self):
        root = self._workspace.save_layout().get('root') or {}
        index = 0
        if root.get('type') == 'split':
            children = root.get('children') or []
            index = (1 if root.get('orientation') == 'horizontal' else 2) if (
                len(children) == 2 and all(child.get('type') == 'group' for child in children)) else 3
        self._layout_combo.blockSignals(True)
        self._layout_combo.setCurrentIndex(index)
        self._layout_combo.blockSignals(False)

    def _set_display_cmap(self, name):
        if name:
            self._workspace.primary_view().set_colormap(str(name))


    def set_colorbar_pref(self, visible):
        for view in self._workspace.all_views():
            view.set_colorbar_visible(bool(visible), notify=False)

    def apply_theme(self, dark):
        self._apply_polish_theme(dark)
        self._workspace.apply_theme(dark)

    def set_settings_manager(self, sm):
        self._sm = sm
        saved = sm.get('processing_workspace_sessions', {}) if sm else {}
        if not isinstance(saved, Mapping):
            return
        for record in saved.get('drafts', []):
            if isinstance(record, dict) and len(record.get('key', [])) in (3, 4):
                key = tuple(record['key'])
                self._drafts[key if len(key) == 4 else (*key, '')] = record.get('value') or {}
        for record in saved.get('layouts', []):
            if isinstance(record, dict) and len(record.get('key', [])) == 2:
                self._layouts[tuple(record['key'])] = record.get('value') or {}

    def _save_session(self):
        if self._suspend or not self._line_id:
            return
        if not self._comparison.has_open():
            self._layouts[(self._project_id, self._line_id)] = self._workspace.save_layout()
        if self._sm is not None:
            self._sm.set('processing_workspace_sessions', {
                'drafts': [{'key': list(k), 'value': v} for k, v in self._drafts.items()],
                'layouts': [{'key': list(k), 'value': v} for k, v in self._layouts.items()]})

    def _restore_workspace(self):
        if self._pending_layout:
            wanted = self._pending_layout
            # Wait until all artifact metadata needed by the saved layout arrives.
            serialized = json.dumps(wanted, ensure_ascii=False)
            if 'original' in serialized and 'original' not in self._sources:
                return
            if 'artifact:' in serialized and not self._artifacts_by_id:
                return
            self._pending_layout = None
            self._raw_opened = True  # even an intentionally empty saved layout is authoritative
            self._workspace.restore_layout(wanted)


    def set_progress(self, completed: int, total: int, message: str) -> None:
        """进度更新：total>0 按比例，否则按百分数；message 显示在进度条右侧。"""
        if total and total > 0:
            self._progress_bar.setRange(0, int(total))
            animate_progress(self._progress_bar,
                             min(int(completed), int(total)))
        else:
            self._progress_bar.setRange(0, 100)
            animate_progress(self._progress_bar,
                             max(0, min(int(completed), 100)))
        self._progress_label.setText(str(message or ''))
        self._progress_label.setToolTip(str(message or ''))

    def set_lines(self, lines: list) -> None:
        """测线列表 → 处理页测线选择下拉。

        显示去重：name 缺失或与 line_id 相同时只显示 line_id
        （否则出现「L01 L01」式拼接重复）。
        """

        self._lines = list(lines or [])

        def _line_display(line) -> str:
            line_id = str(getattr(line, 'line_id', '') or '').strip()
            name = str(getattr(line, 'name', '') or '').strip()
            if not name or name == line_id:
                return line_id
            return f"{line_id} {name}"

        refill_combo(
            self._line_combo, lines or [], _line_display,
            lambda line: str(getattr(line, 'line_id', '') or ''),
            previous_data=self._line_combo.currentData())
        self._comparison.set_lines(lines or [])

    def _set_line_combo_without_emit(self, line_id: str) -> None:
        index = self._line_combo.findData(str(line_id or ''))
        if index < 0:
            index = 0 if self._line_combo.count() else -1
        if index >= 0:
            self._line_combo.blockSignals(True)
            self._line_combo.setCurrentIndex(index)
            self._line_combo.blockSignals(False)

    def _on_line_combo_changed(self, index: int) -> None:
        if index >= 0:
            self.line_changed.emit(str(self._line_combo.itemData(index) or ''))
