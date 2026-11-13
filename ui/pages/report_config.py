"""Formal report scope, section and approval configuration."""

from collections import Counter

from PyQt6.QtCore import QSize, QSignalBlocker, Qt
from PyQt6.QtWidgets import (
    QDialog, QGridLayout, QHBoxLayout, QHeaderView, QScrollArea, QSpinBox,
    QTabWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    CheckBox, ComboBox, DoubleSpinBox, LineEdit, PlainTextEdit, PrimaryPushButton, PushButton,
    setCustomStyleSheet, TreeWidget,
)

from core.report_line_content import ANALYSIS_FIELDS, MANUAL_ACQUISITION_FIELDS
from ui.page_scaffold import card_title, make_form_row, make_hint
from ui.widgets.collapsible_card import make_collapsible_card


_SECTIONS = (
    ('include_project_summary', '项目概况'),
    ('include_line_inventory', '测线台账'),
    ('include_processing_history', '处理历史'),
    ('include_bscan_images', 'B-scan 图件'),
    ('include_interpretation', '解释标注'),
    ('include_spatial_results', '空间成果'),
    ('include_borehole_comparison', '钻孔对比'),
)

def _scroll_form(parent):
    scroll = QScrollArea(parent)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    form = QWidget()
    layout = QVBoxLayout(form)
    scroll.setWidget(form)
    return scroll, form, layout


class ReportConfigDialog(QDialog):
    def __init__(self, lines, artifacts, spatial_results, parent=None):
        super().__init__(parent)
        self.setWindowTitle('生成报告包')
        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)
        layout.addWidget(tabs, 1)
        self._scope_tab, form, layout = _scroll_form(tabs)
        tabs.addTab(self._scope_tab, '交付范围')
        self._line_records = {str(item.line_id): item for item in lines or ()}
        self._artifact_records = {str(item.artifact_id): item for item in artifacts or ()}
        self._line_content = {}
        self._main_artifacts = {}
        self._editing_line = ''
        self._build_scope_tree(form, layout)
        self._scope_error = make_hint('', parent=form, key='error')
        self._scope_error.hide()
        layout.addWidget(self._scope_error)
        self._spatial_results = list(spatial_results or [])
        self._spatial_combo = ComboBox(self)
        self._spatial_row = QWidget(self)
        self._spatial_row.setLayout(make_form_row('空间成果版本', self._spatial_combo, parent=self._spatial_row))
        layout.addWidget(self._spatial_row)
        chapters, chapter_layout = make_collapsible_card('报告章节', parent=form, collapsed=True)
        self._section_checks = {}
        sections_row = QGridLayout()
        for index, (key, label) in enumerate(_SECTIONS):
            check = CheckBox(label, chapters)
            check.setChecked(True)
            self._section_checks[key] = check
            sections_row.addWidget(check, index // 3, index % 3)
        chapter_layout.addLayout(sections_row)
        self._borehole_threshold = DoubleSpinBox(self)
        self._borehole_threshold.setRange(0.0, 10000.0)
        self._borehole_threshold.setDecimals(3)
        self._borehole_threshold.setValue(1.0)
        self._borehole_threshold.setSuffix(' m')
        threshold_row = QWidget(self)
        threshold_row.setLayout(make_form_row('钻孔误差阈值', self._borehole_threshold, parent=threshold_row))
        chapter_layout.addWidget(threshold_row)
        layout.addWidget(chapters)
        self._section_checks['include_borehole_comparison'].toggled.connect(threshold_row.setVisible)
        self._section_checks['include_spatial_results'].toggled.connect(self._sync_scope)

        options, fields = make_collapsible_card('报告信息', parent=self, collapsed=True)
        self._package_name = LineEdit(options)
        self._package_name.setPlaceholderText('留空自动命名')
        self._title = LineEdit(options)
        self._compiler = LineEdit(options)
        self._reviewer = LineEdit(options)
        self._approver = LineEdit(options)
        for label, widget in (('报告包名', self._package_name), ('报告标题', self._title),
                              ('编制', self._compiler), ('审核', self._reviewer), ('批准', self._approver)):
            fields.addLayout(make_form_row(label, widget, parent=options))
        layout.addWidget(options)
        layout.addStretch()
        self._build_report_content(tabs)
        buttons = QHBoxLayout()
        self._scope_summary = make_hint('', parent=self)
        buttons.addWidget(self._scope_summary)
        buttons.addStretch(1)
        cancel = PushButton('取消', self)
        self._confirm = PrimaryPushButton('生成报告包', self)
        cancel.clicked.connect(self.reject)
        self._confirm.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(self._confirm)
        self.layout().addLayout(buttons)
        self._scope_tree.itemChanged.connect(self._scope_item_changed)
        self._sync_scope()
        available = self.screen().availableGeometry()
        self.resize(min(665, available.width() - 24), min(530, available.height() - 48))

    def _build_scope_tree(self, parent, layout):
        tree = self._scope_tree = TreeWidget(parent)
        tree.setObjectName('reportScopeTree')
        row_style = 'QTreeWidget#reportScopeTree::item { min-height: 38px; }'
        setCustomStyleSheet(tree, row_style, row_style)
        tree.setHeaderLabels(['测线与处理成果', '报告采用的处理成果'])
        tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tree.setAlternatingRowColors(True)
        self._line_items = {}
        self._artifact_items = {}
        self._line_artifact_ids = {key: [] for key in self._line_records}
        self._primary_selectors = {}
        for key, line in self._line_records.items():
            item = QTreeWidgetItem(tree, [f'{key} · {line.name}'])
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            item.setCheckState(0, Qt.CheckState.Checked)
            self._line_items[key] = item
            selector = ComboBox(tree)
            selector.setMinimumWidth(0)
            selector.setFixedHeight(32)
            selector.setAccessibleName(f'{key} 报告采用的处理成果')
            selector.currentIndexChanged.connect(lambda _, key=key: self._remember_main_artifact(key))
            self._primary_selectors[key] = selector
            item.setSizeHint(0, QSize(0, 38))
            item.setSizeHint(1, QSize(0, 38))
            tree.setItemWidget(item, 1, selector)
        for key, artifact in self._artifact_records.items():
            parent_item = self._line_items.get(str(artifact.line_id))
            if parent_item is None:
                continue
            item = QTreeWidgetItem(parent_item, [str(artifact.name or key)])
            item.setToolTip(0, key)
            item.setCheckState(0, Qt.CheckState.Checked)
            self._artifact_items[key] = item
            self._line_artifact_ids[str(artifact.line_id)].append(key)
        tree.expandAll()
        tree.itemExpanded.connect(self._fit_scope_tree)
        tree.itemCollapsed.connect(self._fit_scope_tree)
        layout.addWidget(tree)

    def _fit_scope_tree(self):
        height = self._scope_tree.header().sizeHint().height() + 8
        for item in self._line_items.values():
            index = self._scope_tree.indexFromItem(item)
            height += self._scope_tree.sizeHintForIndex(index).height()
            if item.isExpanded() and item.checkState(0) == Qt.CheckState.Checked:
                for row in range(item.childCount()):
                    index = self._scope_tree.indexFromItem(item.child(row))
                    height += self._scope_tree.sizeHintForIndex(index).height()
        self._scope_tree.setFixedHeight(min(292, height))

    def _scope_item_changed(self, item, column):
        if column != 0:
            return
        self._scope_error.hide()
        if item.parent() is None:
            self._sync_scope()
        else:
            self._update_main_artifact(item.parent().data(0, Qt.ItemDataRole.UserRole))
            self._update_count()

    def _build_report_content(self, tabs):
        project_tab, project, project_layout = _scroll_form(tabs)
        tabs.addTab(project_tab, '摘要与结论')
        self._project_text = {}
        for key, label in (('objective', '测量目的'), ('findings', '主要发现'), ('conclusion', '结论与验证')):
            editor = PlainTextEdit(project)
            editor.setAccessibleName(label)
            project_layout.addWidget(card_title(label))
            project_layout.addWidget(editor, 1)
            self._project_text[key] = editor

        line_page = QWidget(tabs)
        tabs.addTab(line_page, '逐线内容')
        layout = QVBoxLayout(line_page)
        self._content_line = ComboBox(line_page)
        layout.addLayout(make_form_row('测线', self._content_line, parent=line_page))
        self._featured = CheckBox('列入重点测线', line_page)
        layout.addWidget(self._featured)
        details = QTabWidget(line_page)
        layout.addWidget(details, 1)
        acquisition_tab, acquisition, fields = _scroll_form(details)
        details.addTab(acquisition_tab, '采集补录')
        self._manual_fields = {}
        for key, label in MANUAL_ACQUISITION_FIELDS.items():
            editor = LineEdit(acquisition)
            editor.setAccessibleName(label)
            fields.addLayout(make_form_row(label, editor, parent=acquisition, trailing_stretch=False))
            self._manual_fields[key] = editor
        self._manual_fields['power'].setPlaceholderText('例如 5 dBm（设备设定）')
        fields.addStretch()

        analysis_tab, analysis, fields = _scroll_form(details)
        details.addTab(analysis_tab, '分析')
        self._first_trace = QSpinBox(analysis)
        self._last_trace = QSpinBox(analysis)
        for editor, label in ((self._first_trace, '起始道'), (self._last_trace, '结束道')):
            editor.setSpecialValueText('整条测线')
            editor.setAccessibleName(label)
            fields.addLayout(make_form_row(label, editor, parent=analysis))
        self._analysis_fields = {}
        for key, label in ANALYSIS_FIELDS.items():
            editor = PlainTextEdit(analysis)
            editor.setAccessibleName(label)
            fields.addWidget(card_title(label))
            fields.addWidget(editor, 1)
            self._analysis_fields[key] = editor
        self._content_line.currentIndexChanged.connect(self._change_content_line)
        self._content_status = make_hint('', parent=line_page, key='error')
        self._content_status.hide()
        layout.addWidget(self._content_status)
        self._content_tab = line_page
        self._content_tabs = tabs

    def _save_line_content(self):
        if self._editing_line:
            self._line_content[self._editing_line] = {
                'acquisition': {key: field.text().strip() for key, field in self._manual_fields.items()},
                'analysis': {**{key: field.toPlainText().strip() for key, field in self._analysis_fields.items()},
                             'first_trace': self._first_trace.value(), 'last_trace': self._last_trace.value()},
                'featured': self._featured.isChecked(),
            }

    def _change_content_line(self):
        self._save_line_content()
        self._editing_line = str(self._content_line.currentData() or '')
        saved = self._line_content.get(self._editing_line, {})
        for key, field in self._manual_fields.items():
            field.setText(saved.get('acquisition', {}).get(key, ''))
        analysis = saved.get('analysis', {})
        for key, field in self._analysis_fields.items():
            field.setPlainText(analysis.get(key, ''))
        count = int(getattr(self._line_records.get(self._editing_line), 'trace_count', 0) or 0)
        for key, field in (('first_trace', self._first_trace), ('last_trace', self._last_trace)):
            field.setRange(0, count)
            field.setValue(analysis.get(key, 0))
        self._featured.setChecked(bool(saved.get('featured')))
        self._content_status.hide()

    def _sync_content_lines(self, line_ids):
        current = self._editing_line
        with QSignalBlocker(self._content_line):
            self._content_line.clear()
            for key, line in self._line_records.items():
                if key in line_ids:
                    self._content_line.addItem(f'{key} · {line.name}', userData=key)
            if current in line_ids:
                self._content_line.setCurrentIndex(self._content_line.findData(current))
        self._content_tab.setEnabled(bool(line_ids))
        self._change_content_line()

    def _update_main_artifact(self, line):
        parent = self._line_items[line]
        selector = self._primary_selectors[line]
        included = parent.checkState(0) == Qt.CheckState.Checked
        candidates = [self._artifact_records[key] for key in self._line_artifact_ids[line]
                      if included and self._artifact_items[key].checkState(0) == Qt.CheckState.Checked]
        candidates = [item for item in candidates
                      if getattr(item, 'status', 'success') in ('success', 'committed')
                      and (getattr(item, 'role', '') or (getattr(item, 'manifest', {}) or {}).get('artifact_role'))
                      != 'display_compare_transform']
        candidates.sort(key=lambda item: (str(getattr(item, 'created_at', '')), item.artifact_id), reverse=True)
        names = Counter(item.name for item in candidates)
        with QSignalBlocker(selector):
            selector.clear()
            selector.setEnabled(included)
            selector.setVisible(included)
            for item in candidates:
                name = item.name or item.artifact_id
                if names[item.name] > 1:
                    name = f'{name} · {item.artifact_id}'
                selector.addItem(name, userData=item.artifact_id)
            selector.addItem('仅原始剖面', userData='')
            if line in self._main_artifacts:
                index = selector.findData(self._main_artifacts[line])
                if index < 0:
                    selector.addItem('请选择处理成果', userData=None)
                    index = selector.count() - 1
                selector.setCurrentIndex(index)
        self._set_artifact_tooltip(line)

    def _remember_main_artifact(self, line):
        value = self._primary_selectors[line].currentData()
        if value is not None:
            self._main_artifacts[line] = str(value)
            self._scope_error.hide()
        self._set_artifact_tooltip(line)

    def _set_artifact_tooltip(self, line):
        selector = self._primary_selectors[line]
        selector.setToolTip(str(selector.currentData() or ''))

    def _checked_lines(self):
        return [key for key, item in self._line_items.items()
                if item.checkState(0) == Qt.CheckState.Checked]

    def _sync_scope(self):
        line_ids = set(self._checked_lines())
        with QSignalBlocker(self._scope_tree):
            for key, item in self._line_items.items():
                for index in range(item.childCount()):
                    item.child(index).setHidden(key not in line_ids)
                self._update_main_artifact(key)
        self._fit_scope_tree()

        current = self._spatial_combo.currentData()
        self._spatial_combo.clear()
        for result in self._spatial_results:
            if set(getattr(result, 'line_ids', ())) <= line_ids:
                self._spatial_combo.addItem(
                    f'{result.name} v{result.revision}', userData=str(result.result_id))
        if current:
            index = self._spatial_combo.findData(current)
            if index >= 0:
                self._spatial_combo.setCurrentIndex(index)
        self._spatial_row.setVisible(self._section_checks['include_spatial_results'].isChecked()
                                     and self._spatial_combo.count() > 0)
        self._sync_content_lines(line_ids)
        self._update_count()

    def _checked_artifacts(self):
        return [key for key, item in self._artifact_items.items()
                if item.parent().checkState(0) == Qt.CheckState.Checked
                and item.checkState(0) == Qt.CheckState.Checked]

    def _update_count(self):
        lines = len(self._checked_lines())
        artifacts = len(self._checked_artifacts())
        self._scope_summary.setText(f'{lines} 条测线 · {artifacts} 项处理成果'
                                    if lines else '未选择测线')
        self._confirm.setEnabled(lines > 0)

    def accept(self):
        self._save_line_content()
        artifacts = set(self._checked_artifacts())
        for key in self._checked_lines():
            selected = self._main_artifacts.get(key)
            if selected and (selected not in artifacts or self._primary_selectors[key].currentData() is None):
                self._content_tabs.setCurrentWidget(self._scope_tab)
                self._scope_error.setText(f'{key} 的处理成果已取消勾选，请重新选择')
                self._scope_error.show()
                self._scope_tree.scrollToItem(self._line_items[key])
                self._primary_selectors[key].setFocus()
                return
            analysis = self._line_content.get(key, {}).get('analysis', {})
            first, last = analysis.get('first_trace', 0), analysis.get('last_trace', 0)
            if (first or last) and not 1 <= first <= last:
                self._content_tabs.setCurrentWidget(self._content_tab)
                self._content_line.setCurrentIndex(self._content_line.findData(key))
                self._content_status.setText(f'{key} 的分析需要有效的起止道。')
                self._content_status.show()
                return
        if self._checked_lines():
            super().accept()

    def payload(self):
        self._save_line_content()
        selected_lines = self._checked_lines()
        artifacts = self._checked_artifacts()
        profile = {
            'title': self._title.text().strip(),
            'compiler': self._compiler.text().strip(),
            'reviewer': self._reviewer.text().strip(),
            'approver': self._approver.text().strip(),
            'selected_line_ids': selected_lines,
            'selected_artifact_ids': artifacts,
            'primary_artifacts': {key: item for key, item in self._main_artifacts.items()
                                  if key in selected_lines},
            'project_analysis': {key: editor.toPlainText().strip() for key, editor in self._project_text.items()},
            'line_analysis': {key: self._line_content[key]['analysis'] for key in selected_lines if key in self._line_content},
            'acquisition_overrides': {key: self._line_content[key]['acquisition'] for key in selected_lines if key in self._line_content},
            'featured_line_ids': [key for key in selected_lines if self._line_content.get(key, {}).get('featured')],
            'spatial_result_id': str(self._spatial_combo.currentData() or ''),
            'borehole_error_threshold_m': float(self._borehole_threshold.value()),
            'include_integrity_manifest': True,
        }
        profile.update({key: check.isChecked() for key, check in self._section_checks.items()})
        return {
            'package_name': self._package_name.text().strip(),
            'report_profile': profile,
        }
