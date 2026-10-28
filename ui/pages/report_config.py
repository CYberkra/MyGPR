"""Formal report scope, section and approval configuration."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QListWidget, QListWidgetItem, QVBoxLayout
from qfluentwidgets import CheckBox, ComboBox, DoubleSpinBox, LineEdit, PrimaryPushButton, PushButton

from ui.page_scaffold import card_title, make_form_row, make_hint


_SECTIONS = (
    ('include_project_summary', '项目概况'),
    ('include_line_inventory', '测线台账'),
    ('include_processing_history', '处理历史'),
    ('include_bscan_images', 'B-scan 图件'),
    ('include_interpretation', '解释标注'),
    ('include_spatial_results', '空间成果'),
    ('include_borehole_comparison', '钻孔对比'),
)


def _check_list(parent, rows, *, key_fn, text_fn):
    widget = QListWidget(parent)
    for row in rows:
        item = QListWidgetItem(text_fn(row))
        item.setData(Qt.ItemDataRole.UserRole, key_fn(row))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        widget.addItem(item)
    return widget


def _checked(widget):
    return [
        widget.item(row).data(Qt.ItemDataRole.UserRole)
        for row in range(widget.count())
        if widget.item(row).checkState() == Qt.CheckState.Checked
    ]


class ReportConfigDialog(QDialog):
    def __init__(self, lines, artifacts, spatial_results, parent=None):
        super().__init__(parent)
        self.setWindowTitle('配置工程报告')
        self.setMinimumSize(720, 760)
        layout = QVBoxLayout(self)
        title = card_title('报告范围与签署信息')
        title.setParent(self)
        layout.addWidget(title)
        self._package_name = LineEdit(self)
        self._package_name.setPlaceholderText('留空自动命名')
        layout.addLayout(make_form_row('报告包名', self._package_name, parent=self))
        self._title = LineEdit(self)
        layout.addLayout(make_form_row('报告标题', self._title, parent=self))
        self._compiler = LineEdit(self)
        self._reviewer = LineEdit(self)
        self._approver = LineEdit(self)
        for label, widget in (('编制', self._compiler), ('审核', self._reviewer), ('批准', self._approver)):
            layout.addLayout(make_form_row(label, widget, parent=self))

        layout.addWidget(card_title('测线与处理成果'))
        self._lines_list = _check_list(
            self, list(lines or []),
            key_fn=lambda item: str(getattr(item, 'line_id', '')),
            text_fn=lambda item: f"{getattr(item, 'line_id', '')} · {getattr(item, 'name', '')}",
        )
        self._lines_list.setMaximumHeight(130)
        layout.addWidget(self._lines_list)
        self._artifacts_list = _check_list(
            self, list(artifacts or []),
            key_fn=lambda item: str(getattr(item, 'artifact_id', '')),
            text_fn=lambda item: f"{getattr(item, 'line_id', '')} · {getattr(item, 'name', '')}",
        )
        self._artifacts_list.setMaximumHeight(130)
        layout.addWidget(self._artifacts_list)

        self._spatial_combo = ComboBox(self)
        layout.addLayout(make_form_row('空间成果版本', self._spatial_combo, parent=self))
        for result in spatial_results or []:
            self._spatial_combo.addItem(
                f"{getattr(result, 'name', '')} v{getattr(result, 'revision', 0)}",
                userData=str(getattr(result, 'result_id', '')),
            )
        self._section_checks = {}
        sections_row = QHBoxLayout()
        for key, label in _SECTIONS:
            check = CheckBox(label, self)
            check.setChecked(True)
            self._section_checks[key] = check
            sections_row.addWidget(check)
        layout.addLayout(sections_row)
        integrity = CheckBox('完整性清单（强制）', self)
        integrity.setChecked(True)
        integrity.setEnabled(False)
        self._section_checks['include_integrity_manifest'] = integrity
        layout.addWidget(integrity)
        self._borehole_threshold = DoubleSpinBox(self)
        self._borehole_threshold.setRange(0.0, 10000.0)
        self._borehole_threshold.setDecimals(3)
        self._borehole_threshold.setValue(1.0)
        self._borehole_threshold.setSuffix(' m')
        layout.addLayout(make_form_row('钻孔误差阈值', self._borehole_threshold, parent=self))
        hint = make_hint('报告只包含勾选范围；生成时会冻结来源身份与当前项目修订。', parent=self)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = PushButton('取消', self)
        confirm = PrimaryPushButton('生成报告包', self)
        cancel.clicked.connect(self.reject)
        confirm.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)

    def payload(self):
        profile = {
            'title': self._title.text().strip(),
            'compiler': self._compiler.text().strip(),
            'reviewer': self._reviewer.text().strip(),
            'approver': self._approver.text().strip(),
            'selected_line_ids': _checked(self._lines_list),
            'selected_artifact_ids': _checked(self._artifacts_list),
            'spatial_result_id': str(self._spatial_combo.currentData() or ''),
            'borehole_error_threshold_m': float(self._borehole_threshold.value()),
        }
        profile.update({key: check.isChecked() for key, check in self._section_checks.items()})
        return {
            'package_name': self._package_name.text().strip(),
            'report_profile': profile,
        }
