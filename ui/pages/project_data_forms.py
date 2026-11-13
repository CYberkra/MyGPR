"""Import and sensor forms embedded in project tools."""
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CardWidget, CheckBox, DoubleSpinBox, LineEdit, PrimaryPushButton, PushButton

from ui import constants
from ui.page_scaffold import make_card, make_form_row, make_hint


class ProjectDataForms:
    def _build_import_card(self, parent) -> CardWidget:
        """Build the import fields and the preflight result area."""
        card, layout = make_card()

        # 数据文件行
        self.file_edit = LineEdit(card)
        self.file_edit.setPlaceholderText('选择 GPR 数据文件…')
        self.file_edit.setMinimumWidth(180)
        browse_btn = PushButton('浏览', card)
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_import_file)
        layout.addWidget(BodyLabel('数据文件:', card))
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)
        self.file_edit.textEdited.connect(self._on_import_text_edited)

        # 测线号 / 名称
        self.line_id_edit = self._add_labeled_edit(
            layout, card, '测线号:', default='L01', placeholder='例如 L01')
        self.line_name_edit = self._add_labeled_edit(
            layout, card, '名称:', placeholder='留空则使用测线号')

        # 介电常数
        self.dielectric_spin = DoubleSpinBox(card)
        self.dielectric_spin.setRange(1.0, 81.0)
        self.dielectric_spin.setDecimals(2)
        self.dielectric_spin.setSingleStep(0.5)
        self.dielectric_spin.setValue(constants.DEFAULT_DIELECTRIC)
        self.dielectric_spin.setMinimumWidth(120)
        layout.addLayout(make_form_row('介电常数:', self.dielectric_spin,
                                       parent=card))

        self._matrix_axes_check = CheckBox('手动填写时窗和道间距', card)
        self._matrix_axes_check.setToolTip('用于单个缺少采集参数的矩阵文件')
        layout.addWidget(self._matrix_axes_check)
        matrix_axes = QWidget(card)
        matrix_layout = QVBoxLayout(matrix_axes)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        matrix_layout.setSpacing(constants.SPACE_2)
        self._matrix_time_window = DoubleSpinBox(matrix_axes)
        self._matrix_time_window.setRange(0.000001, 1000000000.0)
        self._matrix_time_window.setDecimals(6)
        self._matrix_time_window.setValue(250.0)
        self._matrix_time_window.setToolTip('首末采样点的时间差')
        self._matrix_trace_interval = DoubleSpinBox(matrix_axes)
        self._matrix_trace_interval.setRange(0.000001, 10000000.0)
        self._matrix_trace_interval.setDecimals(6)
        self._matrix_trace_interval.setValue(1.0)
        for label, widget in (('采集时窗（ns）', self._matrix_time_window),
                              ('道间距（m）', self._matrix_trace_interval)):
            matrix_layout.addLayout(make_form_row(label, widget, parent=matrix_axes))
        matrix_axes.hide()
        self._matrix_axes_check.toggled.connect(matrix_axes.setVisible)
        layout.addWidget(matrix_axes)

        # 按钮行：预检 + 导入
        btn_row = QHBoxLayout()
        self.preflight_btn = PushButton('预检', card)
        self.import_btn = PrimaryPushButton('导入', card)
        self.preflight_btn.clicked.connect(
            lambda: self._emit_import_request(preflight=True))
        self.import_btn.clicked.connect(
            lambda: self._emit_import_request(preflight=False))
        btn_row.addStretch(1)
        btn_row.addWidget(self.preflight_btn)
        btn_row.addWidget(self.import_btn)
        layout.addLayout(btn_row)

        # 预检结果
        self._preflight_label = make_hint('', parent=card)
        self._preflight_label.setWordWrap(True)
        self._preflight_label.setMinimumHeight(34)
        layout.addWidget(self._preflight_label)
        self._preflight_label.hide()
        return card

    def _build_sync_card(self, parent) -> CardWidget:
        card, layout = make_card(parent=parent)
        from ui.pages.sensor_sync_form import SensorSyncForm
        self._sync_form = SensorSyncForm(card)
        self._sensor_edits = self._sync_form.edits
        self._sync_form.browse_requested.connect(self._browse_sensor_file)
        self._sync_form.save_template_requested.connect(self._save_sensor_template)
        self._sync_form.apply_template_requested.connect(self._apply_sensor_template)
        layout.addWidget(self._sync_form)
        btn_row = QHBoxLayout()
        self.sync_btn = PrimaryPushButton('开始同步', card)
        self.sync_btn.clicked.connect(self._emit_sync_request)
        btn_row.addStretch(1)
        btn_row.addWidget(self.sync_btn)
        layout.addLayout(btn_row)
        layout.addStretch(1)
        return card

    def _add_labeled_edit(self, layout, card, label_text, *, default='',
                          placeholder='') -> LineEdit:
        """标签+输入框表单行（收敛实现见 ui.page_scaffold.make_form_row）。"""
        edit = LineEdit(card)
        edit.setText(default)
        if placeholder:
            edit.setPlaceholderText(placeholder)
        edit.setMinimumWidth(180)
        layout.addLayout(make_form_row(label_text, edit, parent=card,
                                       trailing_stretch=False))
        return edit
