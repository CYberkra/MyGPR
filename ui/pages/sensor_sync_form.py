"""Sensor file selection and trace-alignment settings, without hand-written JSON."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import CheckBox, DoubleSpinBox, LineEdit, PushButton

from ui.page_scaffold import make_form_row, make_hint
from ui.widgets.collapsible_card import make_collapsible_card


_SENSOR_ROWS = (
    ('rtk', 'RTK 文件:'),
    ('imu', 'IMU 文件:'),
    ('altimeter', '高度计文件:'),
    ('trace_timestamps', '道时间戳文件:'),
)

_FIELDS = (
    ('rtk_time_offset_s', 'RTK 时间偏移/s', 0.0, 6),
    ('imu_time_offset_s', 'IMU 时间偏移/s', 0.0, 6),
    ('altimeter_time_offset_s', '高度计时间偏移/s', 0.0, 6),
    ('maximum_nearest_residual_s', '最大匹配残差/s', 0.25, 3),
    ('gap_warning_s', '缺段警告阈值/s', 1.0, 3),
    ('lever_arm_x_m', '杆臂 X/m', 0.0, 3),
    ('lever_arm_y_m', '杆臂 Y/m', 0.0, 3),
    ('lever_arm_z_m', '杆臂 Z/m', 0.0, 3),
    ('radar_trigger_delay_s', '雷达触发延迟/s', 0.0, 6),
    ('antenna_height_offset_m', '天线高度偏移/m', 0.0, 3),
)


class SensorSyncForm(QWidget):
    browse_requested = pyqtSignal(str)
    save_template_requested = pyqtSignal()
    apply_template_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edits = {}
        for key, label in _SENSOR_ROWS:
            edit = LineEdit(self)
            edit.setPlaceholderText('可选' if key != 'rtk' else '选择 RTK 文件（必填）…')
            browse = PushButton('浏览', self)
            browse.setFixedWidth(70)
            browse.clicked.connect(lambda _=False, name=key: self.browse_requested.emit(name))
            layout.addLayout(make_form_row(label, edit, browse, parent=self))
            edit.textChanged.connect(edit.setToolTip)
            self.edits[key] = edit

        self._fields = {}
        advanced_card, advanced = make_collapsible_card('高级同步与标定', collapsed=True)
        for key, label, default, decimals in _FIELDS:
            spin = DoubleSpinBox(advanced_card)
            spin.setRange(-86400.0 if 'offset' in key or 'delay' in key else 0.0, 86400.0)
            spin.setDecimals(decimals)
            spin.setSingleStep(0.01 if decimals >= 3 else 0.1)
            spin.setValue(default)
            advanced.addLayout(make_form_row(label, spin, parent=advanced_card))
            self._fields[key] = spin
        extrapolation = CheckBox('允许时间范围外插（谨慎）', advanced_card)
        advanced.addWidget(extrapolation)
        self._fields['allow_extrapolation'] = extrapolation
        crs = LineEdit(advanced_card)
        crs.setPlaceholderText('留空沿用项目坐标系，例如 EPSG:4547')
        advanced.addLayout(make_form_row('项目坐标系', crs, parent=advanced_card))
        self._fields['project_crs'] = crs
        layout.addWidget(advanced_card)
        hint = make_hint('模板仅保存同步与标定参数，不包含各测线的设备文件路径。', parent=self)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        templates = QHBoxLayout()
        save = PushButton('保存为项目模板', self)
        apply = PushButton('应用项目模板', self)
        save.clicked.connect(self.save_template_requested)
        apply.clicked.connect(self.apply_template_requested)
        templates.addWidget(save)
        templates.addWidget(apply)
        templates.addStretch(1)
        layout.addLayout(templates)

    def settings(self):
        values = {}
        for key, widget in self._fields.items():
            if isinstance(widget, CheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, LineEdit):
                values[key] = widget.text().strip()
            else:
                values[key] = float(widget.value())
        return values

    def set_settings(self, values):
        values = dict(values or {})
        for key, widget in self._fields.items():
            if key not in values:
                continue
            widget.blockSignals(True)
            if isinstance(widget, CheckBox):
                widget.setChecked(bool(values[key]))
            elif isinstance(widget, LineEdit):
                widget.setText(str(values[key] or ''))
            else:
                widget.setValue(float(values[key]))
            widget.blockSignals(False)

    def clear_paths(self):
        for edit in self.edits.values():
            edit.clear()
