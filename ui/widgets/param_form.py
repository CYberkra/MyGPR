"""ParamForm — schema 驱动的参数表单（SPEC §5.3）。

schema: [{name, label, type('int'|'float'|'str'|'bool'), default, min, max, choices}]
- int → SpinBox；float → DoubleSpinBox（6 位小数，步长合理）
- str+choices → ComboBox；str → LineEdit；bool → CheckBox
- 行布局：CaptionLabel(min 100px) + 控件(stretch)
- advanced=True 显式归入高级区；未标记时保持前 4 项为基本参数
- visible_when / enabled_when 支持 {参数: 值或值列表} 条件规则
- hidden=preserve 保留隐藏参数值（默认），hidden=omit 从 values() 省略
- set_values 忽略未知键，blockSignals 防循环
- 无法识别的参数类型降级为只读文本行，不阻断整张表单
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import (CaptionLabel, CheckBox, ComboBox, DoubleSpinBox,
                            LineEdit, PushButton, SpinBox)
from qfluentwidgets import FluentIcon as FIF

_BASIC_LIMIT = 4


def _schema_signature(schema) -> tuple:
    """schema 结构签名，包含分组、默认值和条件显隐规则。"""
    return tuple(
        (str(item.get('name', '')), str(item.get('label', '')),
         str(item.get('type', 'float')), item.get('min'), item.get('max'),
         item.get('editor'), item.get('advanced'),
         item.get('step'), item.get('tooltip'), item.get('hidden'),
         item.get('default'),
         tuple(str(c) for c in (item.get('choices') or ())),
         tuple(sorted((str(k), str(v)) for k, v in (item.get('choice_labels') or {}).items())),
         tuple(sorted((str(k), str(v)) for k, v in (item.get('visible_when') or {}).items())),
         tuple(sorted((str(k), str(v)) for k, v in (item.get('enabled_when') or {}).items())))
        for item in (schema or []))


class ParamForm(QWidget):
    """处理方法参数表单。任意参数值变化时发射 sig_changed。"""

    sig_changed = pyqtSignal()
    gain_preview_requested = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._schema = []
        self._editors = {}
        self._row_labels = {}
        self._advanced_names = set()
        self._advanced_widget = None
        self._advanced_btn = None
        self._advanced_visible = False

    # ------------------------------------------------------------- schema
    def set_schema(self, schema) -> None:
        """重建表单，保留相同 schema 下的编辑器和高级区展开状态。"""
        schema = [dict(item) for item in (schema or [])]
        if self._editors and _schema_signature(schema) == _schema_signature(self._schema):
            return
        self.clear()
        self._schema = schema

        if any('advanced' in item for item in self._schema):
            basic = [item for item in self._schema if not item.get('advanced', False)]
            advanced = [item for item in self._schema if item.get('advanced', False)]
        else:
            basic = self._schema[:_BASIC_LIMIT]
            advanced = self._schema[_BASIC_LIMIT:]
        self._advanced_names = {str(item.get('name', '')) for item in advanced}
        for item in basic:
            self._layout.addLayout(self._build_row(item))

        if advanced:
            self._advanced_btn = PushButton('高级参数', self,
                                            FIF.CHEVRON_RIGHT_MED)
            self._advanced_btn.setCheckable(False)
            self._advanced_btn.setFixedHeight(32)
            self._advanced_btn.clicked.connect(self._toggle_advanced)
            self._layout.addWidget(self._advanced_btn)

            self._advanced_widget = QWidget(self)
            adv_layout = QVBoxLayout(self._advanced_widget)
            adv_layout.setContentsMargins(0, 0, 0, 0)
            adv_layout.setSpacing(8)
            for item in advanced:
                adv_layout.addLayout(self._build_row(item))
            self._advanced_widget.setVisible(False)
            self._layout.addWidget(self._advanced_widget)

        self._layout.addStretch(1)
        self._sync_conditional_states()

    def _build_row(self, item):
        """按参数类型构建编辑行；未知类型/异常降级为只读文本行。"""
        try:
            return self._build_editor_row(item)
        except Exception:
            # 兜底：识别不了的参数类型不阻断整张表单，降级为只读提示行
            name = str(item.get('name', ''))
            label = str(item.get('label', name))
            lbl = CaptionLabel(label, self)
            hint = QLabel(f'参数类型不支持: {item.get("type", "")} ({name})', self)
            hint.setWordWrap(True)
            hint.setMinimumWidth(120)
            row = QHBoxLayout()
            row.setContentsMargins(0, 2, 0, 2)
            row.setSpacing(10)
            row.addWidget(lbl)
            row.addWidget(hint, 1)
            return row

    def _build_editor_row(self, item):
        name = str(item.get('name', ''))
        label = str(item.get('label', name))
        ptype = item.get('type', 'float')
        pmin = item.get('min')
        pmax = item.get('max')
        default = item.get('default', 0)

        if item.get('editor') == 'gain_curve':
            from ui.widgets.manual_gain_editor import GainCurveEditor
            editor = GainCurveEditor(self)
            editor.setValue(default)
            editor.previewRequested.connect(self.gain_preview_requested)
        elif ptype == 'int':
            editor = SpinBox(self)
            editor.setRange(int(pmin) if pmin is not None else -10 ** 9,
                            int(pmax) if pmax is not None else 10 ** 9)
            editor.setValue(int(default if default is not None else 0))
            editor.setSingleStep(max(1, int(item.get('step', 1))))
        elif ptype == 'str':
            choices = item.get('choices')
            if choices:
                editor = ComboBox(self)
                labels = item.get('choice_labels') or {}
                for choice in choices:
                    value = str(choice)
                    editor.addItem(str(labels.get(value, value)), userData=value)
                editor.setCurrentIndex(max(0, editor.findData(str(default))))
            else:
                editor = LineEdit(self)
                editor.setText(str(default if default is not None else ''))
        elif ptype == 'bool':
            editor = CheckBox(self)
            editor.setChecked(bool(default))
        else:
            editor = DoubleSpinBox(self)
            editor.setDecimals(6)
            editor.setRange(float(pmin) if pmin is not None else -1e12,
                            float(pmax) if pmax is not None else 1e12)
            editor.setSingleStep(float(item.get('step', self._float_step(pmin, pmax))))
            editor.setValue(float(default if default is not None else 0.0))

        self._connect_change_signal(editor)
        if item.get('tooltip'):
            editor.setToolTip(str(item['tooltip']))

        lbl = CaptionLabel(label, self)
        lbl.setMinimumWidth(120)
        lbl.setMaximumWidth(180)
        lbl.setWordWrap(True)
        # P2-3：label 悬浮显示参数范围/可选值，避免超限靠 spinbox 静默钳制
        range_tip = self._range_tooltip(item)
        if range_tip:
            lbl.setToolTip(f'{label}\n{range_tip}')
        editor.setMinimumWidth(90)
        if isinstance(editor, CheckBox):
            editor.setFixedHeight(34)
        else:
            editor.setMinimumHeight(34)
        editor.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Fixed)
        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(12)
        row.addWidget(lbl)
        row.addWidget(editor, 1)
        self._editors[name] = editor
        self._row_labels[name] = lbl
        return row

    @staticmethod
    def _range_tooltip(item: dict) -> str:
        """构造参数范围提示（数值 [min, max]、str 可选值列表）。"""
        ptype = item.get('type', 'float')
        pmin = item.get('min')
        pmax = item.get('max')
        if ptype in ('int', 'float') and (pmin is not None or pmax is not None):
            low = str(pmin) if pmin is not None else '−∞'
            high = str(pmax) if pmax is not None else '+∞'
            return f'范围: [{low}, {high}]'
        if ptype == 'str' and item.get('choices'):
            labels = item.get('choice_labels') or {}
            return '可选: ' + '、'.join(str(labels.get(str(c), c)) for c in item['choices'])
        return ''

    def _on_editor_changed(self, *_args) -> None:
        self._sync_conditional_states()
        self.sig_changed.emit()

    def _connect_change_signal(self, editor):
        """按控件类型连接变更信号（int/float=valueChanged，str=text，bool=toggled）。"""
        if isinstance(editor, CheckBox):
            editor.toggled.connect(self._on_editor_changed)
        elif isinstance(editor, ComboBox):
            editor.currentTextChanged.connect(self._on_editor_changed)
        elif isinstance(editor, LineEdit):
            editor.textChanged.connect(self._on_editor_changed)
        else:
            editor.valueChanged.connect(self._on_editor_changed)

    @staticmethod
    def _float_step(pmin, pmax):
        """步长合理：量程的 1/100，退化时 0.1。"""
        if pmin is not None and pmax is not None:
            span = abs(float(pmax) - float(pmin))
            if span > 0:
                return max(span / 100.0, 1e-6)
        return 0.1

    def _toggle_advanced(self):
        self._advanced_visible = not self._advanced_visible
        if self._advanced_widget is not None:
            self._advanced_widget.setVisible(self._advanced_visible)
        if self._advanced_btn is not None:
            self._advanced_btn.setIcon(
                FIF.CHEVRON_DOWN_MED.icon() if self._advanced_visible
                else FIF.CHEVRON_RIGHT_MED.icon())

    @staticmethod
    def _rule_matches(rule, values) -> bool:
        if not rule:
            return True
        for name, expected in rule.items():
            actual = values.get(name)
            if isinstance(expected, (list, tuple, set, frozenset)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def _sync_conditional_states(self) -> None:
        values = self.values()
        for item in self._schema:
            name = str(item.get('name', ''))
            editor = self._editors.get(name)
            if editor is None:
                continue
            visible = self._rule_matches(item.get('visible_when'), values)
            enabled = visible and self._rule_matches(item.get('enabled_when'), values)
            editor.setVisible(visible)
            editor.setEnabled(enabled)
            label = self._row_labels.get(name)
            if label is not None:
                label.setVisible(visible)
                label.setEnabled(enabled)

    # ------------------------------------------------------------- 值存取
    def values(self) -> dict:
        """返回当前参数；隐藏参数默认保留，hidden=omit 时省略。"""
        out: dict = {}
        for item in self._schema:
            name = item.get('name')
            editor = self._editors.get(name)
            if editor is None:
                continue
            if item.get('hidden') == 'omit' and editor.isHidden():
                continue
            out[name] = self._read_editor(editor)
        return out

    @staticmethod
    def _read_editor(editor):
        """按控件类型读取当前值（float=int/float、str=文本、bool=勾选）。"""
        if isinstance(editor, CheckBox):
            return editor.isChecked()
        if isinstance(editor, ComboBox):
            value = editor.currentData()
            return editor.currentText() if value is None else value
        if isinstance(editor, LineEdit):
            return editor.text()
        return editor.value()

    def set_values(self, values) -> None:
        """忽略未知键；blockSignals 防循环。"""
        for name, value in (values or {}).items():
            editor = self._editors.get(name)
            if editor is None:
                continue
            editor.blockSignals(True)
            try:
                self._set_editor(editor, value)
            finally:
                editor.blockSignals(False)
        self._sync_conditional_states()

    @staticmethod
    def _set_editor(editor, value):
        """按控件类型写回值。

        int SpinBox 传 float 会 TypeError abort（PyQt6 严格签名；2026-09-02
        实机 380 次循环崩溃即此根因），持久化/恢复的参数可能带 .0，统一钳到
        控件量程后转换。
        """
        from ui.widgets.manual_gain_editor import GainCurveEditor
        if isinstance(editor, GainCurveEditor):
            editor.setValue(value)
        elif isinstance(editor, CheckBox):
            editor.setChecked(bool(value))
        elif isinstance(editor, ComboBox):
            index = editor.findData(str(value))
            if index >= 0:
                editor.setCurrentIndex(index)
            else:
                editor.setCurrentText(str(value))
        elif isinstance(editor, LineEdit):
            editor.setText(str(value) if value is not None else '')
        elif isinstance(editor, DoubleSpinBox):
            editor.setValue(float(value))
        else:
            editor.setValue(int(value))

    def clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                self._delete_layout(item.layout())
        self._schema = []
        self._editors = {}
        self._row_labels = {}
        self._advanced_names = set()
        self._advanced_widget = None
        self._advanced_btn = None
        self._advanced_visible = False

    @staticmethod
    def _delete_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                ParamForm._delete_layout(item.layout())
