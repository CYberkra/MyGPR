"""ParamForm — schema 驱动的参数表单（SPEC §5.3）。

schema: [{name, label, type('int'|'float'), default, min, max}]
- int → SpinBox；float → DoubleSpinBox（6 位小数，步长合理）
- 行布局：CaptionLabel(min 100px) + 控件(stretch)
- >4 个参数时，超出部分折叠进"高级参数"区（默认收起，PushButton 切换显隐）
- set_values 忽略未知键，blockSignals 防循环
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, DoubleSpinBox, PushButton, SpinBox

_BASIC_LIMIT = 4


class ParamForm(QWidget):
    """处理方法参数表单。任意参数值变化时发射 sig_changed。"""

    sig_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._schema = []
        self._editors = {}          # name -> SpinBox | DoubleSpinBox
        self._advanced_widget = None
        self._advanced_btn = None
        self._advanced_visible = False

    # ------------------------------------------------------------- schema
    def set_schema(self, schema) -> None:
        """重建表单。schema 项: {name,label,type,default,min,max}。"""
        self.clear()
        self._schema = [dict(item) for item in (schema or [])]

        basic = self._schema[:_BASIC_LIMIT]
        advanced = self._schema[_BASIC_LIMIT:]
        for item in basic:
            self._layout.addLayout(self._build_row(item))

        if advanced:
            self._advanced_btn = PushButton('高级参数 ▶', self)
            self._advanced_btn.setCheckable(False)
            self._advanced_btn.clicked.connect(self._toggle_advanced)
            self._layout.addWidget(self._advanced_btn)

            self._advanced_widget = QWidget(self)
            adv_layout = QVBoxLayout(self._advanced_widget)
            adv_layout.setContentsMargins(0, 0, 0, 0)
            adv_layout.setSpacing(6)
            for item in advanced:
                adv_layout.addLayout(self._build_row(item))
            self._advanced_widget.setVisible(False)
            self._layout.addWidget(self._advanced_widget)

        self._layout.addStretch(1)

    def _build_row(self, item):
        name = str(item.get('name', ''))
        label = str(item.get('label', name))
        ptype = item.get('type', 'float')
        pmin = item.get('min')
        pmax = item.get('max')
        default = item.get('default', 0)

        if ptype == 'int':
            editor = SpinBox(self)
            editor.setRange(int(pmin) if pmin is not None else -10 ** 9,
                            int(pmax) if pmax is not None else 10 ** 9)
            editor.setValue(int(default if default is not None else 0))
        else:
            editor = DoubleSpinBox(self)
            editor.setDecimals(6)
            editor.setRange(float(pmin) if pmin is not None else -1e12,
                            float(pmax) if pmax is not None else 1e12)
            editor.setSingleStep(self._float_step(pmin, pmax))
            editor.setValue(float(default if default is not None else 0.0))
        editor.valueChanged.connect(lambda *_: self.sig_changed.emit())

        lbl = CaptionLabel(label, self)
        lbl.setMinimumWidth(120)
        lbl.setMaximumWidth(180)
        lbl.setWordWrap(False)
        editor.setMinimumWidth(90)
        editor.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Fixed)
        row = QHBoxLayout()
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(10)
        row.addWidget(lbl)
        row.addWidget(editor, 1)
        self._editors[name] = editor
        return row

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
            self._advanced_btn.setText(
                '高级参数 ▼' if self._advanced_visible else '高级参数 ▶')

    # ------------------------------------------------------------- 值存取
    def values(self) -> dict:
        """{name: value}，未在 schema 的键不出现。"""
        return {item['name']: self._editors[item['name']].value()
                for item in self._schema
                if item.get('name') in self._editors}

    def set_values(self, values) -> None:
        """忽略未知键；blockSignals 防循环。"""
        for name, value in (values or {}).items():
            editor = self._editors.get(name)
            if editor is None:
                continue
            editor.blockSignals(True)
            try:
                editor.setValue(value)
            finally:
                editor.blockSignals(False)

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
