"""Small, settings-local controls and palette for the desktop preferences page."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ui.design_tokens import tokens


class SettingsChoice(QWidget):
    """An exclusive set of visible buttons whose stored values match the preference."""

    currentIndexChanged = pyqtSignal(int)
    currentTextChanged = pyqtSignal(str)

    def __init__(self, choices: tuple[tuple[str, str, str], ...], parent=None):
        super().__init__(parent)
        self._choices = choices
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        for index, (label, _value, name) in enumerate(choices):
            button = QRadioButton(label, self)
            button.setObjectName(name)
            button.setAccessibleName(label)
            button.setProperty('settingsRole', 'segment')
            self._group.addButton(button, index)
            layout.addWidget(button)
        self._group.button(0).setChecked(True)
        self.setFocusProxy(self._group.button(0))
        self._group.idToggled.connect(self._selection_changed)

    def _selection_changed(self, index: int, checked: bool) -> None:
        if checked:
            self.setFocusProxy(self._group.button(index))
            self.currentIndexChanged.emit(index)
            self.currentTextChanged.emit(self.currentText())

    def currentData(self) -> str:
        return self._choices[self._group.checkedId()][1]

    def currentText(self) -> str:
        return self.currentData()

    def findData(self, value: str) -> int:
        return next((i for i, option in enumerate(self._choices) if option[1] == value), -1)

    def setCurrentIndex(self, index: int) -> None:
        button = self._group.button(index)
        if button is not None:
            button.setChecked(True)

    def setCurrentText(self, value: str) -> None:
        self.setCurrentIndex(self.findData(value))


class SettingsDisclosure(QWidget):
    """A keyboard-accessible expandable row with a visible current value."""

    def __init__(self, label: str, body: QWidget, parent=None):
        super().__init__(parent)
        self._label = label
        self.body = body
        self.toggle = QPushButton(self)
        self.toggle.setProperty('settingsRole', 'disclosure')
        self.toggle.setCheckable(True)
        self.toggle.setAccessibleName(label)
        self.summary = QLabel(self)
        self.summary.setProperty('settingsRole', 'hint')
        self.summary.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.summary.setContentsMargins(0, 0, 8, 0)

        header = QWidget(self)
        header.setProperty('settingsRole', 'disclosureHeader')
        row = QGridLayout(header)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.toggle, 0, 0)
        row.addWidget(self.summary, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(body)
        self.toggle.toggled.connect(self._show_body)
        self._show_body(False)

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setChecked(expanded)

    def _show_body(self, expanded: bool) -> None:
        self.toggle.setText(('⌄  ' if expanded else '›  ') + self._label)
        self.body.setVisible(expanded)


def settings_stylesheet(dark: bool) -> str:
    theme = tokens(dark)
    paper, rail = theme['bg_surface'], theme['bg_subtle']
    text, muted = theme['text_primary'], theme['text_secondary']
    edge, focus = theme['border_default'], theme['border_focus']
    accent, selected = theme['primary'], theme['primary_subtle']
    selected_text = theme['primary_active'] if dark else theme['primary_hover']
    hover, error = theme['bg_muted'], theme['error']
    return f"""
        QWidget#settingsSurface, QWidget#settingsPanel {{ background: {paper}; }}
        QWidget#settingsSidebar {{ background: {rail}; border-right: 1px solid {edge}; }}
        QLabel[settingsRole="title"] {{ color: {text}; font-size: 24px; font-weight: 700; }}
        QLabel[settingsRole="label"] {{ color: {text}; font-weight: 600; }}
        QLabel[settingsRole="hint"] {{ color: {muted}; }}
        QLabel[settingsRole="error"] {{ color: {error}; }}
        QLineEdit#settingsSearch {{
            color: {text}; background: {rail}; border: 1px solid {edge};
            border-radius: 5px; padding: 5px 9px; min-height: 21px;
        }}
        QLineEdit#settingsSearch:focus {{ border-color: {focus}; }}
        QListView#settingsSearchResults {{
            color: {text}; background: {paper}; border: 1px solid {edge};
            selection-color: {text}; selection-background-color: {selected};
            outline: none;
        }}
        QListView#settingsSearchResults::item {{ min-height: 27px; padding: 2px 8px; }}
        QSlider[settingsRole="displaySlider"]::groove:horizontal {{
            height: 4px; background: {edge}; border-radius: 2px;
        }}
        QSlider[settingsRole="displaySlider"]::sub-page:horizontal {{ background: {accent}; }}
        QSlider[settingsRole="displaySlider"]::handle:horizontal {{
            background: {accent}; border: 2px solid {paper}; border-radius: 7px;
            width: 14px; margin: -5px 0;
        }}
        QSlider[settingsRole="displaySlider"]:focus::handle:horizontal {{ border-color: {text}; }}
        QWidget[settingsRole="row"], QWidget[settingsRole="disclosureHeader"] {{
            border-bottom: 1px solid {edge};
        }}
        QPushButton[settingsRole="nav"], QPushButton[settingsRole="disclosure"],
        QRadioButton[settingsRole="segment"] {{
            color: {text}; background: transparent; border: 1px solid transparent;
            border-radius: 4px; padding: 5px 10px; min-height: 22px;
        }}
        QPushButton[settingsRole="nav"], QPushButton[settingsRole="disclosure"] {{
            text-align: left; font-weight: 600;
        }}
        QRadioButton[settingsRole="segment"] {{ border: 1px solid {edge}; spacing: 0; }}
        QRadioButton[settingsRole="segment"]::indicator {{ width: 0; height: 0; }}
        QPushButton[settingsRole="nav"]:hover, QPushButton[settingsRole="disclosure"]:hover,
        QRadioButton[settingsRole="segment"]:hover {{ background: {hover}; }}
        QPushButton[settingsRole="nav"]:checked, QRadioButton[settingsRole="segment"]:checked {{
            color: {selected_text}; background: {selected}; border-color: {edge}; font-weight: 700;
        }}
        QPushButton[settingsRole="nav"]:checked {{ border-left: 3px solid {accent}; }}
        QPushButton[settingsRole]:focus, QRadioButton[settingsRole="segment"]:focus,
        QWidget[settingsFocus="true"] {{
            border: 1px solid {focus};
        }}
    """


def settings_label(text: str, role: str, *, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setProperty('settingsRole', role)
    label.setWordWrap(wrap)
    return label
