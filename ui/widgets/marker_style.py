"""Small, theme-aware form primitives shared by marker tools."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFormLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from ui import constants
from ui.design_tokens import color, radius, rgba


def marker_form(parent):
    form = QFormLayout(parent)
    form.setContentsMargins(0, 0, 0, 0)
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(10)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return form


def section_title(text, parent):
    label = QLabel(text, parent)
    label.setObjectName('markerSectionTitle')
    return label


class MarkerDisclosure(QWidget):
    def __init__(self, title, parent=None, *, expanded=False):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.toggle = QToolButton(self)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setObjectName('markerDisclosure')
        self.body = QWidget(self)
        self.form = marker_form(self.body)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)
        self.toggle.toggled.connect(self._expand)
        self.toggle.setChecked(expanded)
        self._expand(expanded)

    def _expand(self, expanded):
        self.body.setVisible(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)


def marker_style(dark):
    """Scoped by the receiving marker widget; use the application's existing tokens."""
    surface, text = color('bg_surface', dark), color('text_primary', dark)
    border, accent = color('border_default', dark), color('primary', dark)
    return (
        f'QWidget#markerListPanel, QDialog#markerEditor {{background:{surface}; color:{text};}}'
        'QScrollArea#markerEditorScroll {border:0; background:transparent;}'
        'QWidget#markerEditorBody {background:transparent;}'
        f'QLabel#markerSectionTitle {{font-weight:600; color:{text};}}'
        f'QLabel#markerCount {{color:{color("text_muted", dark)}; font-size:{constants.FONT_SIZE_SECONDARY}pt;}}'
        f'QLabel#markerRangePreview {{background:{rgba("primary", .12, dark)};'
        f' color:{text}; padding:8px; border-radius:{radius()}px;}}'
        f'QLabel#markerSaveError {{color:{color("error", dark)};}}'
        'QLineEdit, QComboBox, QSpinBox, QPlainTextEdit {'
        f' background:{surface}; color:{text}; border:1px solid {border};'
        f' border-radius:{radius("sm")}px; padding:5px 7px;}}'
        f'QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QPlainTextEdit:focus {{border-color:{accent};}}'
        f'QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{color:{color("text_muted", dark)};'
        f' background:{color("bg_subtle", dark)};}}'
        f'QPushButton {{min-height:20px; padding:5px 10px; border:1px solid {border};'
        f' border-radius:{radius()}px; background:{surface}; color:{text};}}'
        f'QPushButton:hover {{background:{color("bg_muted", dark)};}}'
        f'QPushButton:focus {{border:1px solid {accent};}}'
        f'QPushButton:disabled {{color:{color("text_muted", dark)};}}'
        f'QPushButton#markerSave {{background:{accent}; color:{color("text_inverse", dark)}; border-color:{accent};}}'
        f'QPushButton#markerSave:hover {{background:{color("primary_hover", dark)};}}'
        f'QPushButton#markerSave:disabled {{background:{color("bg_muted", dark)}; color:{color("text_muted", dark)};}}'
        f'QToolButton#markerDisclosure {{border:0; padding:4px 0; color:{text}; font-weight:600;}}'
        f'QToolButton:hover {{background:{color("bg_muted", dark)};}}'
        'QTableView {border:0; background:palette(base); color:palette(text);}'
        'QTableView::item {padding:4px 6px; border:0;}'
        'QHeaderView::section {border:0; border-bottom:1px solid palette(mid); padding:6px 4px;}'
    )
