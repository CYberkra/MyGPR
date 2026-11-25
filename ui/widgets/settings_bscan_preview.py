"""Compare display settings on a bundled slice of measured radar data."""

from importlib.resources import files
import logging
from zipfile import BadZipFile

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ui.constants import BSCAN_AGC_KEYS, BSCAN_GAIN_KEYS, BSCAN_TVG_NODES
from ui.desktop_backend_facade import manual_gain_preview_bundle
from ui.design_tokens import tokens
from ui.settings_manager import DEFAULT_SETTINGS
from ui.widgets.bscan_view import BScanView


_LEVEL_KEYS = ('bscan_p_low', 'bscan_p_high')
_DISPLAY_KEYS = ('bscan_colormap', 'bscan_contrast', 'bscan_colorbar_visible', *_LEVEL_KEYS, *BSCAN_GAIN_KEYS)
_BASELINE = {'bscan_gain_mode': 'off', 'bscan_contrast': 1.0, 'bscan_p_low': 2.0, 'bscan_p_high': 98.0}
_SOURCE_LABEL = '佩里托莫雷诺冰川 · 降采样'


def _field_bundle():
    """Load the offline sample by index, with no inferred physical axis."""
    resource = files('ui').joinpath('resources', 'settings_field_bscan.npz')
    with resource.open('rb') as source, np.load(source, allow_pickle=False) as archive:
        matrix = archive['matrix']
    if matrix.shape != (256, 384) or matrix.dtype != np.dtype('<f4') or not np.isfinite(matrix).all():
        raise ValueError('Invalid settings B-scan sample')
    matrix.setflags(write=False)
    return manual_gain_preview_bundle(matrix, {})


class _StaticBScanView(BScanView):
    """Retain standard rendering while keeping the settings plot presentation-only."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('settingsBscanView')
        self.setAccessibleName('实测雷达剖面')
        self.setMinimumHeight(230)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for widget in self.findChildren(QWidget):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._fullscreen_shortcut.setEnabled(False)
        self.set_levels_follow_view(False)
        self.set_title_visible(False)

    def _layout_fullscreen_controls(self):
        self._fullscreen_btn.hide()


class SettingsBScanPreview(QFrame):
    """Compare the saved appearance against a local, non-persisted baseline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('settingsBscanPreview')
        self.setAccessibleName('实测剖面 B-scan 显示预览')
        self.setMinimumWidth(270)
        self._view = None
        self._load_failed = False
        self._preferences = {key: DEFAULT_SETTINGS[key] for key in _DISPLAY_KEYS}
        self._rendered = {}
        self._redraw = QTimer(self)
        self._redraw.setSingleShot(True)
        self._redraw.setInterval(45)
        self._redraw.timeout.connect(self._refresh)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 12, 12, 9)
        self._layout.setSpacing(7)
        header = QHBoxLayout()
        title = QLabel('实测剖面', self)
        title.setObjectName('settingsBscanTitle')
        title.setToolTip('PeritoMoreno_1 · Koch 等 · CC BY 4.0')
        header.addWidget(title)
        header.addStretch(1)
        self._current = QPushButton('当前显示', self)
        self._current.setObjectName('settingsBscanCurrent')
        self._baseline = QPushButton('基准', self)
        self._baseline.setObjectName('settingsBscanBaseline')
        self._baseline.setToolTip('关闭增益，使用默认对比度和百分位。保留所选色图和色标。')
        self._modes = QButtonGroup(self)
        for button in (self._current, self._baseline):
            button.setCheckable(True)
            self._modes.addButton(button)
            header.addWidget(button)
        self._current.setChecked(True)
        self._modes.buttonClicked.connect(self._compare)
        self._layout.addLayout(header)

        self._caption = QLabel(self)
        self._caption.setObjectName('settingsBscanCaption')
        self._caption.setWordWrap(True)
        self._layout.addWidget(self._caption)
        self._update_caption()
        self.apply_theme(False)

    def apply_theme(self, dark: bool) -> None:
        self._dark = dark
        palette = tokens(dark)
        self.setStyleSheet(f'''
            QFrame#settingsBscanPreview {{
                background: {palette['plot_background']};
                border: 1px solid {palette['border_default']}; border-radius: 7px;
            }}
            QLabel#settingsBscanTitle {{ color: {palette['text_primary']}; font-weight: 600; }}
            QLabel#settingsBscanCaption {{ color: {palette['plot_foreground']}; padding: 2px; }}
            QPushButton#settingsBscanCurrent, QPushButton#settingsBscanBaseline {{
                background: {palette['surface_card']}; color: {palette['text_primary']};
                border: 1px solid {palette['border_default']};
                border-radius: 4px; padding: 3px 8px; min-height: 20px;
            }}
            QPushButton#settingsBscanCurrent:checked, QPushButton#settingsBscanBaseline:checked {{
                background: {palette['primary_subtle']}; color: {palette['primary_active']};
            }}
            QPushButton#settingsBscanCurrent:hover, QPushButton#settingsBscanBaseline:hover {{
                border-color: {palette['primary_border']};
            }}
            QPushButton#settingsBscanCurrent:focus, QPushButton#settingsBscanBaseline:focus {{
                border-color: {palette['border_focus']};
            }}
        ''')

    def set_preferences(self, values: dict) -> None:
        """Receive committed appearance settings without emitting any write-back."""
        previous = self._display_values(baseline=False)
        self._preferences.update({key: values[key] for key in _DISPLAY_KEYS if key in values})
        if self._display_values(baseline=False) == previous:
            return
        self._current.setChecked(True)
        self._update_caption()
        if self.isVisible() and not self._redraw.isActive():
            self._redraw.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._view is None:
            try:
                bundle = _field_bundle()
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                logging.getLogger(__name__).warning('无法加载设置页实测预览', exc_info=True)
                self._load_failed = True
                self._current.setEnabled(False)
                self._baseline.setEnabled(False)
                self._update_caption()
                return
            self._view = _StaticBScanView(self)
            self._view.apply_theme(self._dark)
            self._view.set_bundle(bundle)
            self._layout.insertWidget(1, self._view, 1)
            self._load_failed = False
            self._current.setEnabled(True)
            self._baseline.setEnabled(True)
            self._update_caption()
        self._refresh()

    def _display_values(self, *, baseline: bool) -> dict:
        settings = self._preferences | _BASELINE if baseline else self._preferences.copy()
        if settings['bscan_gain_mode'] != 'sec':
            settings.pop('bscan_gain_alpha')
        if settings['bscan_gain_mode'] != 'tvg':
            settings.pop('bscan_tvg_db')
            settings.pop('bscan_tvg_power')
        if settings['bscan_gain_mode'] != 'piecewise_tvg':
            for _, key, _ in BSCAN_TVG_NODES:
                settings.pop(key)
        if settings['bscan_gain_mode'] != 'agc':
            for key in BSCAN_AGC_KEYS:
                settings.pop(key)
        return settings

    def _compare(self, _button) -> None:
        self._update_caption()
        self._redraw.stop()
        self._refresh()

    def _update_caption(self) -> None:
        if self._load_failed:
            self._caption.setText('实测预览资源不可用。')
        elif self._baseline.isChecked():
            self._caption.setText('基准关闭增益，使用对比度 1 和百分位 2–98。')
        else:
            self._caption.setText(_SOURCE_LABEL)

    def _refresh(self) -> None:
        if self._view is None or not self.isVisible():
            return
        state = self._display_values(baseline=self._baseline.isChecked())
        changed = {key for key, value in state.items() if value != self._rendered.get(key)}
        if 'bscan_colormap' in changed:
            self._view.set_colormap(state['bscan_colormap'])
        if 'bscan_colorbar_visible' in changed:
            self._view.set_colorbar_visible(state['bscan_colorbar_visible'])
        if 'bscan_contrast' in changed:
            self._view.set_contrast(state['bscan_contrast'])
        if changed.intersection(_LEVEL_KEYS):
            self._view.set_display_levels(state['bscan_p_low'], state['bscan_p_high'])
        if changed.intersection(BSCAN_GAIN_KEYS):
            segment_db = None
            if state['bscan_gain_mode'] == 'piecewise_tvg':
                segment_db = [state[key] for _, key, _ in BSCAN_TVG_NODES]
            self._view.set_gain(
                state['bscan_gain_mode'],
                alpha=state.get('bscan_gain_alpha'),
                db=state.get('bscan_tvg_db'),
                power=state.get('bscan_tvg_power'),
                segment_db=segment_db,
                agc_window=state.get('bscan_agc_window'),
                agc_max_db=state.get('bscan_agc_max_db'),
            )
        self._rendered = state
