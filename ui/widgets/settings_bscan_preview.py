"""Compare display settings on a bounded preview of the active project line."""

from uuid import uuid4

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ui.constants import BSCAN_AGC_KEYS, BSCAN_GAIN_KEYS, BSCAN_TVG_NODES
from ui.design_tokens import tokens
from ui.settings_manager import DEFAULT_SETTINGS
from ui.widgets.bscan_view import BScanView


_LEVEL_KEYS = ('bscan_p_low', 'bscan_p_high')
_DISPLAY_KEYS = ('bscan_colormap', 'bscan_contrast', 'bscan_colorbar_visible', *_LEVEL_KEYS, *BSCAN_GAIN_KEYS)
_BASELINE = {'bscan_gain_mode': 'off', 'bscan_contrast': 1.0, 'bscan_p_low': 2.0, 'bscan_p_high': 98.0}


class _StaticBScanView(BScanView):
    """Retain standard rendering while keeping the settings plot presentation-only."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('settingsBscanView')
        self.setAccessibleName('当前测线雷达剖面')
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

    preview_requested = pyqtSignal(str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('settingsBscanPreview')
        self.setAccessibleName('当前测线 B-scan 显示预览')
        self.setMinimumWidth(270)
        self._view = None
        self._context = ('', '')
        self._request_id = ''
        self._ready = False
        self._error = ''
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
        title = QLabel('测线预览', self)
        title.setObjectName('settingsBscanTitle')
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
            button.setEnabled(False)
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
        self._request_preview()
        self._refresh()

    def set_context(self, project_id: str, line_id: str) -> None:
        context = project_id, line_id
        if context == self._context:
            return
        self._context = context
        self._request_id = self._error = ''
        self._ready = False
        self._redraw.stop()
        if self._view is not None:
            self._view.clear()
            self._view.hide()
        self._current.setEnabled(False)
        self._baseline.setEnabled(False)
        self._update_caption()
        self._request_preview()

    def _request_preview(self) -> None:
        if not self.isVisible() or not all(self._context) or self._request_id or self._ready:
            return
        self._error = ''
        self._request_id = uuid4().hex
        self._update_caption()
        self.preview_requested.emit(*self._context, self._request_id)

    def set_result(self, project_id: str, line_id: str, token: str, bundle, error: str) -> None:
        if (project_id, line_id) != self._context or not token or token != self._request_id:
            return
        self._request_id = ''
        if bundle is None:
            self._error = error or '没有可预览的数据'
            self._update_caption()
            return
        if self._view is None:
            self._view = _StaticBScanView(self)
            self._view.apply_theme(self._dark)
            self._layout.insertWidget(1, self._view, 1)
        self._view.set_bundle(bundle)
        self._view.show()
        self._ready = True
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
        line_id = self._context[1]
        self._caption.setToolTip(self._error)
        if self._error:
            self._caption.setText(f'无法读取测线 {line_id} 的预览。')
        elif self._request_id:
            self._caption.setText(f'正在读取测线 {line_id}…')
        elif not self._ready:
            self._caption.setText('导入并选择测线后即可预览。')
        elif self._baseline.isChecked():
            self._caption.setText(f'测线 {line_id} · 基准关闭增益，使用对比度 1 和百分位 2–98。')
        else:
            self._caption.setText(f'测线 {line_id} · 原始数据降采样')

    def _refresh(self) -> None:
        if not self._ready or not self.isVisible():
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
