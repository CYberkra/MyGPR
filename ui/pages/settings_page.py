"""Application preferences with a searchable, display-only B-scan preview."""

import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QEvent, QPointF, QSignalBlocker, QStringListModel, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCompleter,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CheckBox,
    ComboBox,
    DoubleSpinBox,
    LineEdit,
    MessageBox,
    PushButton,
    ScrollArea,
    SpinBox,
)

from ui import constants, file_dialogs
from ui.page_scaffold import style_transparent_scroll
from ui.settings_support import read_application_version as _read_version
from ui.settings_support import settings_locations
from ui.widgets.settings_bscan_preview import SettingsBScanPreview
from ui.widgets.settings_controls import (
    SettingsChoice,
    SettingsDisclosure,
    settings_label,
    settings_stylesheet,
)
from ui.widgets.update_panel import UpdatePanel

_USER_GUIDE_URL = 'https://github.com/CYberkra/MyGPR/blob/main/docs/user/README.md'
_SEC_GAIN_STEPS = (
    tuple(value / 100 for value in range(101))
    + tuple(value / 10 for value in range(11, 41))
    + tuple(value / 2 for value in range(9, 41))
)
_TVG_DB_STEPS = tuple(value / 2 for value in range(121))
_TVG_POWER_STEPS = tuple(value / 100 for value in range(30, 301))
_AGC_WINDOW_STEPS = tuple(range(constants.BSCAN_AGC_WINDOW_RANGE[0],
                                constants.BSCAN_AGC_WINDOW_RANGE[1] + 1, 2))
_CATEGORIES = {
    'appearance': '外观与工作区',
    'bscan': 'B-scan 显示',
    'processing': '处理默认值',
    'spatial': '空间与底图',
    'storage': '存储与诊断',
    'about': '关于与更新',
}


@dataclass(frozen=True)
class _SettingTarget:
    label: str
    category: str
    control: QWidget
    keywords: str = ''
    disclosure: SettingsDisclosure | None = None

    @property
    def caption(self) -> str:
        return f'{self.label}    {_CATEGORIES[self.category]}'


def _finite(value, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _name(widget: QWidget, name: str, accessible: str) -> None:
    widget.setObjectName(name)
    widget.setAccessibleName(accessible)


def _column(parent: QWidget) -> QVBoxLayout:
    layout = QVBoxLayout(parent)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    return layout


class SettingsPage(QWidget):
    """Display preferences without owning writes or project data."""

    theme_changed = pyqtSignal(str)
    bscan_view_changed = pyqtSignal()
    general_changed = pyqtSignal()
    project_root_requested = pyqtSignal(str)
    shortcuts_requested = pyqtSignal()
    layout_reset_requested = pyqtSignal()
    diagnostics_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._manager = None
        self._project_root = constants.DEFAULT_PROJECT_ROOT
        self._percentiles = (2.0, 98.0)
        self._running_workers = None
        self._active_category = 'bscan'
        self._preview_wide = None
        self._targets: list[_SettingTarget] = []
        self._matches: list[_SettingTarget] = []
        self._highlighted: QWidget | None = None
        self._pending_focus: _SettingTarget | None = None
        self._focus_timer = QTimer(self)
        self._focus_timer.setSingleShot(True)
        self._focus_timer.timeout.connect(self._reveal)
        self._panels: dict[str, ScrollArea] = {}
        self._nav_buttons: dict[str, QPushButton] = {}
        self._location_buttons: dict[str, PushButton] = {}

        outer = _column(self)
        self._surface = QWidget(self, objectName='settingsSurface')
        self._surface.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        outer.addWidget(self._surface, 1)
        shell = _column(self._surface)
        shell.addWidget(self._build_search())

        main = QWidget(self._surface)
        body = QHBoxLayout(main)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self._sidebar = self._build_sidebar(main)
        body.addWidget(self._sidebar)
        self._body = QWidget(main)
        self._body_layout = _column(self._body)
        body.addWidget(self._body, 1)
        shell.addWidget(main, 1)

        self._status = settings_label('', 'hint', wrap=True)
        _name(self._status, 'settingsSaveStatus', '设置保存状态')
        self._status.setContentsMargins(14, 6, 14, 8)
        self._status.hide()
        shell.addWidget(self._status)

        for key, builder in (
            ('appearance', self._build_appearance),
            ('bscan', self._build_bscan),
            ('processing', self._build_processing),
            ('spatial', self._build_spatial),
            ('storage', self._build_storage),
            ('about', self._build_about),
        ):
            builder(key)
        self._select_category('bscan')
        self._sync_bscan_details()
        self._update_root_draft()
        self._update_viewport()
        self.apply_theme(False)

    def _build_search(self) -> QWidget:
        header = QWidget(self._surface)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)
        self._category_choice = ComboBox(header)
        self._protect_scroll_input(self._category_choice)
        _name(self._category_choice, 'settingsCategories', '设置分类')
        for key, label in _CATEGORIES.items():
            self._category_choice.addItem(label, userData=key)
        self._category_choice.currentIndexChanged.connect(
            lambda _index: self._select_category(self._category_choice.currentData()))
        layout.addWidget(self._category_choice)
        layout.addStretch()
        self._empty_search = settings_label('没有匹配项', 'hint')
        self._empty_search.hide()
        layout.addWidget(self._empty_search)

        self._search = QLineEdit(header)
        _name(self._search, 'settingsSearch', '搜索设置')
        self._search.setPlaceholderText('搜索设置')
        self._search.setClearButtonEnabled(True)
        self._search.setMaximumWidth(350)
        self._search.setMinimumWidth(108)
        self._search_model = QStringListModel(self)
        self._completer = QCompleter(self._search_model, self._search)
        self._completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self._completer.setMaxVisibleItems(8)
        _name(self._completer.popup(), 'settingsSearchResults', '设置搜索结果')
        self._search.setCompleter(self._completer)
        self._search.textEdited.connect(self._search_changed)
        self._search.returnPressed.connect(self._open_first_result)
        self._completer.activated[str].connect(self._open_result)
        layout.addWidget(self._search, 1)
        return header

    def _build_sidebar(self, parent: QWidget) -> QWidget:
        sidebar = QWidget(parent, objectName='settingsSidebar')
        sidebar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        sidebar.setFixedWidth(180)
        nav = QVBoxLayout(sidebar)
        nav.setContentsMargins(10, 10, 10, 0)
        nav.setSpacing(3)
        self._category_buttons = QButtonGroup(sidebar)
        for key, label in _CATEGORIES.items():
            button = QPushButton(label, sidebar)
            button.setProperty('settingsRole', 'nav')
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, category=key: self._select_category(category))
            self._category_buttons.addButton(button)
            self._nav_buttons[key] = button
            nav.addWidget(button)
        nav.addStretch()
        return sidebar

    def _category(self, key: str, description: str = '') -> QVBoxLayout:
        scroll = ScrollArea(self._body)
        style_transparent_scroll(scroll)
        panel = QWidget(scroll, objectName='settingsPanel')
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 14, 22, 22)
        layout.setSpacing(9)
        layout.addWidget(settings_label(_CATEGORIES[key], 'title'))
        if description:
            layout.addWidget(settings_label(description, 'hint', wrap=True))
        scroll.setWidget(panel)
        self._panels[key] = scroll
        self._body_layout.addWidget(scroll)
        scroll.hide()
        return layout

    @staticmethod
    def _row(layout: QVBoxLayout, label: str, *controls: QWidget, hint: str = '') -> QWidget:
        row = QWidget()
        row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        row.setProperty('settingsRole', 'row')
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(1, 7, 1, 7)
        row_layout.setSpacing(6)
        captions = QWidget(row)
        caption_layout = _column(captions)
        caption = settings_label(label, 'label', wrap=True)
        if controls and label:
            caption.setBuddy(controls[-1])
        caption_layout.addWidget(caption)
        if hint:
            caption_layout.addWidget(settings_label(hint, 'hint', wrap=True))
        row_layout.addWidget(captions, 1)
        for control in controls:
            row_layout.addWidget(control, 0)
        layout.addWidget(row)
        return row

    def _protect_scroll_input(self, control: QWidget) -> None:
        control.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        control.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Wheel and not watched.hasFocus():
            scroll = self._panels.get(self._active_category)
            if scroll is None or not scroll.isAncestorOf(watched):
                event.ignore()
                return True
            viewport = scroll.viewport()
            position = QPointF(viewport.mapFromGlobal(event.globalPosition().toPoint()))
            forwarded = QWheelEvent(
                position, event.globalPosition(), event.pixelDelta(), event.angleDelta(), event.buttons(),
                event.modifiers(), event.phase(), event.inverted(), device=event.pointingDevice(),
            )
            QApplication.sendEvent(viewport, forwarded)
            event.setAccepted(forwarded.isAccepted())
            return True
        return super().eventFilter(watched, event)

    def _combo(self, options: tuple[tuple[str, str], ...], name: str, label: str) -> ComboBox:
        control = ComboBox()
        self._protect_scroll_input(control)
        _name(control, name, label)
        for text, value in options:
            control.addItem(text, userData=value)
        return control

    def _double(self, name: str, label: str, minimum: float, maximum: float,
                value: float, step: float, decimals: int = 2) -> DoubleSpinBox:
        control = DoubleSpinBox()
        self._protect_scroll_input(control)
        _name(control, name, label)
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setSingleStep(step)
        control.setKeyboardTracking(False)
        control.setValue(value)
        return control

    def _gain_slider_editor(self, spin: DoubleSpinBox, key: str, steps: tuple[float, ...]) -> QWidget:
        editor = QWidget()
        layout = QHBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        slider = QSlider(Qt.Orientation.Horizontal)
        _name(slider, spin.objectName() + 'Slider', spin.accessibleName() + '滑条')
        slider.setProperty('settingsRole', 'displaySlider')
        self._protect_scroll_input(slider)
        slider.setFixedWidth(144)
        slider.setRange(0, len(steps) - 1)
        slider.setTracking(False)
        if key == 'bscan_gain_alpha':
            slider.setToolTip('0–1 段可细调。可在数字框输入精确值。')
        layout.addWidget(slider)

        value_stack = QStackedWidget()
        live_value = settings_label('', 'label')
        live_value.setObjectName(slider.objectName() + 'Draft')
        live_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_stack.addWidget(spin)
        value_stack.addWidget(live_value)
        layout.addWidget(value_stack)
        editor.setFocusProxy(spin)

        def format_value(value: float) -> str:
            return f'{value:.{spin.decimals()}f}{spin.suffix()}'

        def preview_position(position: int) -> None:
            value = steps[position]
            live_value.setText(format_value(value))
            value_stack.setCurrentWidget(live_value)
            self._preview.set_preferences(self.bscan_view_settings() | {key: value})

        def finish_drag() -> None:
            value_stack.setCurrentWidget(spin)
            self._refresh_preview()

        def sync_position() -> None:
            if slider.isSliderDown():
                return
            value = spin.value()
            position = min(range(len(steps)), key=lambda index: abs(steps[index] - value))
            with QSignalBlocker(slider):
                slider.setValue(position)
            slider.setAccessibleDescription(f'当前值 {format_value(value)}')

        slider.sliderMoved.connect(preview_position)
        slider.sliderReleased.connect(finish_drag)
        slider.valueChanged.connect(lambda position: spin.setValue(steps[position]))
        self._sync_gain_slider_positions.append(sync_position)
        return editor

    def _target(self, category: str, label: str, control: QWidget, keywords: str = '',
                disclosure: SettingsDisclosure | None = None) -> None:
        self._targets.append(_SettingTarget(label, category, control, keywords, disclosure))

    def _build_appearance(self, key: str) -> None:
        layout = self._category(key)
        self._theme_combo = SettingsChoice((
            ('浅色', constants.THEME_LIGHT, 'settingsThemeLight'),
            ('深色', constants.THEME_DARK, 'settingsThemeDark'),
        ))
        _name(self._theme_combo, 'settingsTheme', '主题')
        self._theme_combo.currentTextChanged.connect(self.theme_changed)
        self._row(layout, '主题', self._theme_combo)
        reset = PushButton('恢复面板布局…')
        reset.clicked.connect(self._confirm_layout_reset)
        self._row(layout, '面板布局', reset, hint='恢复侧栏、任务区和全屏图窗的位置或显示状态。')
        shortcuts = PushButton('查看快捷键')
        shortcuts.clicked.connect(self.shortcuts_requested)
        self._row(layout, '快捷键', shortcuts, hint='按操作或按键查找。')
        self._target(key, '主题', self._theme_combo, '浅色 深色 外观 颜色')
        self._target(key, '恢复面板布局', reset, '还原 默认 文件树 边栏 侧栏 窗口')
        self._target(key, '查看快捷键', shortcuts, 'F1 键盘')
        layout.addStretch()

    def _build_bscan(self, key: str) -> None:
        layout = self._category(key, '处理、解释和全屏图面共用此显示偏好。设置只影响显示。')
        self._preview_grid = QGridLayout()
        self._preview_grid.setHorizontalSpacing(16)
        self._preview_grid.setVerticalSpacing(12)
        layout.addLayout(self._preview_grid)
        self._bscan_controls = QWidget()
        form = _column(self._bscan_controls)
        self._build_bscan_direct(form, key)
        self._preview = SettingsBScanPreview()
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._bscan_advanced = QWidget()
        advanced = _column(self._bscan_advanced)
        self._build_bscan_levels(advanced, key)
        self._build_bscan_axes(advanced, key)
        self._build_bscan_overlays(advanced, key)
        layout.addStretch()

    def _build_bscan_direct(self, layout: QVBoxLayout, key: str) -> None:
        self._bscan_cmap_combo = self._combo(
            tuple((name, name) for name in constants.COLORMAPS), 'settingsColorMap', '颜色映射')
        self._bscan_cmap_combo.currentIndexChanged.connect(self._emit_bscan_changed)
        self._row(layout, '颜色映射', self._bscan_cmap_combo)
        self._bscan_contrast_spin = self._double('settingsContrast', '对比度', 0.5, 3.0, 1.0, 0.05)
        self._bscan_contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self._protect_scroll_input(self._bscan_contrast_slider)
        self._bscan_contrast_slider.setObjectName('settingsContrastSlider')
        self._bscan_contrast_slider.setProperty('settingsRole', 'displaySlider')
        self._bscan_contrast_slider.setAccessibleName('调节显示对比度')
        self._bscan_contrast_slider.setRange(50, 300)
        self._bscan_contrast_slider.setSingleStep(5)
        self._bscan_contrast_slider.setPageStep(10)
        self._bscan_contrast_slider.setValue(100)
        self._bscan_contrast_slider.setFixedWidth(91)
        self._bscan_contrast_slider.setTracking(False)
        self._bscan_contrast_slider.valueChanged.connect(
            lambda value: self._bscan_contrast_spin.setValue(value / 100))
        self._bscan_contrast_slider.sliderMoved.connect(self._preview_contrast)
        self._bscan_contrast_slider.sliderReleased.connect(self._refresh_preview)
        self._bscan_contrast_spin.valueChanged.connect(self._contrast_changed)
        self._bscan_contrast_reset = PushButton('重置')
        self._bscan_contrast_reset.setAccessibleName('恢复默认对比度')
        self._bscan_contrast_reset.setToolTip('对比度恢复为 1.00')
        self._bscan_contrast_reset.clicked.connect(lambda: self._bscan_contrast_spin.setValue(1.0))
        self._row(layout, '对比度', self._bscan_contrast_slider, self._bscan_contrast_spin,
                  self._bscan_contrast_reset)

        self._bscan_gain_combo = SettingsChoice((
            ('关闭', 'off', 'settingsGainOff'),
            ('SEC', 'sec', 'settingsGainSec'),
            ('TVG', 'tvg', 'settingsGainTvg'),
            ('分段 TVG', 'piecewise_tvg', 'settingsGainPiecewiseTvg'),
            ('限幅 AGC', 'agc', 'settingsGainAgc'),
        ))
        _name(self._bscan_gain_combo, 'settingsGainMode', '显示增益')
        self._bscan_gain_combo.currentIndexChanged.connect(self._emit_bscan_changed)
        self._row(layout, '显示增益', self._bscan_gain_combo)
        self._bscan_gain_alpha_spin = self._double(
            'settingsGainAlpha', 'SEC 衰减补偿系数', 0.0, 20.0, 0.2, 0.1)
        self._bscan_tvg_db_spin = self._double(
            'settingsTvgDb', 'TVG 深端总增益', 0.0, 60.0, 24.0, 0.1)
        self._bscan_tvg_power_spin = self._double(
            'settingsTvgPower', 'TVG 曲线幂次', 0.3, 3.0, 1.0, 0.01)
        self._bscan_tvg_db_spin.setSuffix(' dB')
        self._bscan_agc_window_spin = self._double(
            'settingsAgcWindow', 'AGC 窗口长度', *constants.BSCAN_AGC_WINDOW_RANGE,
            constants.BSCAN_AGC_WINDOW_DEFAULT, 2, decimals=0)
        self._bscan_agc_window_spin.setSuffix(' 点')
        self._bscan_agc_window_spin.setToolTip('窗口沿每道采样轴滑动。比较道间振幅时请关闭 AGC。')
        self._bscan_agc_db_spin = self._double(
            'settingsAgcMaxDb', 'AGC 最大增益', *constants.BSCAN_AGC_MAX_DB_RANGE,
            constants.BSCAN_AGC_MAX_DB_DEFAULT, 0.5)
        self._bscan_agc_db_spin.setSuffix(' dB')
        self._bscan_tvg_node_spins = []
        for tau, pref, default in constants.BSCAN_TVG_NODES:
            percent = round(tau * 100)
            spin = self._double(f'settingsTvgNode{percent}', f'采样轴 {percent}% 处增益',
                                *constants.BSCAN_TVG_NODE_RANGE, default, 0.5)
            spin.setSuffix(' dB')
            spin.setToolTip('节点按整幅采样轴定位，起点固定为 0 dB。')
            self._bscan_tvg_node_spins.append((spin, pref, default))
        gain_spins = (self._bscan_gain_alpha_spin, self._bscan_tvg_db_spin, self._bscan_tvg_power_spin,
                      self._bscan_agc_window_spin, self._bscan_agc_db_spin,
                      *(spin for spin, _, _ in self._bscan_tvg_node_spins))
        gain_value_width = max(spin.sizeHint().width() for spin in gain_spins)
        for spin in gain_spins:
            spin.setMinimumWidth(gain_value_width)
        self._sync_gain_slider_positions = []
        sec_editor = self._gain_slider_editor(self._bscan_gain_alpha_spin, 'bscan_gain_alpha', _SEC_GAIN_STEPS)
        tvg_db_editor = self._gain_slider_editor(self._bscan_tvg_db_spin, 'bscan_tvg_db', _TVG_DB_STEPS)
        tvg_power_editor = self._gain_slider_editor(
            self._bscan_tvg_power_spin, 'bscan_tvg_power', _TVG_POWER_STEPS)
        self._gain_rows = (
            ('sec', self._row(layout, 'SEC 衰减补偿', sec_editor, hint='dB/采样轴单位')),
            ('tvg', self._row(layout, 'TVG 深端总增益', tvg_db_editor)),
            ('tvg', self._row(layout, 'TVG 曲线幂次', tvg_power_editor)),
            *(('piecewise_tvg', self._row(layout, spin.accessibleName(),
                                         self._gain_slider_editor(spin, pref, _TVG_DB_STEPS)))
              for spin, pref, _ in self._bscan_tvg_node_spins),
            ('agc', self._row(layout, 'AGC 窗口长度', self._gain_slider_editor(
                self._bscan_agc_window_spin, 'bscan_agc_window', _AGC_WINDOW_STEPS))),
            ('agc', self._row(layout, 'AGC 最大增益', self._gain_slider_editor(
                self._bscan_agc_db_spin, 'bscan_agc_max_db', _TVG_DB_STEPS))),
        )
        for spin in gain_spins:
            spin.valueChanged.connect(self._emit_bscan_changed)
        self._target(key, '颜色映射', self._bscan_cmap_combo, '色图 色表 colormap seismic')
        self._target(key, '显示对比度', self._bscan_contrast_spin, 'contrast')
        self._target(key, '显示增益', self._bscan_gain_combo, 'SEC TVG 分段增益 节点 AGC 自动增益 衰减补偿')

    def _build_bscan_levels(self, layout: QVBoxLayout, key: str) -> None:
        body = QWidget()
        form = _column(body)
        self._bscan_p_low_spin = self._double(
            'settingsPercentileLow', '低百分位', 0, 100, 2, 0.5, 1)
        self._bscan_p_high_spin = self._double(
            'settingsPercentileHigh', '高百分位', 0, 100, 98, 0.5, 1)
        self._levels_apply = PushButton('应用')
        _name(self._levels_apply, 'settingsApplyLevels', '应用色阶百分位')
        self._levels_apply.clicked.connect(self._apply_percentiles)
        for spin in (self._bscan_p_low_spin, self._bscan_p_high_spin):
            spin.valueChanged.connect(self._update_levels_draft)
        self._row(form, '低百分位（%）', self._bscan_p_low_spin)
        self._row(form, '高百分位（%）', self._bscan_p_high_spin)
        self._levels_status = settings_label('', 'hint', wrap=True)
        self._levels_status.hide()
        actions = QHBoxLayout()
        actions.addWidget(self._levels_status, 1)
        actions.addWidget(self._levels_apply)
        form.addLayout(actions)
        self._bscan_levels_follow_check = CheckBox('色阶跟随查看范围')
        self._bscan_levels_follow_check.setChecked(True)
        self._bscan_levels_follow_check.setToolTip('统一色阶对照时沿用共同色阶。')
        self._bscan_levels_follow_check.toggled.connect(self._emit_bscan_changed)
        form.addWidget(self._bscan_levels_follow_check)
        self._levels = SettingsDisclosure('色阶百分位', body)
        layout.addWidget(self._levels)
        self._target(key, '色阶百分位', self._bscan_p_low_spin, '低位 高位 色标范围', self._levels)
        self._target(key, '色阶跟随查看范围', self._bscan_levels_follow_check, '自动色阶', self._levels)

    def _build_bscan_axes(self, layout: QVBoxLayout, key: str) -> None:
        body = QWidget()
        form = _column(body)
        self._bscan_aspect_combo = self._combo(
            (('拉伸铺满', 'free'), ('数据盒正方形', 'square'), ('数据格 1:1', 'cell')),
            'settingsAspect', '显示比例')
        self._bscan_x_axis_combo = self._combo(
            (('道号', 'trace'), ('距离 (m)', 'distance')), 'settingsXAxis', '横轴')
        self._bscan_y_axis_combo = self._combo(
            (('采样轴（时间/深度）', 'sample'), ('海拔 (m)', 'elevation')), 'settingsYAxis', '纵轴')
        self._bscan_y_axis_combo.setToolTip('缺少逐道高程或深度映射时，使用数据的采样轴。')
        controls = (
            ('显示比例', self._bscan_aspect_combo, '正方形 拉伸 数据格'),
            ('横轴', self._bscan_x_axis_combo, '距离 道号'),
            ('纵轴', self._bscan_y_axis_combo, '时间 深度 海拔 高程'),
        )
        self._axes = SettingsDisclosure('坐标与比例', body)
        for label, control, keywords in controls:
            self._row(form, label, control)
            control.currentIndexChanged.connect(self._emit_bscan_changed)
            self._target(key, label, control, '坐标 轴单位 ' + keywords, self._axes)
        form.addWidget(settings_label('预览固定使用道号和走时轴。', 'hint', wrap=True))
        layout.addWidget(self._axes)

    def _build_bscan_overlays(self, layout: QVBoxLayout, key: str) -> None:
        body = QWidget()
        form = _column(body)
        self._bscan_colorbar_check = CheckBox('显示色标')
        self._bscan_qc_marks_check = CheckBox('显示异常道标记')
        self._bscan_qc_marks_check.setToolTip('适用于带质检记录的工作区图面。')
        self._overlays = SettingsDisclosure('覆盖层', body)
        for control, keywords in ((self._bscan_colorbar_check, '颜色条'),
                                  (self._bscan_qc_marks_check, '质检 异常')):
            control.setChecked(True)
            control.toggled.connect(self._emit_bscan_changed)
            form.addWidget(control)
            self._target(key, control.text(), control, '覆盖层 ' + keywords, self._overlays)
        layout.addWidget(self._overlays)

    def _build_processing(self, key: str) -> None:
        layout = self._category(key)
        self._dielectric_spin = self._double(
            'settingsDielectric', '默认相对介电常数', 1, 81, constants.DEFAULT_DIELECTRIC, 0.5)
        self._dielectric_spin.valueChanged.connect(self._emit_general_changed)
        self._row(layout, '默认相对介电常数', self._dielectric_spin, hint='用于之后打开的导入表单。')
        self._workers_spin = SpinBox()
        self._protect_scroll_input(self._workers_spin)
        _name(self._workers_spin, 'settingsWorkers', '并行工作线程数')
        self._workers_spin.setRange(1, 8)
        self._workers_spin.setKeyboardTracking(False)
        self._workers_spin.setValue(constants.MAX_WORKERS)
        self._workers_spin.valueChanged.connect(self._emit_general_changed)
        self._row(layout, '并行工作线程数', self._workers_spin, hint='重启后生效。')
        self._target(key, '默认相对介电常数', self._dielectric_spin, '导入 εr')
        self._target(key, '并行工作线程数', self._workers_spin, '并发 重启')
        layout.addStretch()

    def _build_spatial(self, key: str) -> None:
        layout = self._category(key)
        self._prefetch_check = CheckBox('自动预下载测线区域底图')
        self._prefetch_check.setChecked(True)
        self._prefetch_check.toggled.connect(self._emit_general_changed)
        layout.addWidget(self._prefetch_check)
        layout.addWidget(settings_label('手动下载和可见地图按需加载仍可联网。', 'hint', wrap=True))
        self._target(key, self._prefetch_check.text(), self._prefetch_check, '地图 下载 联网')
        layout.addStretch()

    def _build_storage(self, key: str) -> None:
        layout = self._category(key)
        layout.addWidget(settings_label('默认项目根目录', 'label'))
        self._root_edit = LineEdit()
        _name(self._root_edit, 'settingsProjectRoot', '默认项目根目录')
        self._root_edit.setText(self._project_root)
        self._root_edit.textChanged.connect(self._update_root_draft)
        self._root_edit.returnPressed.connect(self._request_root)
        browse = PushButton('浏览…')
        browse.clicked.connect(self._browse_project_root)
        self._root_apply = PushButton('应用')
        _name(self._root_apply, 'settingsApplyProjectRoot', '应用默认项目根目录')
        self._root_apply.clicked.connect(self._request_root)
        path_row = QHBoxLayout()
        path_row.addWidget(self._root_edit, 1)
        path_row.addWidget(browse)
        path_row.addWidget(self._root_apply)
        layout.addLayout(path_row)
        self._root_status = settings_label('', 'hint', wrap=True)
        _name(self._root_status, 'settingsProjectRootStatus', '项目目录保存状态')
        layout.addWidget(self._root_status)
        self._target(key, '默认项目根目录', self._root_edit, '路径 新建 项目')
        for location, label, keywords in (
            ('settings', '设置文件目录', '配置'),
            ('logs', '日志目录', '错误 log'),
            ('tiles', '底图缓存目录', '瓦片 缓存'),
        ):
            button = PushButton('打开目录')
            button.setAccessibleName(f'打开{label}')
            button.clicked.connect(lambda _checked=False, kind=location: self._open_location(kind))
            self._location_buttons[location] = button
            self._row(layout, label, button)
            self._target(key, label, button, keywords)
        diagnostic = PushButton('预览诊断摘要')
        diagnostic.clicked.connect(self.diagnostics_requested)
        self._row(layout, '诊断摘要', diagnostic, hint='预览后可复制版本与运行状态。')
        self._target(key, '诊断摘要', diagnostic, '排查 版本')
        layout.addStretch()

    def _build_about(self, key: str) -> None:
        layout = self._category(key)
        self._row(layout, 'MyGPR', settings_label(_read_version(), 'hint'))
        self._row(layout, '作者', settings_label('邸建豪 袁林 詹萍', 'hint'))
        self._row(layout, '版权', settings_label('© 2025 MyGPR 保留所有权利', 'hint'))
        guide = PushButton('使用手册')
        guide.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(_USER_GUIDE_URL)))
        self._row(layout, '文档', guide)
        self.update_panel = UpdatePanel()
        layout.addWidget(self.update_panel)
        self._target(key, '使用手册', guide, '帮助 文档')
        update_target = settings_label('应用更新', 'label')
        update_target.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        layout.insertWidget(layout.indexOf(self.update_panel), update_target)
        self._target(key, '应用更新', update_target, '检查更新 发布 安装版')
        layout.addStretch()

    def _select_category(self, key: str) -> None:
        if key not in self._panels:
            return
        self._highlight(None)
        self._active_category = key
        self._nav_buttons[key].setChecked(True)
        with QSignalBlocker(self._category_choice):
            self._category_choice.setCurrentIndex(list(_CATEGORIES).index(key))
        for category, scroll in self._panels.items():
            scroll.setVisible(category == key)

    def _search_changed(self, text: str) -> None:
        words = text.casefold().split()
        self._matches = [
            target for target in self._targets
            if words and all(word in (target.caption + ' ' + target.keywords).casefold() for word in words)
        ]
        self._search_model.setStringList([target.caption for target in self._matches])
        self._empty_search.setVisible(bool(words and not self._matches))
        if self._matches:
            self._completer.complete()
        else:
            self._completer.popup().hide()

    def _open_first_result(self) -> None:
        if self._matches:
            self._go_to_target(self._matches[0])

    def _open_result(self, caption: str) -> None:
        target = next((item for item in self._targets if item.caption == caption), None)
        if target is not None:
            self._go_to_target(target)

    def _go_to_target(self, target: _SettingTarget) -> None:
        with QSignalBlocker(self._search):
            self._search.clear()
        self._search_changed('')
        self._select_category(target.category)
        if target.disclosure is not None:
            target.disclosure.set_expanded(True)
        self._pending_focus = target
        self._focus_timer.start(0)

    def _reveal(self) -> None:
        target, self._pending_focus = self._pending_focus, None
        if target is None or self._active_category != target.category:
            return
        if self._search.text() == target.caption:
            self._search.clear()
        self._panels[target.category].ensureWidgetVisible(target.control, 12, 16)
        row = target.control.parentWidget()
        if row is not None and row.property('settingsRole') == 'row':
            self._highlight(row)
        target.control.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _highlight(self, row: QWidget | None) -> None:
        if self._highlighted is not None:
            self._highlighted.setProperty('settingsFocus', False)
            self._highlighted.style().unpolish(self._highlighted)
            self._highlighted.style().polish(self._highlighted)
        self._highlighted = row
        if row is not None:
            row.setProperty('settingsFocus', True)
            row.style().unpolish(row)
            row.style().polish(row)

    def focus_bscan_settings(self) -> None:
        self._go_to_target(_SettingTarget('颜色映射', 'bscan', self._bscan_cmap_combo))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, '_preview_grid'):
            self._update_viewport()

    def _update_viewport(self) -> None:
        sidebar = self.width() >= 660
        self._sidebar.setVisible(sidebar)
        self._category_choice.setVisible(not sidebar)
        wide = self.width() - (self._sidebar.width() if sidebar else 0) >= 685
        if wide == self._preview_wide:
            return
        self._preview_wide = wide
        self._preview.setMinimumHeight(366 if wide else 285)
        for widget in (self._bscan_controls, self._preview, self._bscan_advanced):
            self._preview_grid.removeWidget(widget)
        self._preview_grid.addWidget(self._bscan_controls, 0, 0, Qt.AlignmentFlag.AlignTop)
        self._preview_grid.addWidget(self._preview, 0 if wide else 1, int(wide),
                                     2 if wide else 1, 1, Qt.AlignmentFlag.AlignTop)
        self._preview_grid.addWidget(self._bscan_advanced, 1 if wide else 2, 0, Qt.AlignmentFlag.AlignTop)
        self._preview_grid.setColumnStretch(0, 1)
        self._preview_grid.setColumnStretch(1, 1 if wide else 0)
        self._preview_grid.setRowStretch(1, 1 if wide else 0)

    def apply_theme(self, dark: bool) -> None:
        style = settings_stylesheet(dark)
        self._surface.setStyleSheet(style)
        self._completer.popup().setStyleSheet(style)

    def _contrast_changed(self, value: float) -> None:
        with QSignalBlocker(self._bscan_contrast_slider):
            self._bscan_contrast_slider.setValue(round(value * 100))
        self._emit_bscan_changed()

    def _preview_contrast(self, value: int) -> None:
        self._preview.set_preferences(self.bscan_view_settings() | {'bscan_contrast': value / 100})

    def _sync_bscan_details(self) -> None:
        mode = self._bscan_gain_combo.currentData()
        for row_mode, row in self._gain_rows:
            active = mode == row_mode
            row.setVisible(active)
            row.setEnabled(active)
        for sync_position in self._sync_gain_slider_positions:
            sync_position()
        self._bscan_contrast_reset.setEnabled(self._bscan_contrast_spin.value() != 1.0)
        self._levels.summary.setText(f'{self._percentiles[0]:g}–{self._percentiles[1]:g}%')
        self._axes.summary.setText(self._bscan_x_axis_combo.currentText())
        self._overlays.summary.setText('显示色标' if self._bscan_colorbar_check.isChecked() else '隐藏色标')
        self._update_levels_draft()

    def _level_values(self) -> tuple[float, float]:
        return self._bscan_p_low_spin.value(), self._bscan_p_high_spin.value()

    def _update_levels_draft(self) -> None:
        low, high = self._level_values()
        dirty = (low, high) != self._percentiles
        valid = 0 <= low < high <= 100
        self._levels_apply.setEnabled(dirty and valid)
        if dirty and self._status.property('settingsRole') == 'hint':
            self._status.hide()
        if not valid:
            self._show_status(self._levels_status, '低百分位须小于高百分位。', 'error')
        else:
            self._show_status(self._levels_status, '色阶修改待应用。' if dirty else '', 'hint')

    def _apply_percentiles(self) -> None:
        low, high = self._level_values()
        if not 0 <= low < high <= 100:
            self._show_status(self._levels_status, '低百分位须小于高百分位。', 'error')
            self._bscan_p_low_spin.setFocus()
            return
        self._percentiles = low, high
        self._emit_bscan_changed()
        if self._percentiles == (low, high):
            self._show_status(self._levels_status, '色阶已应用。', 'hint')

    def _emit_bscan_changed(self) -> None:
        self.bscan_view_changed.emit()
        self._sync_bscan_details()
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        self._preview.set_preferences(self.bscan_view_settings())

    def _emit_general_changed(self) -> None:
        self.general_changed.emit()

    def _update_root_draft(self) -> None:
        dirty = self._root_edit.text().strip() != self._project_root
        self._root_apply.setEnabled(dirty)
        self._show_status(self._root_status, '目录修改待应用。' if dirty else '用于新建项目的建议位置。', 'hint')

    @staticmethod
    def _show_status(label, text: str, role: str) -> None:
        label.setText(text)
        label.setProperty('settingsRole', role)
        label.style().unpolish(label)
        label.style().polish(label)
        label.setVisible(bool(text))

    def report_save(self, scope: str, saved: bool, message: str = '') -> None:
        if scope == 'general' and saved and self._running_workers != self._workers_spin.value():
            message = message or '设置已保存。线程数重启后生效。'
        message = message or ('设置已保存。' if saved else '设置未保存。保留上次成功保存的值。')
        label = self._root_status if scope == 'root' else self._status
        self._show_status(label, message, 'hint' if saved else 'error')

    def _request_root(self) -> None:
        if self._root_apply.isEnabled():
            self.project_root_requested.emit(self._root_edit.text().strip())

    def commit_project_root(self, path: str) -> None:
        self._project_root = path
        with QSignalBlocker(self._root_edit):
            self._root_edit.setText(path)
        self._update_root_draft()

    def _browse_project_root(self) -> None:
        path = file_dialogs.getExistingDirectory(
            self, '选择默认项目根目录', self._project_root)
        if path:
            self._root_edit.setText(path)

    def _confirm_layout_reset(self) -> None:
        dialog = MessageBox(
            '恢复面板布局',
            '恢复文件树、侧栏、底部输出面板和下次 B-scan 全屏窗口的位置或显示状态。'
            '已打开的项目、测线、图标签和草稿继续使用。',
            self.window(),
        )
        dialog.yesButton.setText('恢复布局')
        dialog.cancelButton.setText('取消')
        dialog.cancelButton.setFocus()
        if dialog.exec():
            self.layout_reset_requested.emit()

    def set_settings_manager(self, manager) -> None:
        self._manager = manager
        for key, path in settings_locations(manager).items():
            self._location_buttons[key].setToolTip(str(path))
            self._location_buttons[key].setAccessibleDescription(str(path))

    def _open_location(self, kind: str) -> None:
        if self._manager is None:
            return
        path = settings_locations(self._manager)[kind]
        try:
            if Path(path).is_dir() and QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
                return
        except OSError:
            pass
        self.report_save('location', False, f'无法打开目录。位置为 {path}')
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._location_buttons[kind].setToolTip(str(path))

    def load_settings(self, data: dict) -> None:
        """Replay persisted preferences without notifying their writers."""
        self.restore_preferences(data, 'theme')
        self.restore_preferences(data, 'general')
        self.commit_project_root(str(data.get('project_root', constants.DEFAULT_PROJECT_ROOT)))
        self.sync_bscan_view_settings(data, initial=True)
        self._running_workers = self._workers_spin.value()

    def restore_preferences(self, values: dict, scope: str) -> None:
        if scope == 'theme':
            self.set_theme_text(str(values.get('theme', constants.THEME_LIGHT)))
        elif scope == 'general':
            with ExitStack() as signals:
                for widget in (self._dielectric_spin, self._workers_spin, self._prefetch_check):
                    signals.enter_context(QSignalBlocker(widget))
                self._dielectric_spin.setValue(_finite(
                    values.get('default_dielectric'), constants.DEFAULT_DIELECTRIC))
                self._workers_spin.setValue(int(_finite(values.get('max_workers'), constants.MAX_WORKERS)))
                self._prefetch_check.setChecked(bool(values.get('auto_prefetch_basemap', True)))
        elif scope == 'bscan':
            self.sync_bscan_view_settings(values)

    def set_theme_text(self, text: str) -> None:
        with QSignalBlocker(self._theme_combo):
            self._theme_combo.setCurrentText(text)

    @staticmethod
    def _select_by_data(combo, value, fallback: str) -> None:
        index = combo.findData(str(value or fallback))
        combo.setCurrentIndex(index if index >= 0 else combo.findData(fallback))

    def sync_bscan_view_settings(self, values: dict, *, initial: bool = False) -> None:
        """Mirror independently edited view preferences and preserve pending input."""
        combos = (
            (self._bscan_aspect_combo, 'bscan_aspect_mode', 'free'),
            (self._bscan_x_axis_combo, 'bscan_x_axis', 'trace'),
            (self._bscan_y_axis_combo, 'bscan_y_axis', 'sample'),
            (self._bscan_gain_combo, 'bscan_gain_mode', 'off'),
            (self._bscan_cmap_combo, 'bscan_colormap', constants.DEFAULT_COLORMAP),
        )
        spins = (
            (self._bscan_contrast_spin, 'bscan_contrast', 1.0),
            (self._bscan_gain_alpha_spin, 'bscan_gain_alpha', 0.2),
            (self._bscan_tvg_db_spin, 'bscan_tvg_db', 24.0),
            (self._bscan_tvg_power_spin, 'bscan_tvg_power', 1.0),
            *self._bscan_tvg_node_spins,
            (self._bscan_agc_window_spin, 'bscan_agc_window', constants.BSCAN_AGC_WINDOW_DEFAULT),
            (self._bscan_agc_db_spin, 'bscan_agc_max_db', constants.BSCAN_AGC_MAX_DB_DEFAULT),
        )
        checks = (
            (self._bscan_colorbar_check, 'bscan_colorbar_visible'),
            (self._bscan_levels_follow_check, 'bscan_levels_follow_view'),
            (self._bscan_qc_marks_check, 'bscan_qc_marks_visible'),
        )
        keep_draft = not initial and self._level_values() != self._percentiles
        with ExitStack() as signals:
            for combo, key, default in combos:
                if key in values or initial:
                    signals.enter_context(QSignalBlocker(combo))
                    fallback = default if initial or key == 'bscan_colormap' else combo.currentData()
                    self._select_by_data(combo, values.get(key), fallback)
            for spin, key, default in spins:
                if key in values or initial:
                    signals.enter_context(QSignalBlocker(spin))
                    spin.setValue(_finite(values.get(key), default if initial else spin.value()))
            for check, key in checks:
                if key in values or initial:
                    signals.enter_context(QSignalBlocker(check))
                    check.setChecked(bool(values.get(key, True)))
            signals.enter_context(QSignalBlocker(self._bscan_contrast_slider))
            self._bscan_contrast_slider.setValue(round(100 * self._bscan_contrast_spin.value()))
            if initial or 'bscan_p_low' in values or 'bscan_p_high' in values:
                low = _finite(values.get('bscan_p_low'), self._percentiles[0])
                high = _finite(values.get('bscan_p_high'), self._percentiles[1])
                if 0 <= low < high <= 100:
                    self._percentiles = low, high
                elif initial:
                    self._percentiles = 2.0, 98.0
                if not keep_draft:
                    for spin, value in zip(
                        (self._bscan_p_low_spin, self._bscan_p_high_spin), self._percentiles, strict=True
                    ):
                        signals.enter_context(QSignalBlocker(spin))
                        spin.setValue(value)
        self._sync_bscan_details()
        self._refresh_preview()

    def bscan_view_settings(self) -> dict:
        return {
            'bscan_aspect_mode': self._bscan_aspect_combo.currentData(),
            'bscan_x_axis': self._bscan_x_axis_combo.currentData(),
            'bscan_y_axis': self._bscan_y_axis_combo.currentData(),
            'bscan_contrast': self._bscan_contrast_spin.value(),
            'bscan_colormap': self._bscan_cmap_combo.currentData(),
            'bscan_p_low': self._percentiles[0],
            'bscan_p_high': self._percentiles[1],
            'bscan_gain_mode': self._bscan_gain_combo.currentData(),
            'bscan_gain_alpha': self._bscan_gain_alpha_spin.value(),
            'bscan_tvg_db': self._bscan_tvg_db_spin.value(),
            'bscan_tvg_power': self._bscan_tvg_power_spin.value(),
            **{key: spin.value() for spin, key, _ in self._bscan_tvg_node_spins},
            'bscan_agc_window': int(self._bscan_agc_window_spin.value()),
            'bscan_agc_max_db': self._bscan_agc_db_spin.value(),
            'bscan_colorbar_visible': self._bscan_colorbar_check.isChecked(),
            'bscan_levels_follow_view': self._bscan_levels_follow_check.isChecked(),
            'bscan_qc_marks_visible': self._bscan_qc_marks_check.isChecked(),
        }

    def settings(self) -> dict:
        return {
            'theme': self._theme_combo.currentText(),
            'default_dielectric': self._dielectric_spin.value(),
            'max_workers': self._workers_spin.value(),
            'project_root': self._project_root,
            'auto_prefetch_basemap': self._prefetch_check.isChecked(),
            **self.bscan_view_settings(),
        }
