"""B-Scan 显示偏好 mixin——设置读盘 / 视图下发 / 设置页回写 / 全屏几何。

2026-10-05 组件化拆分（自 main_window.py 机械搬迁，方法体零改动）：
``MyGPRMainWindow`` 曾达 1104 行/60 方法，其中「B-Scan 显示偏好」是最自洽的
一块——17 个方法只服务一件事：把比例模式 / 轴单位 / 色标 / 色阶 / 增益 /
全屏几何这组偏好跨会话持久化，并与设置页控件保持双副本同步。

状态仍在 ``self``（mixin 不持有状态；``self.settings`` / ``self.pages`` /
``self._page()`` 均由主窗口提供），对外 API 与信号零变化——
``MyGPRMainWindow(BScanPreferenceMixin, FluentWindow)``。

本簇与主窗口其余部分**无双向纠缠**：簇内方法互调，簇外只从
``_register_page`` 调 ``_wire_bscan_preferences``、从 ``_connect_signals``
调 ``_on_settings_bscan_changed``（另两处单点入口）。
"""

from weakref import WeakSet, ref

from PyQt6 import sip
from PyQt6.QtCore import QTimer
from ui import constants
from ui.settings_manager import DEFAULT_SETTINGS
from ui.widgets.bscan_view import BScanView


class BScanPreferenceMixin:
    """B-Scan 显示偏好的持久化与下发（读写设置 + 同步设置页控件）。"""

    def _wire_bscan_preferences(self, page) -> None:
        """把页面内所有 BScanView 的显示偏好接到持久化设置（跨会话记住）。

        覆盖：比例模式 / 横纵轴单位 / 色标映射 / 色阶百分位 / 色标显隐 /
        全屏窗口几何。恢复阶段一律 ``notify=False``，否则形成「读设置 →
        写设置」回环。
        """
        views = page.findChildren(BScanView)
        proc = self.pages.get('processingInterface')
        if proc is not None and hasattr(proc, 'set_colorbar_pref'):
            # 网格卡新建于恢复之后：偏好状态须同步给网格（新卡继承）
            proc.set_colorbar_pref(
                self._setting_flag('bscan_colorbar_visible', True))
        if not views:
            return
        aspect = self._setting_choice('bscan_aspect_mode', ('free', 'square', 'cell'), 'free')
        x_axis = self._setting_choice('bscan_x_axis', ('trace', 'distance'), 'trace')
        y_axis = self._setting_choice('bscan_y_axis', ('sample', 'elevation'), 'sample')
        p_low, p_high = self._setting_levels()
        colorbar_visible = self._setting_flag('bscan_colorbar_visible', True)
        cmap = self._setting_choice('bscan_colormap',
                                    tuple(constants.COLORMAPS),
                                    constants.DEFAULT_COLORMAP)
        gain_mode = self._setting_choice('bscan_gain_mode',
                                         ('off', 'sec', 'tvg'), 'off')
        try:
            gain_alpha = float(self.settings.get('bscan_gain_alpha', 0.2))
        except (TypeError, ValueError):
            gain_alpha = 0.2
        gain_alpha = min(max(gain_alpha, 0.0), 20.0)
        tvg_db, tvg_power = self._setting_tvg_params()
        for view in views:
            view.set_global_display_preferences(self._global_bscan_display_state())
            view.set_levels_follow_view(self._setting_flag('bscan_levels_follow_view', True))
            view.set_aspect_mode(aspect, notify=False)
            view.set_axis_modes(x_axis, y_axis, notify=False)
            view.set_colormap(cmap)
            view.set_contrast(self.settings.get('bscan_contrast', 1.0))
            view.set_display_levels(p_low, p_high, notify=False)
            view.set_colorbar_visible(colorbar_visible, notify=False)
            view.set_qc_marks_visible(self._setting_flag('bscan_qc_marks_visible', True))
            view.set_gain(gain_mode, alpha=gain_alpha,
                          db=tvg_db, power=tvg_power, notify=False)
            view.set_geometry_store(loader=self._load_fullscreen_geometry,
                                    saver=self._save_fullscreen_geometry)
            self._connect_bscan_signals(view)
        # 面板数由处理页 tab 模型驱动（tab 数 → resolve_auto），容器级
        # 布局偏好已随 free/手动档位一并退役（2026-09-24）。

    def _connect_bscan_signals(self, view) -> None:
        """把用户操作（而非恢复动作）接到设置写盘。

        ⚠️ 必须**同时回写设置页控件**（``_mirror_bscan_setting``）：设置页的
        ComboBox/SpinBox 是 B-Scan 偏好的第二份副本，而 ``closeEvent`` 会把
        ``settings_page.settings()`` 整体回写设置文件。若工具条改了偏好却不
        同步到设置页，关窗时就会被那份**过期的默认值**覆盖掉——表现为「在
        工具条切了方形，重开又变回铺满」。这是排查过的真实缺陷，别只写盘。
        """
        if not hasattr(self, '_bscan_preference_views'):
            self._bscan_preference_views = WeakSet()
        if view in self._bscan_preference_views:
            return
        self._bscan_preference_views.add(view)
        view_ref = ref(view)
        view.sig_display_settings_requested.connect(lambda: self._open_bscan_settings(view_ref()))
        view.sig_aspect_changed.connect(
            lambda mode: self._mirror_bscan_setting('bscan_aspect_mode', str(mode)))
        view.sig_x_axis_changed.connect(
            lambda mode: self._mirror_bscan_setting('bscan_x_axis', str(mode)))
        view.sig_y_axis_changed.connect(
            lambda mode: self._mirror_bscan_setting('bscan_y_axis', str(mode)))
        view.sig_colormap_changed.connect(
            lambda name: self._mirror_bscan_setting('bscan_colormap', str(name)))
        view.sig_levels_changed.connect(
            lambda low, high: self._mirror_bscan_levels(low, high))
        view.sig_colorbar_visible_changed.connect(
            lambda visible: self._mirror_bscan_setting(
                'bscan_colorbar_visible', bool(visible)))
        view.sig_gain_changed.connect(
            lambda mode: self._mirror_bscan_setting('bscan_gain_mode',
                                                    str(mode)))
        view.sig_gain_params_changed.connect(self._mirror_bscan_tvg_params)
        view.sig_qc_marks_visible_changed.connect(self._on_qc_marks_visible_changed)

    def _open_bscan_settings(self, source=None) -> None:
        if source is not None and not sip.isdeleted(source):
            source.settle_fullscreen()
        self._goto_page('settingsInterface')
        page = self._page('settingsInterface')
        if page is not None:
            page.window().raise_()
            page.window().activateWindow()
            QTimer.singleShot(0, page.focus_bscan_settings)

    def _global_bscan_display_state(self) -> dict:
        """Preferences moved to Settings must not be restored per line/tab."""
        state = {field: self.settings.get(key, DEFAULT_SETTINGS[key]) for key, field in (
            ('bscan_aspect_mode', 'aspect'), ('bscan_x_axis', 'x_axis'),
            ('bscan_y_axis', 'y_axis'), ('bscan_colormap', 'colormap'),
            ('bscan_colorbar_visible', 'colorbar'), ('bscan_contrast', 'contrast'),
        )}
        state['levels'] = list(self._setting_levels())
        state.update(gain=self._setting_choice('bscan_gain_mode', ('off', 'sec', 'tvg'), 'off'),
                     gain_alpha=self.settings.get('bscan_gain_alpha', 0.2))
        state['gain_db'], state['gain_power'] = self._setting_tvg_params()
        return state

    def _on_qc_marks_visible_changed(self, visible: bool) -> None:
        """One remembered display preference; no processing or preview reload."""
        self._persist_setting('bscan_qc_marks_visible', bool(visible))
        self._sync_settings_page_bscan({'bscan_qc_marks_visible': bool(visible)})
        # Fullscreen temporarily reparents a view out of its page subtree.
        for view in list(self._bscan_preference_views):
            if not sip.isdeleted(view):
                view.set_qc_marks_visible(visible)

    def _mirror_bscan_setting(self, key: str, value) -> None:
        """写盘 + 把设置页控件同步到同一值（避免关窗回写覆盖用户选择）。"""
        self._persist_setting(key, value)
        self._sync_settings_page_bscan({key: value})
        for view in self._iter_bscan_views():
            view.set_global_display_preferences(self._global_bscan_display_state())

    def _mirror_bscan_levels(self, low: float, high: float) -> None:
        """色阶两键一并写盘与同步（避免中途崩溃留下 low > high）。"""
        self._persist_bscan_levels(low, high)
        self._sync_settings_page_bscan({
            'bscan_p_low': float(low), 'bscan_p_high': float(high)})
        for view in self._iter_bscan_views():
            view.set_global_display_preferences(self._global_bscan_display_state())

    def _sync_settings_page_bscan(self, values: dict) -> None:
        """把 B-Scan 偏好同步进设置页控件（blockSignals 防信号回环）。"""
        settings_page = self._page('settingsInterface')
        syncer = getattr(settings_page, 'sync_bscan_view_settings', None)
        if callable(syncer):
            syncer(values)

    def _setting_choice(self, key: str, allowed: tuple, fallback: str) -> str:
        """读枚举型设置：越界/损坏一律回落默认（不让坏设置影响出图）。"""
        value = str(self.settings.get(key, fallback) or fallback)
        return value if value in allowed else fallback

    def _setting_flag(self, key: str, fallback: bool) -> bool:
        """读布尔型设置：兼容 JSON bool 与 'true'/'false' 文本，坏值回落。"""
        value = self.settings.get(key, fallback)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ('true', '1', 'yes'):
            return True
        if text in ('false', '0', 'no'):
            return False
        return fallback

    def _iter_bscan_views(self):
        """All wired views, including views temporarily detached to fullscreen."""
        views = getattr(self, '_bscan_preference_views', ())
        for view in list(views):
            if not sip.isdeleted(view):
                yield view

    def _on_settings_bscan_changed(self) -> None:
        """设置页改了 B-Scan 视图项：即时下发到所有视图并写盘（无需重启）。"""
        settings_page = self._page('settingsInterface')
        getter = getattr(settings_page, 'bscan_view_settings', None)
        if not callable(getter):
            return
        values = getter()
        changed = {key for key, value in values.items() if self.settings.get(key) != value}
        for key, value in values.items():
            self.settings.set(key, value)
        self.settings.save()
        for view in self._iter_bscan_views():
            view.set_global_display_preferences(self._global_bscan_display_state())
            view.set_levels_follow_view(bool(values.get('bscan_levels_follow_view', True)))
            view.set_qc_marks_visible(bool(values.get('bscan_qc_marks_visible', True)))
            if 'bscan_contrast' in changed:
                view.set_contrast(values.get('bscan_contrast', 1.0))
            # Changing a global checkbox must retain wheel/rectangle zoom.
            if 'bscan_aspect_mode' in changed and view.aspect_mode() != values['bscan_aspect_mode']:
                view.set_aspect_mode(values['bscan_aspect_mode'], notify=False)
            if ({'bscan_x_axis', 'bscan_y_axis'} & changed
                    and (view.x_axis_mode(), view.y_axis_mode()) != (values['bscan_x_axis'], values['bscan_y_axis'])):
                view.set_axis_modes(values['bscan_x_axis'], values['bscan_y_axis'], notify=False)
            if 'bscan_colormap' in changed:
                view.set_colormap(str(values.get('bscan_colormap', constants.DEFAULT_COLORMAP)))
            if {'bscan_p_low', 'bscan_p_high'} & changed:
                view.set_display_levels(values['bscan_p_low'], values['bscan_p_high'], notify=False)
            if 'bscan_colorbar_visible' in changed:
                view.set_colorbar_visible(bool(values.get('bscan_colorbar_visible', True)), notify=False)
            if {'bscan_gain_mode', 'bscan_gain_alpha', 'bscan_tvg_db', 'bscan_tvg_power'} & changed:
                tvg_db, tvg_power = self._setting_tvg_params()
                view.set_gain(str(values.get('bscan_gain_mode', 'off')),
                              alpha=float(values.get('bscan_gain_alpha', 0.2)),
                              db=tvg_db, power=tvg_power, notify=False)
        proc = self.pages.get('processingInterface')
        if proc is not None and hasattr(proc, 'set_colorbar_pref'):
            proc.set_colorbar_pref(
                bool(values.get('bscan_colorbar_visible', True)))
        # 面板数由处理页 tab 模型驱动，容器布局设置已退役（2026-09-24）

    def _setting_levels(self) -> tuple[float, float]:
        """读色阶百分位；非法值回落 BScanView 默认（2 / 98）。"""
        try:
            low = float(self.settings.get('bscan_p_low', 2.0))
            high = float(self.settings.get('bscan_p_high', 98.0))
        except (TypeError, ValueError):
            return 2.0, 98.0
        if not 0.0 <= low < high <= 100.0:
            return 2.0, 98.0
        return low, high

    def _persist_setting(self, key: str, value) -> None:
        """用户操作触发的一次写盘（共享 SettingsManager 唯一写者）。"""
        self.settings.set(key, value)
        self.settings.save()

    def _setting_tvg_params(self) -> tuple[float, float]:
        """读 TVG 参数（dB/弯度）；坏值回落默认，弯度夹到曲线安全区间。"""
        try:
            db = float(self.settings.get('bscan_tvg_db', 24.0))
        except (TypeError, ValueError):
            db = 24.0
        try:
            power = float(self.settings.get('bscan_tvg_power', 1.0))
        except (TypeError, ValueError):
            power = 1.0
        return min(max(db, 0.0), 60.0), min(max(power, 0.3), 3.0)

    def _mirror_bscan_tvg_params(self, db: float, power: float) -> None:
        values = {'bscan_tvg_db': float(db), 'bscan_tvg_power': float(power)}
        for key, value in values.items():
            self.settings.set(key, value)
        self.settings.save()
        self._sync_settings_page_bscan(values)

    def _persist_bscan_levels(self, low: float, high: float) -> None:
        """色阶两个键必须同一次写盘，否则中途崩溃会留下 low>high 的坏状态。"""
        self.settings.set('bscan_p_low', float(low))
        self.settings.set('bscan_p_high', float(high))
        self.settings.save()

    def _load_fullscreen_geometry(self):
        """回放 B-Scan 全屏窗口几何；从未存过（空串）返回 None 走默认尺寸。"""
        from PyQt6.QtCore import QByteArray
        raw = self.settings.get('bscan_fullscreen_geometry', '')
        if not raw:
            return None
        try:
            return QByteArray.fromBase64(str(raw).encode('ascii'))
        except (ValueError, UnicodeEncodeError):
            return None

    def _save_fullscreen_geometry(self, blob) -> None:
        """把 QByteArray 几何转 base64 存入 JSON 设置（空几何不覆盖旧值）。"""
        if blob is None or blob.isEmpty():
            return
        payload = bytes(blob.toBase64().data()).decode('ascii')
        self._persist_setting('bscan_fullscreen_geometry', payload)
