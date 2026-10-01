"""BScanView 全屏/视图范围 mixin。

2026-09-24 组件化收尾（自 bscan_view.py 机械搬迁，方法体零改动）：
「铺满」请求/几何持久化回调/全屏进出。注意区分：既有
``bscan_fullscreen.py`` 的 FullscreenHost 是全屏宿主窗口本体，
本模块只是 BScanView 的行为切片。
"""

from ui.widgets.bscan_fullscreen import FullscreenHost

class BScanViewFullscreenMixin:
    # ------------------------------------------------------------------ 视图范围
    def set_expand_enabled(self, enabled: bool) -> None:
        """由宿主页决定右键是否露「铺满窗口」（页面能折叠侧栏才有意义）。"""
        self._expand_enabled = bool(enabled)

    def set_geometry_store(self, loader=None, saver=None) -> None:
        """注入全屏窗口几何的读写回调（view 自己不碰文件/设置）。

        :param loader: ``() -> QByteArray | None``，回放历史几何；
        :param saver:  ``(QByteArray) -> None``，窗口关闭时保存。
        """
        self._geometry_loader = loader
        self._geometry_saver = saver

    def request_expand(self) -> None:
        """「铺满」：本视图不认识页面布局，发信号让宿主页面去折叠侧栏。"""
        self.sig_expand_requested.emit()

    def is_fullscreen(self) -> bool:
        """当前是否处于独立窗口展开态。"""
        return self._fullscreen is not None and self._fullscreen.isVisible()

    def set_thumbnail_mode(self, on: bool) -> None:
        """缩略模式（主辅布局的导航格）：隐藏轴/图内标题/全屏钮。

        缩略格只承担"认出这是哪个源 + 点击升主窗"两个职责，完整轴系在
        170px 宽里挤成噪音。升主窗时以同调用还原（轴重现、标题按
        ``_export_title`` 重挂、全屏钮回归）。
        """
        for axis in ('bottom', 'left'):
            (self._plot.hideAxis if on else self._plot.showAxis)(axis)
        self._fullscreen_btn.setVisible(not on)
        if on:
            self._plot.setTitle(None)      # pyqtgraph 隐藏标题的唯一方式
        elif self._export_title:
            self._plot.setTitle(self._compact_title_html(self._export_title))

    def toggle_fullscreen(self) -> None:
        """进出独立窗口展开（重复调用即切换）。"""
        if self.is_fullscreen():
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def enter_fullscreen(self) -> None:
        """把本控件临时改嫁到独立窗口（同一实例，交互状态完全连续）。

        几何回放走宿主注入的 loader；没有历史几何时给一个合理的初始尺寸
        （按屏幕的 85%，保证足够大的同时不遮死其它窗口）。
        """
        if self._fullscreen is None:
            self._fullscreen = FullscreenHost(self.window())
            self._fullscreen.closed.connect(self._on_fullscreen_closed)
        host = self._fullscreen
        host.take(self)
        restored = False
        if self._geometry_loader is not None:
            try:
                restored = host.restore_saved_geometry(self._geometry_loader())
            except Exception:  # noqa: BLE001 - 几何回放失败不该挡住全屏
                restored = False
        if not restored:
            screen = self.screen()
            available = screen.availableGeometry() if screen is not None else None
            if available is not None:
                host.resize(int(available.width() * 0.85),
                            int(available.height() * 0.85))
        host.show()
        host.raise_()
        host.activateWindow()
        self._fit_current_mode()

    def exit_fullscreen(self) -> None:
        """退出独立窗口展开（窗口自行 close，closed 信号里做收尾）。"""
        if self._fullscreen is not None:
            self._fullscreen.close()

    def _on_fullscreen_closed(self) -> None:
        """窗口关闭后：几何存回宿主，画布按当前比例重新铺满。"""
        if self._geometry_saver is not None and self._fullscreen is not None:
            try:
                self._geometry_saver(self._fullscreen.captured_geometry())
            except Exception:  # noqa: BLE001 - 存几何失败不该报错给用户
                pass
        self._fit_current_mode()

