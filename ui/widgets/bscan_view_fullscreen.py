"""BScanView adapter for per-view, state-preserving fullscreen transitions."""

from qfluentwidgets import FluentIcon as FIF, ToolButton

from ui import constants
from ui.widgets.bscan_fullscreen import FullscreenHost


class BScanViewFullscreenMixin:
    def set_geometry_store(self, loader=None, saver=None) -> None:
        self._geometry_loader = loader
        self._geometry_saver = saver

    def is_fullscreen(self) -> bool:
        return (self._fullscreen is not None
                and self._fullscreen.state in ('expanding', 'fullscreen'))

    def set_thumbnail_mode(self, on: bool) -> None:
        self._thumbnail_mode = bool(on)
        for axis in ('bottom', 'left'):
            (self._plot.hideAxis if on else self._plot.showAxis)(axis)
        self.marker_strip.refresh_visibility()
        if on:
            self._fullscreen_btn.hide()
            for name in ('_fullscreen_fit_btn', '_fullscreen_close_btn'):
                button = getattr(self, name, None)
                if button is not None:
                    button.hide()
            self._plot.setTitle(None)
        else:
            self._layout_fullscreen_controls()
            if self._export_title:
                self._plot.setTitle(self._compact_title_html(self._export_title))

    def toggle_fullscreen(self) -> None:
        state = self._fullscreen.state if self._fullscreen is not None else 'embedded'
        if state in ('embedded', 'collapsing'):
            self.enter_fullscreen()
        else:
            self.exit_fullscreen()

    def enter_fullscreen(self) -> None:
        self._ensure_fullscreen_controls()
        if self._fullscreen is None:
            self._fullscreen = FullscreenHost(self.window())
            self._fullscreen.closed.connect(self._on_fullscreen_closed)
            self._fullscreen.take(self)
            self._move_companions_to_host()
        self._fullscreen.expand()
        self._layout_fullscreen_controls()

    def exit_fullscreen(self) -> None:
        if self._fullscreen is not None:
            self._fullscreen.collapse()

    def settle_fullscreen(self) -> None:
        if self._fullscreen is not None:
            self._fullscreen.settle()

    def _on_fullscreen_closed(self) -> None:
        host = self._fullscreen
        self._restore_companions()
        self._fullscreen = None
        self._layout_fullscreen_controls()
        if host is not None:
            host.deleteLater()

    def _ensure_fullscreen_controls(self) -> None:
        if getattr(self, '_fullscreen_fit_btn', None) is not None:
            return
        self._fullscreen_fit_btn = ToolButton(FIF.FIT_PAGE, self)
        self._fullscreen_fit_btn.setFixedSize(*constants.TOOL_BTN_COMPACT)
        self._fullscreen_fit_btn.setToolTip('回到全览')
        self._fullscreen_fit_btn.setAccessibleName('回到全览')
        self._fullscreen_fit_btn.setStyleSheet(self._fullscreen_btn.styleSheet())
        self._fullscreen_fit_btn.clicked.connect(self.fit_to_data)
        self._fullscreen_close_btn = ToolButton(FIF.CLOSE, self)
        self._fullscreen_close_btn.setFixedSize(*constants.TOOL_BTN_COMPACT)
        self._fullscreen_close_btn.setToolTip('关闭全屏并返回原视图（Esc）')
        self._fullscreen_close_btn.setAccessibleName('关闭全屏并返回原视图')
        self._fullscreen_close_btn.setStyleSheet(self._fullscreen_btn.styleSheet())
        self._fullscreen_close_btn.clicked.connect(self.exit_fullscreen)

    def _layout_fullscreen_controls(self) -> None:
        active = self._fullscreen is not None and self._fullscreen.state != 'embedded'
        plot_corner = self._glw.mapFromScene(self._plot.vb.sceneBoundingRect().topRight())
        plot_corner = self._glw.viewport().mapTo(self, plot_corner)
        right = max(6, min(self.width(), plot_corner.x()) - self._fullscreen_btn.width() - 6)
        top = max(6, plot_corner.y() + 6)
        self._fullscreen_btn.move(right, top)
        self._fullscreen_btn.setVisible(not active and not self._thumbnail_mode)
        self._fullscreen_btn.raise_()
        for name in ('_fullscreen_fit_btn', '_fullscreen_close_btn'):
            button = getattr(self, name, None)
            if button is not None:
                button.setVisible(active and not self._thumbnail_mode)
                button.raise_()
        if active:
            self._fullscreen_close_btn.move(right, top)
            self._fullscreen_fit_btn.move(
                max(6, right - self._fullscreen_close_btn.width() - 6), top)

    def _move_companions_to_host(self) -> None:
        self._fullscreen_companions = []
        for name in ('_ascan_popup', '_tvg_dialog'):
            widget = getattr(self, name, None)
            if widget is None:
                continue
            self._fullscreen_companions.append({
                'widget': widget,
                'parent': widget.parentWidget(),
                'flags': widget.windowFlags(),
                'geometry': widget.geometry(),
                'visible': widget.isVisible(),
            })
            widget.setParent(self._fullscreen, widget.windowFlags())
            if self._fullscreen_companions[-1]['visible']:
                widget.show()

    def _restore_companions(self) -> None:
        for record in getattr(self, '_fullscreen_companions', []):
            widget = record['widget']
            widget.setParent(record['parent'], record['flags'])
            widget.setGeometry(record['geometry'])
            widget.setVisible(record['visible'])
        self._fullscreen_companions = []
