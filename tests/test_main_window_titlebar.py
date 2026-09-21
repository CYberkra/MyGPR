# -*- coding: utf-8 -*-
"""标题栏按钮鼠标命中回归测试（真实例化主窗口）。

回归背景：OutputPanel 改造把右列容器 ``_right_area``（有实体的全高
widget，顶部 48px 标题栏留白由 widgetLayout 的 margin 承担）挂进根布局，
它创建于 titleBar 之后、z-order 压过标题栏右半——最小化/最大化/关闭
按钮的鼠标事件被裸容器吞掉（UIA invoke 有效、鼠标点击无效）。
修复：``_build_ui`` 末尾 ``titleBar.raise_()``。

本测试用 ``QApplication.widgetAt`` 断言三个按钮中心的最上层 widget 都
落在 titleBar 内；变异验证（手动 ``_right_area.raise_()`` 模拟回归）证明
该断言对 z-order 敏感。
"""
import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture
def window(qapp, monkeypatch):
    import ui.main_window as mw
    # 控制器置 None：不启动后端线程，纯 UI 组装
    for name in ('BackendController', 'ProjectController',
                 'ProcessingController', 'InterpretationController',
                 'DeliveryController'):
        monkeypatch.setattr(mw, name, None)
    # 文件树打桩为裸 QWidget：本测试只验标题栏 z-order，不需要树；
    # 且 offscreen 下 qfluentwidgets TreeWidget 与其他用例的树实例
    # 交替走原生窗口生命周期会触发库级 access violation（同
    # tests/test_file_tree.py fixture 注释记载的库级缺陷）
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import QWidget

    class _FileTreeStub(QWidget):
        """主窗口/接线器对文件树面板的全部调用点的 no-op 桩。"""
        line_selected = pyqtSignal(str)
        line_process_requested = pyqtSignal(str)
        line_delete_requested = pyqtSignal(list)
        delivery_focus_requested = pyqtSignal(str)
        artifact_focus_requested = pyqtSignal(str, str)

        def set_settings_manager(self, _settings):
            pass
        def apply_page(self, _name):
            pass
        def set_artifacts(self, _artifacts):
            pass
        def set_spatial_results(self, _results):
            pass
        def set_reports(self, _packages):
            pass

    monkeypatch.setattr(mw, 'FileTreePanel', _FileTreeStub)
    w = mw.MyGPRMainWindow()
    w.ensure_pages_ready()   # 非首屏页为预热构造，测试需显式确保全部就位
    # 开屏画面是与主窗口同位的独立 frameless 窗口（自带一套 TitleBar），
    # SPLASH_DURATION_MS(600ms) 未到时 QApplication.widgetAt 会命中它而不是
    # 主窗口按钮。构造提速把 __init__ 压到 600ms 以内后这个竞态在快机器上
    # 稳定复现（Linux offscreen 首挂）。本测试只验主窗口标题栏 z-order，
    # 显式关闭开屏以去掉时间依赖。
    w.splashScreen.close()
    w.show()
    qapp.processEvents()
    yield w
    w.close()
    w.deleteLater()
    qapp.processEvents()


def test_titlebar_buttons_receive_mouse(window):
    """三个标题栏按钮中心的最上层 widget 必须是按钮自身（落在 titleBar 内）。"""
    for attr in ('minBtn', 'maxBtn', 'closeBtn'):
        btn = getattr(window.titleBar, attr)
        hit = QApplication.widgetAt(btn.mapToGlobal(btn.rect().center()))
        assert hit is not None and window.titleBar.isAncestorOf(hit), (
            f'{attr} 中心命中最上层 widget '
            f'{type(hit).__name__ if hit is not None else None}，'
            '不在 titleBar 内——标题栏被遮挡，鼠标点击将被吞'
        )
