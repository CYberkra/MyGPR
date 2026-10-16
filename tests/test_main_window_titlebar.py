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
def window(qapp, monkeypatch, tmp_path):
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
        artifact_delete_requested = pyqtSignal(str, str)

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
    from ui.settings_manager import SettingsManager
    settings = SettingsManager(str(tmp_path / 'window-settings.json'))
    w = mw.MyGPRMainWindow(settings=settings)
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
    # FluentWidgets 的全局路由没有注销页面栈的接口；本夹具主动销毁
    # 窗口前清理它拥有的记录，避免后续用例访问已删除的 StackedWidget。
    from qfluentwidgets import qrouter
    stacked = w.stackedWidget
    qrouter.stackHistories.pop(stacked, None)
    qrouter.history[:] = [item for item in qrouter.history if item.stacked is not stacked]
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


def test_top_navigation_receives_mouse(window, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    # Data-dependent routes are gated until a project has lines.
    window._page('projectInterface').set_project_info(type('Summary', (), {'name': 'Test'})())
    window._page('projectInterface').set_lines([type('Line', (), {'line_id': 'L01'})()])
    for key in ('projectInterface', 'processingInterface', 'projectInterface'):
        item = window._top_nav.items[key]
        hit = QApplication.widgetAt(item.mapToGlobal(item.rect().center()))
        assert hit is item or item.isAncestorOf(hit), (key, type(hit).__name__)
        QTest.mouseClick(item, Qt.MouseButton.LeftButton)
        qapp.processEvents()
        assert window.stackedWidget.currentWidget() is window.pages[key]


def test_project_is_the_only_startup_navigation_entry(window):
    assert 'homeInterface' not in window._top_nav.items
    assert window.stackedWidget.count() == 7
    window._goto_page('homeInterface')
    assert window.stackedWidget.currentWidget() is window.pages['projectInterface']
    assert window._page('homeInterface') is window._page('projectInterface').startup


def test_empty_workspace_disables_data_navigation_with_reason(window):
    item = window._top_nav.items['processingInterface']
    assert not item.isEnabled()
    assert '项目' in item.toolTip()
    assert not window._file_tree_panel.isVisible()


def test_processing_uses_its_workspace_resource_button(window):
    from types import SimpleNamespace
    project = window._page('projectInterface')
    project.set_project_info(SimpleNamespace(project_id='P1', name='Test'))
    project.set_lines([SimpleNamespace(line_id='L01')])
    window._goto_page('processingInterface')
    assert not window._file_tree_btn.isVisible()
    window._goto_page('projectInterface')
    assert window._file_tree_btn.isVisible()


def test_future_split_views_receive_preferences_once(window, monkeypatch):
    from ui.widgets.bscan_view import BScanView
    view = BScanView(window)
    wired = []
    monkeypatch.setattr(window, '_wire_bscan_preferences',
                        lambda owner: wired.extend(owner.findChildren(BScanView)))
    processing = window._page('processingInterface')
    processing.bscan_view_created.emit(view)
    processing.bscan_view_created.emit(view)
    assert wired == [view]


def test_startup_and_close_use_existing_project_commands(window, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    created = Mock()
    opened = Mock()
    monkeypatch.setattr(window, '_show_new_project_dialog', created)
    monkeypatch.setattr(window, '_open_project_dialog', opened)
    project = window._page('projectInterface')
    project.startup.new_btn.click()
    project.startup.open_btn.click()
    created.assert_called_once_with()
    opened.assert_called_once_with()
    controller = SimpleNamespace(close_current=Mock())
    monkeypatch.setattr(window, 'project_controller', controller)
    monkeypatch.setattr(window.page_coordinator, 'require_project', lambda: True)
    project.close_project_requested.emit()
    controller.close_current.assert_called_once_with()


def test_project_switch_updates_name_even_when_both_projects_are_empty(window):
    from types import SimpleNamespace
    page = window._page('projectInterface')
    page.set_project_info(SimpleNamespace(project_id='A', name='First'))
    page.set_project_info(SimpleNamespace(project_id='B', name='Second'))
    assert 'Second' in window._project_switch_btn.text()


def test_missing_last_project_leaves_startup_with_specific_reason(window, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    missing = str(tmp_path / 'moved-project')
    window.settings.set('recent_projects', [missing])
    monkeypatch.setattr(window, 'project_controller', SimpleNamespace(open_project=Mock()))
    window._backend_ready = True
    window._last_project_restored = False
    window._restore_last_project()
    assert window._page('projectInterface').workspace_state() == 'startup'
    assert missing in window._page('homeInterface')._startup_error.text()


class TestOutputPanelSplitter:
    """输出面板 QSplitter 分配回归（问题：面板拖不动/一拖就消失）。

    根因：上格宿主 min hint 透传 QStackedWidget「所有页最大 min」（实测
    835 > splitter 总高），QSplitter 永远把下格压到自身最小；且下格 min
    hint 随收/展变化（37↔97）触发 invalidate 重算洗掉显式分配。
    修复：_PageHostWidget 覆写 min hint 固定 300 + OutputPanel content
    min 归零（面板 min hint 恒定）。
    """

    def test_content_host_min_hint_is_capped(self, window):
        """上格宿主 min hint 必须是固定上限，而非页面栈全页最大 min。"""
        import ui.main_window as mw
        host = window._v_splitter.widget(0)
        assert host.minimumSizeHint().height() == mw._CONTENT_HOST_MIN_H
        stacked_min = window.stackedWidget.minimumSizeHint().height()
        assert stacked_min > host.minimumSizeHint().height(), (
            '页面栈全页最大 min 应大于宿主上限，否则覆写无意义')

    def test_output_panel_min_hint_stable_across_toggle(self, window):
        """收/展切换时面板 min hint 恒定（invalidate 重算才不会洗掉分配）。"""
        panel = window.output_panel
        h_open = panel.minimumSizeHint().height()
        panel.set_open(False)
        h_closed = panel.minimumSizeHint().height()
        panel.set_open(True)
        h_reopen = panel.minimumSizeHint().height()
        assert h_open == h_closed == h_reopen, (
            f'min hint 随开合变化：open={h_open} closed={h_closed} '
            f'reopen={h_reopen}')

    def test_output_panel_height_assignable_and_sticky(self, window, qapp):
        """setSizes 显式分配下格高度必须生效且在事件排空后保持。"""
        from PyQt6.QtCore import QTimer
        from PyQt6.QtTest import QTest
        panel = window.output_panel
        panel.set_open(True)
        # 启动/展开会排队恢复保存高度；先处理这些零延迟回调，再模拟
        # 用户拖动，避免 Linux 事件顺序让恢复值覆盖本测试的显式分配。
        QTest.qWait(1)
        sp = window._v_splitter
        total = sum(sp.sizes())
        sp.setSizes([total - 260, 260])
        qapp.processEvents()
        assert sp.sizes()[1] >= 240, f'显式分配被驳回: {sp.sizes()}'

        sticky = []
        QTimer.singleShot(30, lambda: sticky.append(sp.sizes()[1]))
        # processEvents 不等 timer；processEvents + sleep 交替让 30ms 计时器到期
        for _ in range(10):
            qapp.processEvents()
            qapp.thread().msleep(15)
        assert sticky and sticky[0] >= 240, (
            f'事件排空后高度丢失（重算洗掉）: {sticky}')
