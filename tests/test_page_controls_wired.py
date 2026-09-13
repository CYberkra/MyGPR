# -*- coding: utf-8 -*-
"""页面控件接线回归防线 + P0 缺陷批次针对性回归。

防线背景：spatial_page 三维显示卡 4 个控件（地形来源下拉 / 导入 DEM /
清除 DEM / 影像贴图开关）曾因 _connect_internal 漏连而毫无响应。本文件守住：

1. 专门断言：上述 4 个控件的信号必须有接收者（本次回归的事发点）；
2. 通用防线：SpatialPage 上属性命名的交互控件（``_xxx_btn/_combo/_switch``
   等），其主信号至少 1 个接收者——后续重构再漏连时立即红；
3. 本批次其余修复的针对性回归（共享 SettingsManager、JobTable 删除漂移、
   delivery 勾选保持、AScanPopup 勾选态同步、DepthSliceView 自适应、
   controller 异步化）。
"""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py

from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QAbstractButton, QComboBox, QDoubleSpinBox, QLineEdit, QListWidget,
    QSlider, QSpinBox,
)

from ui.pages.spatial_page import SpatialPage  # noqa: E402


def _await(emitted: list, timeout_s: float = 5.0) -> None:
    """等异步 worker 的信号回包（无 pytest-qt，轮询 processEvents）。"""
    deadline = time.monotonic() + timeout_s
    while not emitted and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)


# ------------------------------------------------------------ 通用防线
# 只覆盖常见交互控件；纯展示控件不在其列、不检查。
# 注意：qfluentwidgets.ComboBox 继承 QPushButton（拆分按钮样式），
# 必须优先按 currentIndexChanged 识别，否则会误判成 clicked 按钮。


def _primary_signals(widget) -> list[str]:
    """返回控件应被接线的主信号名列表（无则空）。"""
    names = []
    # qfluentwidgets SwitchButton 不是 QAbstractButton，按信号特征识别
    if hasattr(widget, 'checkedChanged'):
        names.append('checkedChanged')
    if isinstance(widget, QComboBox) or (
            isinstance(widget, QAbstractButton)
            and hasattr(widget, 'currentIndexChanged')):
        # QComboBox 及 qfluentwidgets ComboBox（QPushButton 子类）
        names.append('currentIndexChanged')
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
        names.append('valueChanged')
    elif isinstance(widget, QLineEdit):
        names.append('textChanged')
    elif isinstance(widget, QListWidget):
        names.append('itemChanged')
    elif isinstance(widget, QAbstractButton):
        names.append('clicked')
    return names


def _named_interactive_widgets(page) -> list[tuple[str, object]]:
    """页面属性命名（``_xxx``）的交互控件：(属性名, 控件)。"""
    found = []
    for name, value in vars(page).items():
        if not name.startswith('_'):
            continue
        if _primary_signals(value):
            found.append((name, value))
    return found


# ============================================================ 空间页接线
class TestSpatialPageControlsWired:
    """SpatialPage 全部命名交互控件的接线防线。"""

    def test_every_named_interactive_widget_has_receiver(self, qapp):
        page = SpatialPage()
        widgets = _named_interactive_widgets(page)
        assert widgets, '未扫描到任何命名交互控件（扫描口径失效？）'
        unwired = []
        for name, widget in widgets:
            for sig_name in _primary_signals(widget):
                signal = getattr(widget, sig_name)
                if widget.receivers(signal) < 1:
                    unwired.append(f'{name}.{sig_name}')
        assert not unwired, f'控件未接线: {unwired}'

    def test_3d_terrain_combo_wired(self, qapp):
        """回归事发点 1：地形来源下拉 → _on_terrain_source_changed。"""
        page = SpatialPage()
        assert page._3d_terrain_combo.receivers(
            page._3d_terrain_combo.currentIndexChanged) >= 1

    def test_3d_dem_buttons_wired(self, qapp):
        """回归事发点 2/3：导入 / 清除 DEM 按钮。"""
        page = SpatialPage()
        assert page._3d_dem_btn.receivers(page._3d_dem_btn.clicked) >= 1
        assert page._3d_dem_clear_btn.receivers(
            page._3d_dem_clear_btn.clicked) >= 1

    def test_3d_imagery_switch_wired_and_drives_view(self, qapp):
        """回归事发点 4：影像贴图开关 → 3D 视图（含功能验证）。"""
        page = SpatialPage()
        assert page._3d_imagery_switch.receivers(
            page._3d_imagery_switch.checkedChanged) >= 1
        page._3d_imagery_switch.setChecked(False)
        assert page._3d_view._imagery_on is False
        page._3d_imagery_switch.setChecked(True)
        assert page._3d_view._imagery_on is True


# ============================================================ 设置持久化（共享实例）
class TestSharedSettingsManager:
    """修复 2：共享 SettingsManager 唯一写者，两页 save 不互相覆盖。"""

    def test_default_settings_contains_spatial_keys(self):
        from ui.settings_manager import DEFAULT_SETTINGS
        for key in ('spatial_basemap_source', 'spatial_terrain_source',
                    'spatial_left_collapsed', 'spatial_right_collapsed'):
            assert key in DEFAULT_SETTINGS, f'DEFAULT_SETTINGS 缺键 {key}'

    def test_two_pages_share_one_writer(self, qapp, tmp_path):
        from ui.pages.processing_page import ProcessingPage
        from ui.settings_manager import SettingsManager
        sm = SettingsManager(str(tmp_path / 'settings.json'))
        spatial = SpatialPage()
        spatial.set_settings_manager(sm)
        processing = ProcessingPage()
        processing.set_settings_manager(sm)

        spatial._persist_setting('spatial_terrain_source', 'estimated')
        processing._save_panel_state()

        # 共享实例：任一侧保存后另一侧的键都还在（旧实现互相覆盖丢键）
        assert sm.get('spatial_terrain_source') == 'estimated'
        assert 'processing_left_collapsed' in sm.get_all()
        data = json.loads((tmp_path / 'settings.json').read_text(
            encoding='utf-8'))
        assert data['spatial_terrain_source'] == 'estimated'
        assert 'processing_left_collapsed' in data

    def test_page_without_injected_manager_never_writes(self, qapp, tmp_path):
        """未注入（如单元测试）时页面静默跳过持久化，不读盘不写盘。"""
        from ui.settings_manager import SettingsManager
        sm = SettingsManager(str(tmp_path / 'settings.json'))
        spatial = SpatialPage()
        spatial.set_settings_manager(sm)
        spatial._persist_setting('spatial_terrain_source', 'estimated')

        orphan = SpatialPage()          # 未注入
        orphan._persist_setting('spatial_terrain_source', 'online')
        orphan._save_panel_state()
        assert sm.get('spatial_terrain_source') == 'estimated'

    def test_project_page_column_widths_roundtrip(self, qapp, tmp_path):
        """列宽新写入只走 SettingsManager（JSON list）。"""
        from ui.pages.project_page import ProjectPage
        from ui.settings_manager import SettingsManager
        path = tmp_path / 'settings.json'
        page = ProjectPage()
        page.set_settings_manager(SettingsManager(str(path)))
        page._lines_table.setColumnWidth(0, 123)
        page._save_column_widths(page._lines_table, 'lines')

        reloaded = SettingsManager(str(path))
        widths = reloaded.get('ui/project_page/lines_column_widths')
        assert isinstance(widths, list)
        assert widths[0] == 123

    def test_project_page_qsettings_migration(self, qapp, tmp_path,
                                              monkeypatch):
        """旧 QSettings 列宽读到即写入 SettingsManager 并清掉旧键。"""
        from ui.pages import project_page
        from ui.settings_manager import SettingsManager
        key = 'ui/project_page/lines_column_widths'
        store = {key: [111, 222]}

        class _FakeQSettings:
            def __init__(self, _org, _app):
                pass

            def value(self, k):
                return store.get(k)

            def setValue(self, k, v):
                store[k] = v

            def remove(self, k):
                store.pop(k, None)

        monkeypatch.setattr(project_page, 'QSettings', _FakeQSettings)
        from ui.pages.project_page import ProjectPage
        page = ProjectPage()    # 构造期恢复（此时无 sm，只应用不迁移）
        sm = SettingsManager(str(tmp_path / 'settings.json'))
        page.set_settings_manager(sm)   # 注入重放恢复 → 迁移发生

        widths = sm.get(key)
        assert isinstance(widths, list)
        assert widths[:2] == [111, 222]   # 旧列宽已落入 SettingsManager
        assert len(widths) == page._lines_table.columnCount()
        assert key not in store          # 旧键已清除
        header = page._lines_table.horizontalHeader()
        assert header.sectionSize(0) == 111


# ============================================================ JobTable.clear_finished
class TestJobTableClearFinished:
    """修复 3：非相邻多任务删除时行号漂移，删错/删不掉。"""

    def _make_table(self, qapp):
        from ui.widgets.job_widgets import JobTable
        table = JobTable()
        for jid in ('j1', 'j2', 'j3', 'j4'):
            table.upsert_job(jid, jid)
        return table

    def test_non_adjacent_finished_rows_removed(self, qapp):
        table = self._make_table(qapp)
        table.set_status('j1', 'completed')
        table.set_status('j3', 'failed')   # 非相邻终态
        table.clear_finished()

        assert table._table.rowCount() == 2
        assert set(table._rows) == {'j2', 'j4'}
        # 幸存行按可视顺序重排，映射与内容一致（删错行的回归断言）
        assert table._rows == {'j2': 0, 'j4': 1}
        assert table._table.item(0, table._COL_TITLE).text() == 'j2'
        assert table._table.item(1, table._COL_TITLE).text() == 'j4'

    def test_active_rows_untouched_when_nothing_finished(self, qapp):
        table = self._make_table(qapp)
        table.set_status('j2', 'running')
        table.clear_finished()
        assert table._table.rowCount() == 4
        assert table._rows == {'j1': 0, 'j2': 1, 'j3': 2, 'j4': 3}

    def test_trailing_finished_rows_removed(self, qapp):
        table = self._make_table(qapp)
        table.set_status('j3', 'completed')
        table.set_status('j4', 'cancelled')
        table.clear_finished()
        assert table._rows == {'j1': 0, 'j2': 1}
        assert table._table.rowCount() == 2


# ============================================================ DeliveryPage.set_lines
class TestDeliverySetLinesPreservesChecks:
    """修复 4：重刷测线集合后丢用户勾选。"""

    def test_checks_preserved_across_rebuild(self, qapp):
        from ui.pages.delivery_page import DeliveryPage
        page = DeliveryPage()
        lines = [SimpleNamespace(line_id='L01', name='L01'),
                 SimpleNamespace(line_id='L02', name='L02')]
        page.set_lines(lines)
        page._lines_list.item(0).setCheckState(Qt.CheckState.Checked)

        # 模拟 lines_updated 重刷（追加 L03）
        page.set_lines(lines + [SimpleNamespace(line_id='L03', name='L03')])

        assert page._lines_list.count() == 3
        assert (page._lines_list.item(0).checkState()
                == Qt.CheckState.Checked)    # L01 勾选保持
        assert (page._lines_list.item(1).checkState()
                == Qt.CheckState.Unchecked)
        assert (page._lines_list.item(2).checkState()
                == Qt.CheckState.Unchecked)  # 新测线维持默认不勾选

    def test_unchecked_stays_unchecked(self, qapp):
        from ui.pages.delivery_page import DeliveryPage
        page = DeliveryPage()
        page.set_lines([SimpleNamespace(line_id='L01', name='L01')])
        page.set_lines([SimpleNamespace(line_id='L01', name='L01')])
        assert (page._lines_list.item(0).checkState()
                == Qt.CheckState.Unchecked)


# ============================================================ AScanPopup ↔ BScanView
class TestAscanPopupClosedSync:
    """修复 5：用户关闭浮窗后宿主菜单勾选态失同步。"""

    def test_popup_emits_closed_signal(self, qapp):
        from ui.widgets.ascan_popup import AScanPopup
        popup = AScanPopup()
        hits = []
        popup.closed.connect(lambda: hits.append(1))
        popup.show()
        popup.close()
        assert hits == [1]
        assert popup._ascan_view is not None   # 关闭仅隐藏，实例复用

    def test_bscan_follow_reset_when_popup_closed(self, qapp):
        from ui.widgets.bscan_view import BScanView
        view = BScanView()
        view.set_ascan_follow(True)
        assert view._ascan_follow is True
        assert view._ascan_popup is not None

        view._ascan_popup.close()              # 用户点关浮窗

        assert view._ascan_popup.isHidden()
        assert view._ascan_follow is False     # 勾选态回落，下次菜单同步

    def test_bscan_follow_toggle_off_still_works(self, qapp):
        """回归：菜单取消勾选路径不受影响（hide 不触发 closed）。"""
        from ui.widgets.bscan_view import BScanView
        view = BScanView()
        view.set_ascan_follow(True)
        view.set_ascan_follow(False)
        assert view._ascan_follow is False
        assert view._ascan_popup.isHidden()


# ============================================================ DepthSliceView._auto_range
class TestDepthSliceAutoRange:
    """修复 6：无网格时永不自适应。"""

    def test_fits_tracks_without_grid(self, qapp):
        from ui.widgets.depth_slice_view import DepthSliceView
        view = DepthSliceView()
        track = SimpleNamespace(
            line_id='L01',
            points=tuple(SimpleNamespace(x=x, y=y) for x, y in
                         ((100.0, 200.0), (300.0, 400.0))))
        view.set_tracks([track], {'L01': '#ff0000'})

        rect = view._plot_item.vb.viewRect()
        assert rect.left() <= 100.0 and rect.right() >= 300.0
        # QRectF：y = ymin、y+height = ymax（pyqtgraph item 坐标 y 向上）
        assert rect.top() <= 200.0 and rect.bottom() >= 400.0

    def test_noop_without_grid_and_tracks(self, qapp):
        from ui.widgets.depth_slice_view import DepthSliceView
        view = DepthSliceView()
        view._auto_range()   # 不抛异常即可

    def test_grid_path_still_works(self, qapp):
        import numpy as np
        from ui.widgets.depth_slice_view import DepthSliceView
        view = DepthSliceView()
        view.set_grid(np.arange(9, dtype=float).reshape(3, 3),
                      x_origin_m=0.0, y_origin_m=10.0, cell_size_m=1.0)
        rect = view._plot_item.vb.viewRect()
        assert rect.width() >= 3.0
        assert rect.height() >= 3.0


# ============================================================ controller 异步化
class TestAsyncControllerIO:
    """修复 7：UI 线程同步 backend I/O 改 run_command + 信号回包。"""

    def test_get_artifact_descendants_async(self, qapp):
        from ui.controllers.project_controller import ProjectController
        calls = []

        def _descendants(project_id, line_id, aid):
            calls.append((project_id, line_id, aid))
            return (aid, 'child-1')

        backend = SimpleNamespace(
            list_artifact_descendants=_descendants,
            projects=SimpleNamespace(list_artifacts=lambda pid, lid: [
                SimpleNamespace(artifact_id='a1', name='成果A'),
                SimpleNamespace(artifact_id='child-1', name='子成果'),
            ]))
        controller = ProjectController()
        controller.set_backend(SimpleNamespace(backend=backend,
                                               job_bridge=None))
        controller._current = SimpleNamespace(project_id='P1')

        results = []
        controller.artifact_descendants_ready.connect(
            lambda lid, desc, names: results.append((lid, desc, names)))
        controller.get_artifact_descendants('L01', ['a1'])
        _await(results)

        assert calls == [('P1', 'L01', 'a1')]
        lid, descendants, names = results[-1]
        assert lid == 'L01'
        assert list(descendants) == ['a1', 'child-1']
        assert names['child-1'] == '子成果'

    def test_get_artifact_descendants_failure_emits_nothing(self, qapp):
        from ui.controllers.project_controller import ProjectController

        def _boom(project_id, line_id, aid):
            raise RuntimeError('backend down')

        backend = SimpleNamespace(
            list_artifact_descendants=_boom, projects=SimpleNamespace())
        controller = ProjectController()
        controller.set_backend(SimpleNamespace(backend=backend,
                                               job_bridge=None))
        controller._current = SimpleNamespace(project_id='P1')
        results = []
        controller.artifact_descendants_ready.connect(
            lambda *args: results.append(args))
        controller.get_artifact_descendants('L01', ['a1'])
        _await(results, timeout_s=1.0)
        assert results == []   # 失败不弹确认框（与原同步版返回 [] 同语义）

    def test_close_session_fire_and_forget(self, qapp):
        from ui.controllers.interpretation_controller import (
            InterpretationController)
        closed = []
        backend = SimpleNamespace(interpretation_edit=SimpleNamespace(
            close_session=lambda sid: closed.append(sid)))
        controller = InterpretationController()
        controller.set_backend(SimpleNamespace(backend=backend,
                                               job_bridge=None))
        controller._session_id = 'S1'
        controller.close_session()
        # fire-and-forget：调用即返回、会话立即失效，不等结果
        assert controller._session_id is None
        _await(closed)
        assert closed == ['S1']

    def test_close_session_without_backend_is_noop(self, qapp):
        from ui.controllers.interpretation_controller import (
            InterpretationController)
        controller = InterpretationController()
        controller.close_session()   # 不抛异常即可
