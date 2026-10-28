"""Empty maps do not fetch tiles, and accepted close drains spatial workers."""
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtCore import QCoreApplication, QEvent, QRunnable


def _offline_workers(monkeypatch):
    from ui.widgets import map_view
    started = []
    class Worker(QRunnable):
        def __init__(self, *args):
            super().__init__()
            started.append(args)
        def run(self):
            pass
    monkeypatch.setattr(map_view, '_TileWorker', Worker)
    return started


def _track(mapped=True):
    return SimpleNamespace(line_id='L01', name='Synthetic', source='synthetic',
        coordinate_system='EPSG:4326' if mapped else 'local coordinates',
        points=[SimpleNamespace(x=x, y=y, elevation_m=1.) for x, y in (
            [(104., 30.), (104.001, 30.001)] if mapped else [(10000., 20000.), (10001., 20001.)])])


def _delete(view, qapp):
    shutdown = getattr(view, 'shutdown', None)
    if shutdown:
        shutdown()
    else:
        view._layer.shutdown()
        if view._overlay_layer is not None:
            view._overlay_layer.shutdown()
    view.close()
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_empty_or_unmapped_map_does_not_start_tile_downloads(qapp, monkeypatch):
    from ui.widgets.map_view import MapView
    started = _offline_workers(monkeypatch)
    view = MapView()
    try:
        view.resize(900, 600)
        view.show()
        qapp.processEvents()
        assert started == []
        view.set_tracks([_track(False)])
        qapp.processEvents()
        assert not view._layer.isVisible()
        assert view.prefetch_current_view() == 0
        assert started == []
        view.set_source('osm')
        qapp.processEvents()
        assert started == []
    finally:
        _delete(view, qapp)


def test_mapped_tracks_enable_basemap_and_clearing_tracks_hides_layers(qapp, monkeypatch):
    from ui.widgets.map_view import MapView
    started = _offline_workers(monkeypatch)
    view = MapView()
    try:
        view.resize(900, 600)
        view.set_tracks([_track()])
        view.show()
        qapp.processEvents()
        assert view._layer.isVisible()
        assert started
        view.set_tracks([])
        qapp.processEvents()
        assert not view._layer.isVisible()
        assert view._overlay_layer is None or not view._overlay_layer.isVisible()
    finally:
        _delete(view, qapp)


def test_shutdown_prevents_new_work_and_late_tile_updates(qapp):
    from ui.widgets.map_view import TileLayer
    layer = TileLayer()
    calls = []
    layer._pool = SimpleNamespace(clear=lambda: calls.append('clear'),
        waitForDone=lambda *args: calls.append(('wait', args)),
        start=lambda worker: calls.append('started'))
    try:
        layer.shutdown()
        layer.shutdown()
        assert calls == ['clear', ('wait', ())]
        layer._enqueue(1, 0, 0)
        layer._on_tile_finished(1, 0, 0, '')
        assert layer.prefetch_current_view() == 0
        assert layer.prefetch_region(104., 30., 104.001, 30.001) == 0
        assert calls == ['clear', ('wait', ())]
        assert not layer._failed
    finally:
        layer.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_main_window_only_shuts_spatial_page_after_close_is_accepted(qapp, tmp_path, monkeypatch):
    from ui.main_window import MyGPRMainWindow
    from ui.settings_manager import SettingsManager
    _offline_workers(monkeypatch)
    window = MyGPRMainWindow(settings=SettingsManager(str(tmp_path / 'settings.json')))
    window.ensure_pages_ready()
    spatial = window._page('spatialInterface')
    calls = []
    spatial.shutdown = lambda: calls.append('spatial')
    project = window.page_coordinator.project
    original = project.confirm_interpretation_close
    try:
        window.show()
        project.confirm_interpretation_close = lambda: False
        assert not window.close()
        assert calls == []
        project.confirm_interpretation_close = lambda: True
        assert window.close()
        assert calls == ['spatial']
    finally:
        project.confirm_interpretation_close = original
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


def test_tile_pool_is_drained_before_native_parent_deletion(tmp_path):
    program = dedent('''
        import os, socket, threading, time
        from PyQt6 import sip
        from PyQt6.QtCore import QRunnable
        from PyQt6.QtWidgets import QApplication
        from ui.widgets.map_view import TileLayer
        socket.socket.connect = lambda *args: (_ for _ in ()).throw(AssertionError('network forbidden'))
        app = QApplication([])
        entered = threading.Event()
        class Worker(QRunnable):
            def run(self):
                entered.set()
                time.sleep(4)
        layer = TileLayer(cache_root=os.environ['TILE_TEST_CACHE'])
        layer._pool.start(Worker())
        assert entered.wait(2)
        layer.shutdown()
        assert layer._pool.activeThreadCount() == 0
        sip.delete(layer)
        print('SPATIAL_POOL_EXIT', flush=True)
    ''')
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen', TILE_TEST_CACHE=str(tmp_path / 'cache'))
    result = subprocess.run([sys.executable, '-B', '-X', 'faulthandler', '-c', program],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=12)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'SPATIAL_POOL_EXIT' in result.stdout
