"""Closing terrain views cancels future tiles and drains active Qt workers."""
from pathlib import Path
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip('PyQt6')


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


def _worker(tmp_path, prep):
    from ui.widgets.trajectory_3d_view import _TerrainWorker
    signals = SimpleNamespace(finished=_Signal(), notice=_Signal())
    worker = _TerrainWorker(1, prep, str(tmp_path), signals)
    worker._cancel_requested = threading.Event()
    return worker, signals


def _prep():
    return dict(zoom=1, tile_range=(0, 1, 0, 0), shape=(2, 2),
                gx=np.arange(2), gy=np.arange(2), mx=np.zeros(4), my=np.zeros(4))


def test_cancelled_terrain_does_not_start_downloads_or_emit(tmp_path):
    worker, signals = _worker(tmp_path, _prep())
    fetch = worker._fetch_tile = Mock(return_value=np.ones((256, 256)))
    worker._cancel_requested.set()
    worker.run()
    fetch.assert_not_called()
    assert signals.finished.values == []
    assert signals.notice.values == []


def test_cancellation_after_terrain_request_skips_next_tile_and_result(tmp_path):
    worker, signals = _worker(tmp_path, _prep())
    calls = []

    def fetch(*tile):
        calls.append(tile)
        worker._cancel_requested.set()
        return np.ones((256, 256))

    worker._fetch_tile = fetch
    worker.run()
    assert len(calls) == 1
    assert signals.finished.values == []
    assert signals.notice.values == []


def test_cancellation_before_status_emission_skips_notice(tmp_path):
    worker, signals = _worker(tmp_path, _prep())

    def fetched(*args):
        worker._cancel_requested.set()
        return {}

    worker._fetch_tiles = fetched
    worker.run()
    assert signals.notice.values == []
    assert signals.finished.values == []


def test_cancellation_after_imagery_request_skips_next_tile_and_result(tmp_path, monkeypatch):
    from ui.widgets import trajectory_3d_view as module
    prep = _prep()
    prep['estimate'] = dict(px=[0, 1], py=[0, 1], pz=[1, 1], qx=[0] * 4, qy=[0] * 4)
    prep['imagery'] = dict(zoom=1, tile_range=(0, 1, 0, 0), qpx=[0] * 4, qpy=[0] * 4)
    worker, signals = _worker(tmp_path, prep)
    calls = []
    monkeypatch.setattr(module, 'interpolate_scatter', lambda *args: np.ones(4))
    monkeypatch.setattr(module, 'sample_imagery_pixels', lambda *args: np.ones((4, 3)))

    def fetch(*tile):
        calls.append(tile)
        worker._cancel_requested.set()
        return np.ones((256, 256, 3), dtype=np.uint8)

    worker._fetch_imagery_tile = fetch
    worker.run()
    assert len(calls) == 1
    assert signals.finished.values == []
    assert signals.notice.values == []


@pytest.mark.parametrize('method', ['_fetch_tile', '_fetch_imagery_tile'])
def test_cancelled_http_response_is_not_written_to_cache(tmp_path, monkeypatch, method):
    from ui.widgets import trajectory_3d_view as module
    worker, _signals = _worker(tmp_path, _prep())

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            worker._cancel_requested.set()
            return b'synthetic incomplete image'

    monkeypatch.setattr(module.urllib.request, 'urlopen', lambda *args, **kwargs: Response())
    assert getattr(worker, method)(1, 0, 0) is None
    assert not [path for path in tmp_path.rglob('*') if path.is_file()]


def test_closed_terrain_view_cannot_queue_new_work(qapp, monkeypatch):
    from ui.widgets.trajectory_3d_view import Trajectory3DView
    view = Trajectory3DView()
    view.shutdown()
    pool = Mock()
    view._terrain_pool = pool
    view._terrain_src = (4326, (0, 0, 1, 1))
    monkeypatch.setattr(view, '_prepare_terrain', lambda *args: {})
    view._start_terrain_build()
    pool.start.assert_not_called()
    view.deleteLater()


def test_terrain_shutdown_drains_active_worker_before_native_deletion(tmp_path):
    pytest.importorskip('PyQt6')
    program = '''
import threading,time
from PyQt6 import sip
from PyQt6.QtCore import QRunnable,QThreadPool
from PyQt6.QtWidgets import QApplication
from ui.widgets.trajectory_3d_view import Trajectory3DView
app=QApplication([])
entered=threading.Event()
class SlowWorker(QRunnable):
 def run(self):
  entered.set()
  time.sleep(4)
view=Trajectory3DView()
view._terrain_pool=QThreadPool(view)
view._terrain_pool.start(SlowWorker())
assert entered.wait(2)
view.shutdown()
active=view._terrain_pool.activeThreadCount()
# Drain even on the old implementation, so the failed assertion exits safely.
assert view._terrain_pool.waitForDone(6000)
assert active==0, 'shutdown returned while a Python Qt worker was still active'
assert view._terrain_cancel.is_set()
sip.delete(view)
print('terrain-shutdown-drained',flush=True)
'''
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1',
               MYGPR_RUNTIME_ROOT=str(tmp_path / 'runtime'),
               LOCALAPPDATA=str(tmp_path / 'local'), APPDATA=str(tmp_path / 'roaming'))
    result = subprocess.run([sys.executable, '-B', '-X', 'faulthandler', '-c', program],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'terrain-shutdown-drained' in result.stdout
