"""One owner per animated value; immediate updates cancel previous motion."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtCore import QCoreApplication, QEvent, QVariantAnimation, Qt
from PyQt6.QtWidgets import QLabel, QProgressBar
from qfluentwidgets import ProgressBar

from ui import motion


@pytest.fixture(autouse=True)
def drain_motion_deletions(qapp):
    yield
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_windows_animation_preference_uses_full_bool_buffer(monkeypatch):
    sizes = []

    def get_preference(_action, _param, pointer, _flags):
        sizes.append(ctypes.sizeof(pointer._obj))
        pointer._obj.value = 1
        return 1

    monkeypatch.setattr(ctypes, 'windll', SimpleNamespace(user32=SimpleNamespace(
        SystemParametersInfoW=get_preference)), raising=False)
    assert motion.animations_enabled() is True
    assert sizes == [ctypes.sizeof(wintypes.BOOL)]


@pytest.mark.parametrize('immediate', ['same_value', 'reduced_motion'])
def test_immediate_progress_update_cancels_old_animation(qapp, monkeypatch, immediate):
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    motion.animate_progress(bar, 100)
    animation = bar._motion_anim
    animation.setCurrentTime(70)
    target = bar.value() if immediate == 'same_value' else 20
    if immediate == 'reduced_motion':
        monkeypatch.setattr(motion, 'animations_enabled', lambda: False)
    motion.animate_progress(bar, target)
    assert animation.state() is QVariantAnimation.State.Stopped
    assert bar.value() == target
    bar.deleteLater()


@pytest.mark.parametrize('immediate', ['same_color', 'reduced_motion'])
def test_immediate_badge_update_cancels_old_animation(qapp, monkeypatch, immediate):
    badge = QLabel()
    template = 'background-color:%s;'
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    motion.animate_badge_color(badge, template, '#000000', '#ffffff')
    animation = badge._motion_color_anim
    animation.setCurrentTime(70)
    if immediate == 'reduced_motion':
        monkeypatch.setattr(motion, 'animations_enabled', lambda: False)
    motion.animate_badge_color(badge, template,
                              '#112233' if immediate == 'same_color' else '#000000', '#112233')
    assert animation.state() is QVariantAnimation.State.Stopped
    assert badge.styleSheet() == template % '#112233'
    badge.deleteLater()


def test_badge_reuses_one_animation_across_status_updates(qapp, monkeypatch):
    badge = QLabel()
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    for index in range(12):
        motion.animate_badge_color(badge, 'background-color:%s;', '#000000',
                                  '#ffffff' if index % 2 else '#112233')
    assert len(badge.findChildren(QVariantAnimation)) == 1
    badge._motion_color_anim.setCurrentTime(motion.DEFAULT_DURATION_MS)
    assert badge.styleSheet() == 'background-color:#ffffff;'
    badge.deleteLater()


def test_fluent_progress_has_one_animation_owner(qapp, monkeypatch):
    bar = ProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    motion.animate_progress(bar, 100)
    bar._motion_anim.setCurrentTime(70)
    assert bar.ani.state() is QVariantAnimation.State.Stopped
    assert bar.val == bar.value()
    bar.deleteLater()


def test_fluent_progress_reduced_motion_updates_rendered_value_immediately(qapp, monkeypatch):
    bar = ProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    monkeypatch.setattr(motion, 'animations_enabled', lambda: False)
    motion.animate_progress(bar, 100)
    assert bar.value() == 100
    assert bar.val == 100
    assert bar.ani.state() is QVariantAnimation.State.Stopped
    bar.deleteLater()


def test_badge_theme_refresh_interrupts_previous_status_color(qapp, monkeypatch):
    from ui.widgets import job_widgets
    badge = QLabel()
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    motion.animate_badge_color(badge, job_widgets.BADGE_QSS, '#000000', '#ffffff')
    animation = badge._motion_color_anim
    monkeypatch.setattr(job_widgets, '_status_badge_color', lambda _status: '#123456')
    job_widgets._restyle_status_badge(badge, 'running')
    assert animation.state() is QVariantAnimation.State.Stopped
    assert badge.styleSheet() == job_widgets.BADGE_QSS % '#123456'
    badge.deleteLater()


@pytest.mark.parametrize('reduced', [False, True])
def test_segment_retarget_preserves_visual_position_or_snaps_when_reduced(qapp, monkeypatch, reduced):
    from ui.widgets.segment_tabs import SlimSegment
    segment = SlimSegment()
    segment.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    for key in ('a', 'b', 'c'):
        segment.addItem(key, key * 8)
    segment.resize(500, 40)
    segment.show()
    qapp.processEvents()
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    segment.setCurrentItem('a')
    segment.setCurrentItem('b')
    segment.slideAni.setCurrentTime(30)
    position = segment.slideAni.value()
    assert 0 < position < segment.widget('b').x()
    if reduced:
        monkeypatch.setattr(motion, 'animations_enabled', lambda: False)
    segment.setCurrentItem('c')
    if reduced:
        assert segment.slideAni.state() is QVariantAnimation.State.Stopped
        assert segment.slideAni.value() == segment.widget('c').x()
    else:
        assert segment.slideAni.startValue() == position
        segment.slideAni.setCurrentTime(segment.slideAni.duration())
        assert segment.slideAni.value() == segment.widget('c').x()
    assert segment.currentRouteKey() == 'c'
    segment.close()
    segment.deleteLater()


def test_chain_selection_reduced_motion_stops_inflight_pill(qapp, monkeypatch):
    from ui.widgets.chain_strip import ChainStrip
    strip = ChainStrip()
    strip.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    strip.resize(700, 40)
    strip.set_steps([{'label': 'first'}, {'label': 'second'}, {'label': 'third'}])
    strip.show()
    qapp.processEvents()
    monkeypatch.setattr(motion, 'animations_enabled', lambda: True)
    strip.select_step(0)
    strip.select_step(1)
    strip._pill_anim.setCurrentTime(40)
    monkeypatch.setattr(motion, 'animations_enabled', lambda: False)
    strip.select_step(2)
    assert strip._pill_anim.state() is QVariantAnimation.State.Stopped
    assert strip._pill.geometry() == strip._list.visualItemRect(strip._list.item(2))
    assert strip._list.currentRow() == 2
    strip.close()
    strip.deleteLater()


def test_animated_widgets_delete_with_gc_and_exit_normally(tmp_path):
    code = '''
import atexit, gc
from PyQt6.QtCore import QCoreApplication, QEvent, QObject, QTimer, pyqtSlot
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QProgressBar
from qfluentwidgets import ProgressBar
from ui import motion
atexit.register(lambda: print("motion-lifetime-atexit", flush=True))
app = QApplication([])
app.setQuitOnLastWindowClosed(False)
motion.animations_enabled = lambda: True
host = QWidget()
host.show()
orphans = []
class Observer(QObject):
    def __init__(self):
        super().__init__()
        self.destroyed_count = 0
    @pyqtSlot()
    def on_destroyed(self):
        self.destroyed_count += 1
observer = Observer()
for parent in (host, None):
    for kind in ("plain", "fluent", "badge"):
        widget = QLabel(parent) if kind == "badge" else ProgressBar(parent) if kind == "fluent" else QProgressBar(parent)
        widget.destroyed.connect(observer.on_destroyed)
        if kind == "badge":
            motion.animate_badge_color(widget, "background-color:%s;", "#000000", "#ffffff")
        else:
            widget.setRange(0, 100)
            widget.setValue(0)
            motion.animate_progress(widget, 100)
        if parent is None:
            orphans.append(widget)
del widget
def dispose():
    host.close()
    host.deleteLater()
    for orphan in orphans:
        orphan.deleteLater()
    orphans.clear()
    gc.collect()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
QTimer.singleShot(20, dispose)
QTimer.singleShot(300, app.quit)
status = app.exec()
assert observer.destroyed_count == 6, observer.destroyed_count
print("motion-lifetime-return", status, flush=True)
raise SystemExit(status)
'''
    env = os.environ.copy()
    env.update(QT_QPA_PLATFORM='offscreen', MYGPR_RUNTIME_ROOT=str(tmp_path / 'runtime'))
    result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', code],
                            cwd=Path(__file__).parents[1], env=env, capture_output=True,
                            text=True, encoding='utf-8', timeout=20)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert 'motion-lifetime-return 0' in output
    assert 'motion-lifetime-atexit' in output
    assert not any(term in output.lower() for term in ('fatal exception', 'fatal python error', 'access violation'))
