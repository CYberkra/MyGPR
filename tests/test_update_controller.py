"""Updater callbacks cross into Qt through cancellable, generation-bound requests."""
from datetime import datetime, timezone
from types import SimpleNamespace
import threading
import time

import pytest

pytest.importorskip('PyQt6')
pytest.importorskip('qfluentwidgets')
from PyQt6.QtWidgets import QWidget
from ui.controllers.update_controller import UpdateController
from ui.update_binding import UpdateBinding
from ui.widgets.update_panel import UpdatePanel


class Client:
    def start(self, **kwargs):
        self.callbacks = kwargs
    def check(self, **kwargs):
        self.silent = kwargs['silent']
    def cleanup(self):
        self.stopped = True


def controller(prepare=lambda: (True, ''), close=lambda: True):
    frozen, statuses = [], []
    result = UpdateController(prepare_close=prepare, freeze=frozen.append, close_window=close)
    result.status_changed.connect(statuses.append)
    client = Client()
    result.start(client, SimpleNamespace(public_key='public', feed=lambda channel: 'https://example.com'), '0.12.0', 'stable')
    frozen.clear()
    return result, client, frozen, statuses


def pump(qapp, thread):
    deadline = time.monotonic() + 2
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        thread.join(0.001)
    assert not thread.is_alive()
    qapp.processEvents()


def test_success_uses_gui_thread_and_close_cancellation_thaws(qapp):
    seen = []
    ctrl, client, frozen, statuses = controller(prepare=lambda: (seen.append(threading.get_ident()) or True, ''), close=lambda: False)
    values = []
    worker = threading.Thread(target=lambda: values.append(client.callbacks['can_shutdown']()))
    worker.start()
    pump(qapp, worker)
    assert values == [1] and seen == [threading.get_ident()] and frozen == [True]
    client.callbacks['shutdown']()
    assert frozen == [True, False]
    assert '安装器' in statuses[-1]
    ctrl.stop()


@pytest.mark.parametrize('reason', ['任务运行中', '标注未保存', '设置保存失败'])
def test_refused_install_does_not_freeze(qapp, reason):
    ctrl, client, frozen, statuses = controller(prepare=lambda: (False, reason))
    values = []
    thread = threading.Thread(target=lambda: values.append(client.callbacks['can_shutdown']()))
    thread.start()
    pump(qapp, thread)
    assert values == [0] and frozen == [] and statuses[-1] == reason
    ctrl.stop()


def test_timeout_late_gui_delivery_does_not_freeze(qapp):
    ctrl, _, frozen, statuses = controller()
    values = []
    thread = threading.Thread(target=lambda: values.append(ctrl.request_shutdown(0.01)))
    thread.start()
    thread.join(1)  # Do not process Qt events until AFTER the timeout.
    assert values == [0]
    qapp.processEvents()
    assert frozen == [] and '稍后' in statuses[-1]
    ctrl.stop()


def test_timeout_during_slow_save_does_not_freeze(qapp):
    ctrl, _, frozen, _ = controller(prepare=lambda: (time.sleep(0.05) or True, ''))
    values = []
    thread = threading.Thread(target=lambda: values.append(ctrl.request_shutdown(0.01)))
    thread.start()
    pump(qapp, thread)
    assert values == [0] and frozen == []
    ctrl.stop()


def test_stop_wakes_waiter_before_native_cleanup_and_stale_callbacks_are_ignored(qapp):
    ctrl, client, frozen, statuses = controller()
    queued = threading.Event()
    ctrl._probe.connect(lambda request: queued.set())
    values = []
    thread = threading.Thread(target=lambda: values.append(ctrl.request_shutdown()))
    thread.start()
    # A queued probe exists even though the GUI event loop has not handled it.
    deadline = time.monotonic() + 1
    while not ctrl._requests and time.monotonic() < deadline:
        time.sleep(0.001)
    ctrl.stop()
    thread.join(1)
    assert values == [0] and client.stopped
    old = client.callbacks
    new = Client()
    ctrl.start(new, SimpleNamespace(public_key='public', feed=lambda channel: 'https://example.com'), '0.12.0', 'stable')
    old['event']('available')
    old['shutdown']()
    qapp.processEvents()
    assert frozen == [] and statuses[-1] == ''
    assert old['can_shutdown']() == 0
    ctrl.stop()


@pytest.mark.parametrize('event', ['cancelled', 'timeout'])
def test_failure_restores_ui_and_allows_retry(qapp, event):
    ctrl, client, frozen, _ = controller()
    worker = threading.Thread(target=client.callbacks['can_shutdown'])
    worker.start()
    pump(qapp, worker)
    assert ctrl.check() and ctrl.busy
    assert not ctrl.check()
    client.callbacks['event']('available')
    client.callbacks['event'](event)
    assert not ctrl.busy and frozen == [True, False]
    assert ctrl.check(silent=True) and client.silent
    ctrl.stop()


def test_dismissing_check_does_not_allow_overlapping_workers_or_unlock_install(qapp):
    ctrl, client, frozen, _ = controller()
    assert ctrl.check()
    client.callbacks['event']('dismissed')
    assert ctrl.busy and not ctrl.check()
    client.callbacks['event']('error')
    assert not ctrl.busy and ctrl.check()
    client.callbacks['event']('available')
    worker = threading.Thread(target=client.callbacks['can_shutdown'])
    worker.start()
    pump(qapp, worker)
    client.callbacks['event']('error')  # An unrelated late checker cannot revoke launch approval.
    assert frozen == [True] and ctrl.busy
    client.callbacks['event']('cancelled')
    assert frozen == [True, False] and not ctrl.busy
    ctrl.stop()


class Settings:
    def __init__(self, **values):
        self.values = values
        self.saved = True
    def get(self, name, default=None):
        return self.values.get(name, default)
    def set(self, name, value):
        self.values[name] = value
    def save(self):
        return self.saved


def window_state():
    return SimpleNamespace(settings=Settings(), _backend_ready=True, _pages_warmed=True,
        interpretation_controller=SimpleNamespace(has_unsaved_changes=lambda: False),
        page_coordinator=SimpleNamespace(project=SimpleNamespace(_busy_sources=set(), _active_job_ids=set()),
            processing=SimpleNamespace(processing_job_id=''),
            jobs=SimpleNamespace(import_job_ids=set(), spatial_job_ids=set())))


@pytest.mark.parametrize('busy', ['processing', 'import', 'spatial', 'busy', 'active', 'dirty', 'settings', 'warmup'])
def test_live_install_gate_covers_submission_races(busy):
    window = window_state()
    co = window.page_coordinator
    if busy == 'processing':
        co.processing.processing_job_id = 'job'
    elif busy == 'import':
        co.jobs.import_job_ids.add('job')
    elif busy == 'spatial':
        co.jobs.spatial_job_ids.add('job')
    elif busy == 'busy':
        co.project._busy_sources.add('delivery')
    elif busy == 'active':
        co.project._active_job_ids.add('job')
    elif busy == 'dirty':
        window.interpretation_controller.has_unsaved_changes = lambda: True
    elif busy == 'settings':
        window.settings.saved = False
    elif busy == 'warmup':
        window._pages_warmed = False
    assert UpdateBinding.prepare_close(SimpleNamespace(window=window))[0] is False
    assert UpdateBinding.prepare_close(SimpleNamespace(window=window_state())) == (True, '')


def test_source_environment_and_daily_schedule(qapp):
    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.settings = Settings()
            self.panel = UpdatePanel(self)
        def _page(self, name):
            return SimpleNamespace(update_panel=self.panel)
    window = Window()
    binding = UpdateBinding(window)
    binding._timer.stop()
    assert not window.panel.check_button.isEnabled()
    assert window.panel.releases_button.isEnabled()
    checks = []
    binding.controller.check = lambda **kw: checks.append(kw) or True
    binding.automatic_check()
    assert checks == [{'silent': True}]
    binding.automatic_check()
    assert len(checks) == 1
    window.settings.set('update_last_check', datetime.now(timezone.utc).isoformat())
    window.settings.set('update_automatic', False)
    binding.automatic_check()
    assert len(checks) == 1
    binding.controller.stop()
    window.deleteLater()


def test_channel_preferences_defer_native_reconfiguration_until_next_process(qapp, monkeypatch, tmp_path):
    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.settings = Settings(update_channel='stable', update_automatic=True)
            self.panel = UpdatePanel(self)
        def _page(self, name):
            return SimpleNamespace(update_panel=self.panel)
    monkeypatch.setattr('ui.update_binding.sys', SimpleNamespace(platform='win32', frozen=True, _MEIPASS=str(tmp_path)))
    configuration = SimpleNamespace(public_key='pub', preview_feed='https://example.com/preview',
                                    feed=lambda channel: 'https://example.com/' + channel)
    monkeypatch.setattr('ui.update_binding.windows_update_configuration', lambda path: configuration)
    clients = []
    monkeypatch.setattr('ui.update_binding.windows_update_client', lambda path: clients.append(Client()) or clients[-1])
    window = Window()
    binding = UpdateBinding(window)
    binding._timer.stop()
    assert len(clients) == 1
    window.panel.automatic.setChecked(False)
    assert len(clients) == 1 and not hasattr(clients[0], 'stopped')
    window.panel.channel.setCurrentIndex(1)
    assert window.settings.get('update_channel') == 'preview'
    assert len(clients) == 1 and binding._channel_pending
    assert not window.panel.check_button.isEnabled()
    clients[0].callbacks['event']('current')
    assert '重启' in window.panel.status.text() and not window.panel.check_button.isEnabled()
    with pytest.raises(RuntimeError):
        binding.configure()
    binding.controller.stop()
    clients[0].callbacks['event']('available')  # delayed native completion after UI cleanup
    assert '重启' in window.panel.status.text()
    window.deleteLater()
