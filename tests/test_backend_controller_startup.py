"""Backend composition loads in the init worker, with visible failures."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
from textwrap import dedent
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("qfluentwidgets")

from ui.controllers import backend_controller as module


@pytest.mark.parametrize("entry", ["ui.controllers.backend_controller", "app_qt"])
def test_gui_import_does_not_load_backend_composition(entry, tmp_path):
    env = os.environ.copy()
    env.update(
        QT_QPA_PLATFORM="offscreen",
        MYGPR_RUNTIME_ROOT=str(tmp_path),
        MYGPR_LOG_DIR=str(tmp_path / "logs"),
        MYGPR_SETTINGS_FILE=str(tmp_path / "settings.json"),
    )
    result = subprocess.run(
        [sys.executable, "-c", (
            "import importlib, sys; importlib.import_module(sys.argv[1]); "
            "assert 'mygpr.interfaces.backend' not in sys.modules, "
            "'GUI import eagerly loaded backend composition'"
        ), entry],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _worker_outcomes(worker):
    outcomes = SimpleNamespace(succeeded=[], failed=[])
    worker.succeeded.connect(outcomes.succeeded.append)
    worker.failed.connect(outcomes.failed.append)
    return outcomes


def _install_backend_module(monkeypatch, factory):
    backend_module = ModuleType("mygpr.interfaces.backend")
    backend_module.MyGPRBackend = SimpleNamespace(create_default=factory)
    monkeypatch.setitem(sys.modules, "mygpr.interfaces.backend", backend_module)


def test_init_worker_imports_backend_and_emits_created_instance(monkeypatch):
    backend = object()
    factory = Mock(return_value=backend)
    _install_backend_module(monkeypatch, factory)
    worker = module._BackendInitWorker(max_workers=3)
    outcomes = _worker_outcomes(worker)

    worker.run()

    factory.assert_called_once_with(max_workers=3)
    assert outcomes.succeeded == [backend]
    assert outcomes.failed == []


def test_init_worker_surfaces_backend_import_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "mygpr.interfaces.backend", None)
    worker = module._BackendInitWorker()
    outcomes = _worker_outcomes(worker)

    worker.run()

    assert outcomes.succeeded == []
    assert len(outcomes.failed) == 1
    assert "mygpr.interfaces.backend" in outcomes.failed[0]


def test_init_worker_surfaces_backend_construction_failure(monkeypatch):
    factory = Mock(side_effect=RuntimeError("backend initialization failed"))
    _install_backend_module(monkeypatch, factory)
    worker = module._BackendInitWorker()
    outcomes = _worker_outcomes(worker)

    worker.run()

    assert outcomes.succeeded == []
    assert outcomes.failed == ["backend initialization failed"]


@pytest.mark.parametrize("init_fails", [False, True])
def test_shutdown_waits_for_init_and_does_not_publish_late_outcome(
        qapp, monkeypatch, init_fails):
    entered, release = threading.Event(), threading.Event()
    backend = SimpleNamespace(shutdown=Mock())

    def create_default(**kwargs):
        entered.set()
        assert release.wait(timeout=5)
        if init_fails:
            raise RuntimeError("late initialization failure")
        return backend

    _install_backend_module(monkeypatch, create_default)
    controller = module.BackendController()
    ready, failed = [], []
    controller.backend_ready.connect(lambda: ready.append(True))
    controller.backend_failed.connect(failed.append)
    controller.start()
    init_thread = controller._thread
    releaser = threading.Timer(0.05, release.set)
    try:
        assert entered.wait(timeout=5)
        releaser.start()
        controller.shutdown()

        assert not init_thread.isRunning()
        if init_fails:
            backend.shutdown.assert_not_called()
        else:
            backend.shutdown.assert_called_once_with(wait=True)
        qapp.processEvents()
        assert ready == []
        assert failed == []
        assert not controller.is_ready
        controller.shutdown()
        if not init_fails:
            backend.shutdown.assert_called_once_with(wait=True)
    finally:
        release.set()
        if releaser.is_alive():
            releaser.join(timeout=5)
        init_thread.quit()
        assert init_thread.wait(5000)
        controller.shutdown()


@pytest.mark.parametrize("shutdown_error", [None, RuntimeError("close failed")])
def test_shutdown_is_idempotent_after_backend_ready(qapp, shutdown_error):
    backend = SimpleNamespace(shutdown=Mock(side_effect=shutdown_error))
    controller = module.BackendController()
    controller._on_backend_created(backend)
    bridge = controller.job_bridge
    assert controller.is_ready

    controller.shutdown()
    controller.shutdown()
    controller.start()

    backend.shutdown.assert_called_once_with(wait=True)
    assert bridge._shutdown.is_set()
    assert not controller.is_ready
    assert controller._thread is None


def test_shutdown_immediately_after_start_exits_init_thread(tmp_path):
    env = os.environ.copy()
    env.update(
        QT_QPA_PLATFORM="offscreen",
        MYGPR_RUNTIME_ROOT=str(tmp_path),
        MYGPR_LOG_DIR=str(tmp_path / "logs"),
        MYGPR_SETTINGS_FILE=str(tmp_path / "settings.json"),
    )
    code = dedent("""
        import sys
        from types import ModuleType, SimpleNamespace
        from unittest.mock import Mock
        from PyQt6.QtCore import QCoreApplication
        from ui.controllers.backend_controller import BackendController

        created = []
        def create_default(**kwargs):
            backend = SimpleNamespace(shutdown=Mock())
            created.append(backend)
            return backend

        stub = ModuleType('mygpr.interfaces.backend')
        stub.MyGPRBackend = SimpleNamespace(create_default=create_default)
        sys.modules['mygpr.interfaces.backend'] = stub
        app = QCoreApplication([])
        for _ in range(8):
            controller = BackendController()
            ready = []
            controller.backend_ready.connect(lambda: ready.append(True))
            controller.start()
            thread = controller._thread
            controller.shutdown()
            assert not thread.isRunning()
            app.processEvents()
            assert not controller.is_ready
            assert ready == []
            controller.shutdown()
        for backend in created:
            backend.shutdown.assert_called_once_with(wait=True)
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
