"""Late project reads must not replace lists/tracks for another project."""
from types import SimpleNamespace as NS
from threading import Thread

import pytest

pytest.importorskip('PyQt6')

from ui.controllers.project_controller import (
    ProjectController, _RefreshLinesCommand, _RefreshArtifactsCommand,
    _RefreshAllArtifactsCommand, _LoadSpatialTracksCommand,
)


@pytest.mark.parametrize('kind', ['lines', 'artifacts', 'all_artifacts', 'tracks'])
@pytest.mark.parametrize('replacement', ['B', None])
def test_late_refresh_after_project_switch_or_close_is_discarded(qapp, kind, replacement):
    controller = ProjectController()
    controller._current = NS(project_id='A')

    def finish_old_read(*args):
        controller._current = NS(project_id=replacement) if replacement else None
        return [NS(line_id='A_LINE')]

    controller.set_backend(NS(backend=NS(projects=NS(list_lines=finish_old_read, list_artifacts=finish_old_read), spatial=NS(load_tracks=finish_old_read))))
    received = []
    if kind == 'lines':
        controller.lines_updated.connect(received.append)
        command = _RefreshLinesCommand(controller, 'A')
    elif kind == 'artifacts':
        controller.artifacts_updated.connect(lambda line, items: received.append(items))
        command = _RefreshArtifactsCommand(controller, 'A', 'L01')
    elif kind == 'all_artifacts':
        controller.all_artifacts_updated.connect(received.append)
        command = _RefreshAllArtifactsCommand(controller, 'A')
    else:
        controller.spatial_tracks_ready.connect(received.append)
        command = _LoadSpatialTracksCommand(controller, 'A')
    command.execute()
    assert received == []


def test_current_project_refresh_remains_visible(qapp):
    controller = ProjectController()
    controller._current = NS(project_id='A')
    expected = [NS(line_id='L01')]
    controller.set_backend(NS(backend=NS(projects=NS(list_lines=lambda pid: expected))))
    received = []
    controller.lines_updated.connect(received.append)
    _RefreshLinesCommand(controller, 'A').execute()
    assert received == [expected]


def test_refresh_queued_before_project_switch_is_discarded_on_gui_delivery(qapp):
    controller = ProjectController()
    controller._current = NS(project_id='A')
    controller.set_backend(NS(backend=NS(projects=NS(list_lines=lambda pid: [NS(line_id='A_LINE')]))))
    received = []
    controller.lines_updated.connect(received.append)
    command = _RefreshLinesCommand(controller, 'A')
    worker = Thread(target=command.execute)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    # Worker already emitted while A was current, GUI has not delivered it yet.
    controller._current = NS(project_id='B')
    qapp.processEvents()
    assert received == []


def test_refresh_before_reopening_same_project_is_discarded(qapp, monkeypatch):
    from ui.controllers import project_controller as module
    controller = ProjectController()
    controller._current = NS(project_id='A')
    controller.set_backend(NS(backend=NS(projects=NS(list_lines=lambda pid: [NS(line_id='OLD')]))))
    old_command = _RefreshLinesCommand(controller, 'A')
    received = []
    controller.lines_updated.connect(received.append)
    monkeypatch.setattr(module, 'run_command', lambda *args, **kwargs: None)
    controller.open_project('A')
    old_command.execute()
    assert received == []


def test_line_preview_invalidations_do_not_discard_current_refresh(qapp):
    controller = ProjectController()
    controller._current = NS(project_id='A')
    expected = [NS(line_id='L01')]
    controller.set_backend(NS(backend=NS(projects=NS(list_lines=lambda pid: expected))))
    command = _RefreshLinesCommand(controller, 'A')
    received = []
    controller.lines_updated.connect(received.append)
    controller.invalidate_artifact_previews()
    command.execute()
    assert received == [expected]
