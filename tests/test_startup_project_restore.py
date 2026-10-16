"""Opening the last project must wait for both backend and page wiring."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip('PyQt6')

from ui.main_window import MyGPRMainWindow


@pytest.mark.parametrize('backend_ready,pages_warmed', [(True, False), (False, True)])
def test_restore_waits_for_both_ready_events(tmp_path, backend_ready, pages_warmed):
    controller = SimpleNamespace(open_project=Mock())
    state = SimpleNamespace(
        _backend_ready=backend_ready, _pages_warmed=pages_warmed,
        project_controller=controller,
        settings={'recent_projects': [str(tmp_path)]})
    MyGPRMainWindow._restore_last_project(state)
    controller.open_project.assert_not_called()
    state._backend_ready = True
    state._pages_warmed = True
    MyGPRMainWindow._restore_last_project(state)
    MyGPRMainWindow._restore_last_project(state)
    controller.open_project.assert_called_once_with(str(tmp_path))
