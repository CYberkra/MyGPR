"""Cheap directory probes; no project contents are loaded on the GUI thread."""
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR
from threading import Event
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from ui.controllers.backend_controller import run_command


def _directory_error(path: str) -> str:
    try:
        return '' if S_ISDIR(Path(path).stat().st_mode) else '不是项目目录'
    except FileNotFoundError:
        return '目录不存在'
    except PermissionError:
        return '无权访问该目录'
    except (OSError, ValueError) as exc:
        return f'目录不可访问：{exc}'


@dataclass
class _ProbeDirectoriesCommand:
    generation: int
    paths: tuple[str, ...]
    cancelled: Event
    result: Callable
    finished: Callable

    def execute(self) -> None:
        try:
            for path in self.paths:
                if self.cancelled.is_set():
                    break
                error = _directory_error(path)
                if not self.cancelled.is_set():
                    self.result(self.generation, path, error)
        except RuntimeError:
            # Qt may already have destroyed the owner during application exit.
            pass
        finally:
            try:
                self.finished(self.generation)
            except RuntimeError:
                pass


class RecentProjectsController(QObject):
    """One active probe and one latest pending request, even on stalled drives.

    These advisory stat calls are not backend jobs. Signals bridge the daemon
    command to the GUI thread; formal project opening retains its JobBridge.
    """

    availability_changed = pyqtSignal(str, str)
    _result = pyqtSignal(int, str, str)
    _finished = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._generation = 0
        self._active = None
        self._pending = None
        self._result.connect(self._accept_result)
        self._finished.connect(self._finish)

    def check(self, paths) -> None:
        self.cancel()
        paths = tuple(dict.fromkeys(str(path) for path in paths if path))
        if paths:
            self._pending = _ProbeDirectoriesCommand(
                self._generation, paths, Event(), self._result.emit, self._finished.emit)
            self._start_pending()

    def cancel(self) -> None:
        self._generation += 1
        self._pending = None
        if self._active is not None:
            self._active.cancelled.set()

    def _start_pending(self) -> None:
        if self._active is None and self._pending is not None:
            self._active, self._pending = self._pending, None
            run_command(self._active, name='mygpr-recent-directories')

    def _accept_result(self, generation: int, path: str, error: str) -> None:
        if generation == self._generation:
            self.availability_changed.emit(path, error)

    def _finish(self, generation: int) -> None:
        if self._active is not None and self._active.generation == generation:
            self._active = None
            self._start_pending()
