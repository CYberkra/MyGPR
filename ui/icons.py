"""工作台操作图标；复用 Fluent 绘制和设置页的主题切换。"""
from enum import Enum
from pathlib import Path

from qfluentwidgets import FluentIcon, FluentIconBase, Theme, isDarkTheme

from ui.design_tokens import color

_ICON_ROOT = Path(__file__).resolve().parent / 'resources' / 'icons'


class WorkbenchIcon(FluentIconBase, Enum):
    UNDO = 'undo'
    REDO = 'redo'
    DIRECTORY_TREE = 'directory-tree'
    TASK_LIST = 'task-list'
    WAVEFORM_FILE = 'waveform-file'
    RESULT_CHART = 'result-chart'

    def path(self, theme=Theme.AUTO) -> str:
        dark = isDarkTheme() if theme == Theme.AUTO else theme == Theme.DARK
        return str(_ICON_ROOT / ('dark' if dark else 'light') / f'{self.value}.svg')


# 按钮绘制时自动取主题色；菜单每次打开时取当前危险色。
RESULT_DELETE = FluentIcon.DELETE.colored(color('error', False), color('error', True))


def result_delete_icon():
    return FluentIcon.DELETE.icon(color=color('error'))
