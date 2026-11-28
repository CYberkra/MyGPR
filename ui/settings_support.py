"""Shared help and workspace commands for the desktop settings page."""

from __future__ import annotations

import math
import platform
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from PyQt6.QtCore import QT_VERSION_STR
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.constants import LOG_DIR
from ui.desktop_backend_facade import get_tile_cache_dir
from ui.page_scaffold import PanelStateMixin
from ui.settings_manager import DEFAULT_SETTINGS, SettingsManager
from ui.widgets.bscan_fullscreen import FullscreenHost
from ui.widgets.settings_controls import SettingsDisclosure


SETTINGS_CATEGORIES = {
    'appearance': '外观与布局',
    'bscan': 'B-scan 显示',
    'processing': '处理默认值',
    'spatial': '空间与底图',
    'storage': '存储与诊断',
    'about': '关于与更新',
}
_DEFERRED_SEARCH = {
    'appearance': (
        ('主题', '浅色 深色 外观 颜色'),
        ('恢复面板布局', '还原 默认 文件树 边栏 侧栏 窗口'),
        ('查看快捷键', 'F1 键盘'),
    ),
    'processing': (
        ('默认相对介电常数', '导入 εr'),
        ('并行工作线程数', '并发 重启'),
    ),
    'spatial': (('自动预下载测线区域底图', '地图 下载 联网'),),
    'storage': (
        ('默认项目根目录', '路径 新建 项目'),
        ('设置文件目录', '配置'),
        ('日志目录', '错误 log'),
        ('底图缓存目录', '瓦片 缓存'),
        ('诊断摘要', '排查 版本'),
    ),
    'about': (('使用手册', '帮助 文档'), ('应用更新', '检查更新 发布 安装版')),
}


@dataclass(frozen=True)
class SettingTarget:
    label: str
    category: str
    control: QWidget | None
    keywords: str = ''
    disclosure: SettingsDisclosure | None = None

    @property
    def caption(self) -> str:
        return f'{self.label}    {SETTINGS_CATEGORIES[self.category]}'


def finite_setting_value(value, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def deferred_setting_targets() -> dict[tuple[str, str], SettingTarget]:
    return {
        (category, label): SettingTarget(label, category, None, keywords)
        for category, entries in _DEFERRED_SEARCH.items()
        for label, keywords in entries
    }


def match_setting_targets(
    targets: dict[tuple[str, str], SettingTarget], words: list[str],
) -> list[SettingTarget]:
    return [
        target for category in SETTINGS_CATEGORIES
        for target in targets.values()
        if target.category == category and words
        and all(word in (target.caption + ' ' + target.keywords).casefold() for word in words)
    ]


def read_application_version() -> str:
    """Read the version from the checkout, or from installed package metadata."""
    try:
        version_file = Path(__file__).resolve().parents[1] / 'VERSION'
        text = version_file.read_text(encoding='utf-8-sig').strip()
        if text:
            return text
    except OSError:
        pass
    try:
        return metadata.version('mygpr')
    except metadata.PackageNotFoundError:
        return 'unknown'


# Widget-scoped bindings also appear when their pages have yet to be initialized.
_CONTEXT_SHORTCUTS = (
    ('运行待执行处理', 'Ctrl+R', '处理页'),
    ('关闭当前标签', 'Ctrl+W', 'B-scan 成果区'),
    ('恢复最近关闭的标签', 'Ctrl+Shift+T', 'B-scan 成果区'),
    ('切换到下一个标签', 'Ctrl+Tab', 'B-scan 成果区'),
    ('切换到上一个标签', 'Ctrl+Shift+Tab', 'B-scan 成果区'),
    ('向右拆分当前标签', 'Ctrl+Alt+Right', 'B-scan 成果区'),
    ('向下拆分当前标签', 'Ctrl+Alt+Down', 'B-scan 成果区'),
    ('切换全屏浏览', 'F11', 'B-scan 图面'),
    ('退出全屏浏览', 'Esc', 'B-scan 独立全屏'),
)


def _shortcut_rows(window: QWidget) -> list[tuple[str, str, str]]:
    rows = [(f'切换到{title}', sequence, '工作台')
            for sequence, _route, title in getattr(window, '_page_shortcuts', ())]
    bindings = (
        ('显示或收起文件树', '_file_tree_toggle', '工作台'),
        ('显示或收起输出面板', '_panel_toggle', '工作台'),
        ('打开任务列表', '_show_tasks_shortcut', '工作台'),
        ('打开快捷键清单', '_shortcuts_help', '工作台'),
    )
    for label, attr, scope in bindings:
        shortcut = getattr(window, attr, None)
        if shortcut is not None:
            rows.append((label, shortcut.key().toString(), scope))
    rows.extend(_CONTEXT_SHORTCUTS)
    return rows


def _add_close_button(dialog: QDialog, layout: QVBoxLayout) -> QDialogButtonBox:
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dialog)
    close = buttons.button(QDialogButtonBox.StandardButton.Close)
    close.setText('关闭')
    close.setAutoDefault(False)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    return buttons


def show_shortcuts(parent: QWidget) -> None:
    """Show the same searchable read-only shortcut list from settings and F1."""
    rows = _shortcut_rows(parent.window())
    dialog = QDialog(parent)
    dialog.setWindowTitle('快捷键')
    dialog.setObjectName('settingsShortcutDialog')
    dialog.resize(625, 460)
    layout = QVBoxLayout(dialog)
    search = QLineEdit(dialog)
    search.setObjectName('settingsShortcutSearch')
    search.setAccessibleName('搜索快捷键')
    search.setPlaceholderText('输入操作、按键或使用位置')
    search.setClearButtonEnabled(True)
    layout.addWidget(search)

    table = QTableWidget(len(rows), 3, dialog)
    table.setObjectName('settingsShortcutList')
    table.setAccessibleName('快捷键清单')
    table.setHorizontalHeaderLabels(['操作', '按键', '使用位置'])
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().hide()
    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
    for row, data in enumerate(rows):
        for column, value in enumerate(data):
            table.setItem(row, column, QTableWidgetItem(value))
    layout.addWidget(table)

    empty = QLabel('没有匹配的快捷键。', dialog)
    empty.hide()
    layout.addWidget(empty)
    _add_close_button(dialog, layout)

    def filter_rows(text: str) -> None:
        tokens = text.casefold().split()
        matched = False
        for row, item in enumerate(rows):
            searchable = ' '.join(item).casefold()
            visible = all(token in searchable for token in tokens)
            table.setRowHidden(row, not visible)
            matched |= visible
        empty.setVisible(not matched)

    search.textChanged.connect(filter_rows)
    search.setFocus()
    dialog.exec()


def settings_locations(manager: SettingsManager) -> dict[str, Path]:
    """Return the resolved directory targets without creating a cache."""
    return {
        'settings': Path(manager.settings_file).parent,
        'logs': Path(LOG_DIR),
        'tiles': Path(get_tile_cache_dir(create=False)),
    }


def show_diagnostic_summary(parent: QWidget, *, backend_ready: bool | None = None) -> None:
    """Preview only allowlisted runtime information and copy on explicit request."""
    app_version = read_application_version()
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.+!_-]{0,63}', app_version) is None:
        app_version = 'unknown'
    backend_status = {True: '已就绪', False: '未就绪', None: '暂不可用'}[backend_ready]
    mode = '打包版' if getattr(sys, 'frozen', False) else 'Python 运行'
    summary = '\n'.join((
        f'MyGPR  {app_version}',
        f'运行方式  {mode}',
        f'操作系统  {platform.system()} {platform.release()}',
        f'系统架构  {platform.machine()}',
        f'Qt  {QT_VERSION_STR}',
        f'Qt 窗口平台  {QGuiApplication.platformName()}',
        f'后端服务  {backend_status}',
    ))

    dialog = QDialog(parent)
    dialog.setWindowTitle('诊断摘要')
    dialog.setObjectName('settingsDiagnosticDialog')
    dialog.resize(490, 300)
    layout = QVBoxLayout(dialog)
    editor = QPlainTextEdit(summary, dialog)
    editor.setObjectName('settingsDiagnosticSummary')
    editor.setAccessibleName('诊断摘要内容')
    editor.setReadOnly(True)
    layout.addWidget(editor)
    copied = QLabel('已复制摘要。', dialog)
    copied.hide()
    layout.addWidget(copied)
    buttons = _add_close_button(dialog, layout)
    copy = buttons.addButton('复制摘要', QDialogButtonBox.ButtonRole.ActionRole)
    copy.setObjectName('settingsCopyDiagnosticSummary')
    copy.setAutoDefault(False)

    def copy_summary() -> None:
        QApplication.clipboard().setText(summary)
        copied.show()

    copy.clicked.connect(copy_summary)
    dialog.exec()


def restore_workspace_layout(window) -> bool:
    """Reset only the desktop layout keys and replay them after a successful save."""
    defaults = {
        'file_tree_page_states': {},
        'line_tree_page_states': {},
        'interpretation_right_collapsed': False,
        'spatial_left_collapsed': DEFAULT_SETTINGS['spatial_left_collapsed'],
        'spatial_right_collapsed': DEFAULT_SETTINGS['spatial_right_collapsed'],
        'output_panel_open': DEFAULT_SETTINGS['output_panel_open'],
        'output_panel_active_tab': DEFAULT_SETTINGS['output_panel_active_tab'],
        'output_panel_height': None,
        'bscan_fullscreen_geometry': DEFAULT_SETTINGS['bscan_fullscreen_geometry'],
    }
    if not window.settings.commit(defaults):
        return False

    if window._file_tree_panel is not None:
        window._file_tree_panel.restore_default_layout()
    for route in ('interpretationInterface', 'spatialInterface'):
        page = window._page(route)
        if isinstance(page, PanelStateMixin):
            page._auto_collapsed_sides.clear()
            page._restore_panel_state()
    if window.output_panel is not None:
        window.output_panel.restore_layout(
            defaults['output_panel_active_tab'], defaults['output_panel_open'])
        window._restore_output_panel_height()
    for host in window.findChildren(FullscreenHost):
        host.discard_pending_geometry()
    return True
