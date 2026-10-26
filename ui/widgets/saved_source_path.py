"""Read-only saved-input navigation; pending algorithm order lives elsewhere."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget
from qfluentwidgets import CaptionLabel, TransparentPushButton, isDarkTheme

from ui.design_tokens import tokens
from ui.widgets.context_menus import add_action, make_menu
from ui.widgets.fluent_menu_keyboard import FluentMenuKeyboard


class _PathButton(TransparentPushButton):
    """Keep the beginning of long source names readable in constrained rows."""

    def __init__(self, title, parent):
        self._full_title = str(title)
        super().__init__(parent=parent)
        self.setText(self._full_title)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setText(self.fontMetrics().elidedText(
            self._full_title, Qt.TextElideMode.ElideRight, max(0, self.width() - 24)))

    def sizeHint(self):
        size = super().sizeHint()
        size.setWidth(min(160, self.fontMetrics().horizontalAdvance(self._full_title) + 24))
        return size


class SavedSourcePath(QWidget):
    source_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signature = None
        self._generation = 0
        self._buttons = []
        self._menu = None
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(3)

    def buttons(self):
        return list(self._buttons)

    def set_path(self, entries, context):
        signature = (tuple(context), tuple(tuple(entry) for entry in entries))
        if signature == self._signature:
            return
        self._signature = signature
        self._generation += 1
        generation = self._generation
        if self._menu is not None:
            self._menu.close()
            self._menu = None
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        self._buttons = []
        self._row.addWidget(CaptionLabel('已完成来源' if entries else '未选择输入', self))
        visible = entries if len(entries) <= 4 else [entries[0], None, *entries[-3:]]
        for index, entry in enumerate(visible):
            if index:
                self._row.addWidget(CaptionLabel('›', self))
            if entry is None:
                button = TransparentPushButton(f'… {len(entries) - 4} 项', self)
                button.clicked.connect(lambda _checked=False, items=tuple(entries), gen=generation:
                                       self._show_all(items, gen))
            else:
                key, title, _revision = entry[:3]
                details = entry[3] if len(entry) > 3 else ''
                button = _PathButton(title, self)
                button.setProperty('source_key', key)
                tooltip = str(title) + (f'\n{details}' if details else '')
                button.setToolTip(tooltip + ('\n已保存结果；点击查看，不重放算法' if key else
                    '\n执行记录；该步未单独保存，或来源不完整，不能打开中间结果'))
                button.setEnabled(bool(key))
                button.clicked.connect(lambda _checked=False, key=key, gen=generation:
                                       self._open(key, gen))
            button.setMaximumWidth(160)
            button.setMinimumWidth(0)
            self._buttons.append(button)
            self._row.addWidget(button)
        self.apply_theme(isDarkTheme())

    def apply_theme(self, dark):
        t = tokens(dark)
        for label in self.findChildren(CaptionLabel):
            label.setStyleSheet(f'color: {t["text_muted"]};')
        for button in self._buttons:
            button.setFixedHeight(26)
            button.setStyleSheet(
                f'QPushButton {{ color: {t["text_secondary"]}; background: transparent;'
                ' border: none; border-radius: 4px; padding: 2px 8px; }'
                f'QPushButton:hover {{ background: {t["bg_muted"]}; }}'
                f'QPushButton:focus {{ border: 1px solid {t["border_focus"]}; }}'
                f'QPushButton:disabled {{ color: {t["warning"]}; }}')

    def _open(self, key, generation):
        if generation == self._generation and key:
            self.source_requested.emit(str(key))

    def _show_all(self, entries, generation):
        if generation != self._generation:
            return
        menu = make_menu(self)
        self._menu = menu
        for entry in entries:
            key, title, _revision = entry[:3]
            add_action(menu, None, str(title),
                       lambda _checked=False, key=key: self._open(key, generation), enabled=bool(key))
        menu._keyboard = FluentMenuKeyboard(menu)
        menu.closedSignal.connect(lambda: setattr(self, '_menu', None))
        menu.exec(self.mapToGlobal(self.rect().bottomLeft()))
        menu._keyboard.focus_menu(menu)
