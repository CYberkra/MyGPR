from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QPushButton, QToolButton

from ui.field_workbench_window import FieldWorkbenchWindow

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if isinstance(app, QApplication):
        _APP = app
        return app
    _APP = QApplication([])
    return _APP


def test_project_level_actions_move_to_header_project_switcher() -> None:
    app = _app()
    window = FieldWorkbenchWindow("MyGPR 勘探定位工作台")
    app.processEvents()

    button = window.findChild(QToolButton, "projectSelectorButton")
    assert button is not None
    assert "当前项目" in button.text()
    assert button.menu() is not None
    menu_texts = [action.text() for action in button.menu().actions() if action.text()]
    for expected in ["新建项目", "打开项目", "项目设置", "项目备份", "删除项目…"]:
        assert expected in menu_texts
    window.close()


def test_project_page_right_side_keeps_only_high_frequency_actions() -> None:
    app = _app()
    window = FieldWorkbenchWindow("MyGPR 勘探定位工作台")
    window.switch_workspace("data_management")
    app.processEvents()

    visible_button_texts = [button.text().replace("\n", "") for button in window.findChildren(QPushButton)]
    assert any("导入测线" in text for text in visible_button_texts)
    assert any("批量导入" in text for text in visible_button_texts)
    assert any("导入RTK/IMU" in text for text in visible_button_texts)
    assert any("运行质检" in text for text in visible_button_texts)
    assert any("数据运维" in text for text in visible_button_texts)
    assert window.recent_project_combo is None
    for low_frequency in ["检查源文件", "重新定位", "来源清单", "删除项目", "移除最近"]:
        assert not any(low_frequency in text for text in visible_button_texts)
    window.close()


def test_project_tree_uses_context_menu_for_line_level_operations() -> None:
    app = _app()
    window = FieldWorkbenchWindow("MyGPR 勘探定位工作台")
    window.switch_workspace("data_management")
    app.processEvents()

    tree = window.project_tree_widget
    assert tree is not None
    assert tree.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    line_root = tree.topLevelItem(0)
    assert line_root is not None
    line_item = line_root.child(0)
    assert line_item is not None
    assert line_item.data(0, Qt.ItemDataRole.UserRole)
    window.close()
