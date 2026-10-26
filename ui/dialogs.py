# -*- coding: utf-8 -*-
"""统一的对话框创建模块（架构纪律唯一许可点之一）。

qfluentwidgets 的 MessageBox / Dialog 只能在页面层或本模块创建；
ui/coordinator_* 接线器与 ui/controllers 均不得创建 QWidget，
需要阻塞式用户确认时调用本模块的函数式 API。
"""

from __future__ import annotations

from pathlib import Path
import re

from PyQt6.QtWidgets import QWidget
from qfluentwidgets import MessageBox

__all__ = ["ask_artifact_delete", "ask_discard_interpretation", "choose_processing_package_path"]


def ask_display_name(parent, title, current, *, subject=""):
    from qfluentwidgets import MessageBoxBase, SubtitleLabel, BodyLabel, LineEdit
    from ui.desktop_backend_facade import validate_display_name

    box = MessageBoxBase(parent.window())
    box.widget.setMinimumWidth(380)
    box.viewLayout.addWidget(SubtitleLabel(title, box))
    if subject:
        box.viewLayout.addWidget(BodyLabel(subject, box))
    edit = LineEdit(box)
    edit.setAccessibleName("名称")
    edit.setMaxLength(80)
    edit.setText(current)
    box.viewLayout.addWidget(edit)
    hint = BodyLabel("仅修改显示名称，编号、文件和已有成果保持不变。", box)
    hint.setWordWrap(True)
    box.viewLayout.addWidget(hint)
    box.yesButton.setText("保存")
    box.cancelButton.setText("取消")

    def validate(text):
        try:
            valid = validate_display_name(text) != str(current).strip()
        except ValueError:
            valid = False
        box.yesButton.setEnabled(valid)

    edit.textChanged.connect(validate)
    validate(current)
    edit.selectAll()
    edit.setFocus()
    return edit.text().strip() if box.exec() == 1 else None


def ask_project_destination(parent, root):
    from qfluentwidgets import MessageBoxBase, SubtitleLabel, BodyLabel, LineEdit, PushButton
    from PyQt6.QtWidgets import QHBoxLayout
    from ui import file_dialogs

    box = MessageBoxBase(parent.window())
    box.widget.setMinimumWidth(460)
    box.viewLayout.addWidget(SubtitleLabel("迁移项目目录", box))
    current = BodyLabel(f"当前位置：{root}", box)
    current.setWordWrap(True)
    box.viewLayout.addWidget(current)
    box.viewLayout.addWidget(BodyLabel("目标项目目录（新目录）", box))
    row = QHBoxLayout()
    edit = LineEdit(box)
    edit.setAccessibleName("目标项目目录")
    edit.setText(str(Path(root).parent / (Path(root).name + "_迁移")))
    browse = PushButton("浏览…", box)
    row.addWidget(edit, 1)
    row.addWidget(browse)
    box.viewLayout.addLayout(row)

    def choose():
        folder = file_dialogs.getExistingDirectory(parent, "选择目标父目录", str(Path(root).parent))
        if folder:
            edit.setText(str(Path(folder) / Path(root).name))

    browse.clicked.connect(choose)
    hint = BodyLabel("测线、成果、标注和项目内附件将一并复制并校验。成功后打开新位置，旧目录保留。", box)
    hint.setWordWrap(True)
    box.viewLayout.addWidget(hint)
    box.yesButton.setText("开始迁移")
    box.cancelButton.setText("取消")
    edit.textChanged.connect(lambda text: box.yesButton.setEnabled(bool(text.strip())))
    return edit.text().strip() if box.exec() == 1 else None


def show_project_transfer(parent, cancel):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QProgressDialog

    dialog = QProgressDialog("正在检查并迁移项目…", "取消迁移", 0, 100, parent.window())
    dialog.setWindowTitle("迁移项目目录")
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.canceled.connect(cancel)
    dialog.show()
    return dialog


def choose_processing_package_path(
    parent: QWidget | None, line_id: str, *, artifact_id: str = "", directory: str = ""
) -> str:
    """One native Save As dialog; export publishes with a suffix on collisions."""
    from ui import file_dialogs

    safe_line = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", line_id).strip(" .") or "测线"
    name = f"{safe_line}_处理结果" + ("_到此步" if artifact_id else "") + ".zip"
    folder = Path(directory).expanduser() if directory else Path.home() / "Documents"
    if not folder.is_dir():
        folder = Path.home()
    path, _ = file_dialogs.getSaveFileName(
        parent,
        "打包处理结果（同名文件将自动编号保留）",
        str(folder / name),
        "ZIP 压缩包 (*.zip)",
        confirm_overwrite=False,
    )
    return path


def choose_spatial_export_path(parent: QWidget | None, result_id: str) -> tuple[str, str]:
    from ui import file_dialogs

    filters = (
        "空间成果包 ZIP (*.zip);;GeoJSON (*.geojson);;CSV (*.csv);;"
        "KML (*.kml);;GeoPackage (*.gpkg);;Shapefile (*.shp)"
    )
    path, selected = file_dialogs.getSaveFileName(
        parent, "导出空间成果", str(Path.home() / "Documents" / f"{result_id}.zip"),
        filters, confirm_overwrite=False,
    )
    if not path:
        return "", ""
    suffix = Path(path).suffix.lower().lstrip('.')
    if suffix not in {'zip', 'geojson', 'csv', 'kml', 'gpkg', 'shp'}:
        suffix = next((fmt for fmt in ('geojson', 'csv', 'kml', 'gpkg', 'shp', 'zip')
                       if fmt in selected.lower()), 'zip')
    return path, suffix


def ask_spatial_delete(parent: QWidget | None, result_id: str) -> bool:
    box = MessageBox(
        "移除空间成果",
        f"将把空间成果 {result_id} 移入项目回收站，不影响测线、处理成果和解释标注。确认继续？",
        parent.window() if parent is not None else None,
    )
    box.yesButton.setText("移入回收站")
    box.cancelButton.setText("取消")
    return box.exec() == 1


def ask_discard_interpretation(parent: QWidget | None, line_id: str) -> bool:
    """Discard only after explicit consent; cancelling keeps the edit session."""
    box = MessageBox(
        "标注尚未保存",
        f"测线 {line_id or '当前测线'} 有未保存的标注修改。继续将关闭当前会话并放弃这些修改。\n\n"
        "如需保留，请取消并先在解释页保存标注。",
        parent.window() if parent is not None else None,
    )
    box.yesButton.setText("放弃修改并继续")
    box.cancelButton.setText("返回保存")
    box.cancelButton.setFocus()
    return box.exec() == 1


def ask_artifact_delete(parent: QWidget | None, names: list[str]) -> bool:
    """成果删除级联确认框：列出将删除的成果名（含级联派生成果）。

    数据会移入项目 .trash 回收站（可恢复）。返回 True = 用户确认删除；
    名单为空时直接返回 False（不弹框）。
    """
    names = [str(name) for name in (names or []) if str(name)]
    if not names:
        return False
    shown = "\n".join(f"  • {name}" for name in names)
    box = MessageBox(
        "确认删除成果？",
        f"将删除 {len(names)} 个成果（含级联派生成果，数据会移入项目 .trash 回收站，可恢复）：\n{shown}\n\n确认继续？",
        parent.window() if parent is not None else None,
    )
    box.yesButton.setText("删除")
    box.cancelButton.setText("取消")
    return box.exec() == 1  # QDialog.DialogCode.Accepted
