"""External line-reference drops reuse the workspace's existing edge layout."""
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QWidget


class BScanLineDrop(QObject):
    def __init__(self, workspace, owner):
        super().__init__(workspace)
        self.workspace, self.owner = workspace, owner
        self.targets = {}
        workspace.view_created.connect(self.install)
        for view in workspace.views():
            self.install(view)

    def install(self, view):
        group = view.parentWidget()
        for widget in (group, *group.findChildren(QWidget)):
            widget.setAcceptDrops(True)
            widget.installEventFilter(self)
            self.targets[id(widget)] = group.group_id
            widget.destroyed.connect(lambda _=None, key=id(widget): self.targets.pop(key, None))

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.DragLeave:
            self.hide_hints()
            return False
        if kind not in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            return False
        line_id = self.owner.read_mime(event.mimeData())
        group = self.workspace._group(self.targets.get(id(watched)))
        if not line_id or group is None:
            return False
        edge, index = self.workspace._drop_target(group, watched.mapTo(group, event.position().toPoint()))
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        if kind == QEvent.Type.Drop:
            self.hide_hints()
            self.owner.open_line(line_id, group.group_id, edge, index)
        else:
            self.workspace._show_hint(group, edge, index)
            if edge != 'tabs':
                group.hint.setText(f'{self.owner.label(line_id)} · ' +
                    ('添加对比标签' if edge == 'center' else '在高亮区域分屏对比'))
        return True

    def hide_hints(self):
        for group in self.workspace._groups.values():
            group.hint.hide()
