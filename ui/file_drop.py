"""External single-file drops reuse the normal import job and backend preflight."""
from pathlib import Path
from PyQt6.QtCore import QObject, QEvent

DATA_SUFFIXES = {'.csv', '.txt', '.npy', '.npz', '.h5', '.hdf5'}

def dropped_data_file(mime):
    if not mime.hasUrls() or len(mime.urls()) != 1:
        return None
    url = mime.urls()[0]
    if not url.isLocalFile():
        return None
    path = Path(url.toLocalFile())
    try:
        return str(path) if path.suffix.lower() in DATA_SUFFIXES and path.is_file() else None
    except OSError:
        return None

class FileDropFilter(QObject):
    """Catch external files over child viewports without intercepting internal drags."""
    def __init__(self, window, callback):
        super().__init__(window)
        self.window = window
        self.callback = callback

    def eventFilter(self, obj, event):
        if event.type() not in (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.Drop):
            return False
        if not obj.isWidgetType() or obj.window() is not self.window:
            return False
        path = dropped_data_file(event.mimeData())
        if path is None:
            return False
        if event.type() == QEvent.Type.Drop:
            self.callback(path)
        event.acceptProposedAction()
        return True


def import_data_file(page, source: str) -> bool:
    """Submit through the page's existing signal; readers perform backend preflight."""
    from PyQt6.QtCore import QMimeData, QUrl
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(source)])
    if not page._can_write() or dropped_data_file(mime) is None:
        return False
    page.file_edit.setText(source)
    page.line_id_edit.setText(page._suggest_line_id())
    page.line_name_edit.setText(Path(source).stem)
    page._emit_import_request(preflight=False)
    return True
