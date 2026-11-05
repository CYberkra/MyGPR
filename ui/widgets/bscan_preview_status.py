"""Loading and retry controls for a B-scan preview group."""
import weakref

from qfluentwidgets import ProgressBar, PushButton


class _PreviewStatus:
    def __init__(self, group):
        self._group = weakref.ref(group)
        view = group.view
        self.progress = ProgressBar(view)
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(2)
        self.progress.hide()
        self.retry = PushButton('重试预览', view)
        self.retry.setAccessibleName('重试预览')
        self.retry.hide()
        self.retry.clicked.connect(self.retry_current)
        self.layout()

    @property
    def group(self):
        return self._group()

    def retry_current(self):
        group = self.group
        key = group.current_key() if group is not None else ''
        if key:
            group.workspace.preview_retry_requested.emit(key)

    def set_state(self, error, loading):
        self.progress.setVisible(bool(loading))
        self.retry.setVisible(bool(error))
        self.layout()

    def layout(self):
        group = self.group
        if group is None:
            return
        view = group.view
        self.progress.setGeometry(0, 0, view.width(), 2)
        self.retry.adjustSize()
        self.retry.move(max(8, (view.width() - self.retry.width()) // 2),
                        max(60, int(view.height() * 0.62)))
        self.progress.raise_()
        self.retry.raise_()
