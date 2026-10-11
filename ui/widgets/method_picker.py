# -*- coding: utf-8 -*-
"""MethodPickerDialog — 非模态方法库选择弹窗（P0-2，2026-10-11）。

「+」号与「添加所选方法」的统一入口：顶部搜索 + 分类算法树（内嵌复用
MethodBrowser 的数据形态，``set_methods(list[dict])`` 同源灌入）。双击
条目 → ``sig_method_chosen(method_id)`` 且**保持打开**（连续添加，PRD
Q1 定案）；Esc / 点击外部关闭；再开状态复位（搜索词清空）。

窗口形态沿用 TVG 对话框先例（QDialog + setModal(False) +
Qt.WindowType.Tool，不进任务栏、随主窗），内联于处理页（parent = 页面）。

外部点击关闭判定：focusOutEvent 里查 ``QApplication.focusWidget()`` 是否
本窗后代——窗内控件互相切换（搜索框 → 树）焦点仍是本窗后代，不误关；
点到窗外则 200ms 复核一次再关（真机焦点移交有一拍延迟）。
"""
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QWidget

from ui.widgets.method_browser import MethodBrowser

# 外部点击关闭的复核延迟（ms）：焦点移交到外部控件有一拍延迟，
# 立即判定会把「点开 colorbox 子菜单」这类窗内操作误判为失焦
_FOCUS_RECHECK_MS = 200


class MethodPickerDialog(QDialog):
    """方法库选择弹窗：双击条目发 sig_method_chosen，不关窗连续添加。"""

    sig_method_chosen = pyqtSignal(str)   # method_id（双击条目）

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('添加处理步骤')
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Tool)      # 不进任务栏，随主窗
        self.setMinimumSize(360, 420)

        self._browser = MethodBrowser(self)
        self._browser.sig_add_requested.connect(self._on_add_requested)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._browser, 1)

    # ------------------------------------------------------------ 数据
    def set_methods(self, methods: list) -> None:
        """与方法库控件同源的方法清单（结构见 ProcessingPage.set_methods）。"""
        self._browser.set_methods(methods or [])

    # ------------------------------------------------------------ 复位
    def reset_state(self) -> None:
        """搜索词清空 + 选中复位：树经 _apply_filter('') 回到全展开全可见。"""
        self._browser._search.clear()
        self._browser._tree.setCurrentItem(None)

    # ------------------------------------------------------------ 信号
    def _on_add_requested(self, method_id: str) -> None:
        """双击条目：转发选中方法，**不关窗**（连续添加，PRD Q1 定案）。"""
        if method_id:
            self.sig_method_chosen.emit(str(method_id))

    # ------------------------------------------------------------ 事件
    def showEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        """每次显示复位状态：关闭再开 = 全新浏览态（PRD 验收④）。"""
        self.reset_state()
        super().showEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        # 窗内控件切换时 focusWidget 仍是本窗后代（不误关）；
        # 点到窗外 → 复核一次再关（真机下焦点移交有一拍延迟）
        QTimer.singleShot(_FOCUS_RECHECK_MS, self._close_if_focus_left)
        super().focusOutEvent(event)

    def _close_if_focus_left(self) -> None:
        """复核焦点是否已落到本窗之外；是则关闭。

        复核由 singleShot 延迟触发：窗可能已被宿主销毁（teardown 竞态），
        届时 sip 包装器失效——静默丢弃，不得把 RuntimeError 漏进测试/运行期。
        """
        try:
            widget = QApplication.focusWidget()
            while widget is not None:
                if widget is self:
                    return
                widget = widget.parentWidget()
            self.close()
        except RuntimeError:
            return

    # ------------------------------------------------------------ 主题
    def apply_theme(self, dark: bool) -> None:
        """主题切换钩子（主窗 findChildren 鸭子类型遍历自动调用）。

        必须幂等（纯令牌重刷）：遍历可能命中隐藏弹窗。内嵌 MethodBrowser
        的 apply_theme 当前为空操作，透传以保持单一路径。
        """
        self._browser.apply_theme(dark)
