# -*- coding: utf-8 -*-
"""DockPanel — 统一坞面板基类（左坞/右坞共用的一套"面板语言"）。

解决的问题：此前文件树（面板头收起钮）、日志/任务（窗口边缘独立折叠长条）、
方法库（页面内固定卡片）是三套互不相干的面板习语，开关散落各处。本基类
把它们收敛为一种语言：

- **统一的头**：标题 + 右上角收起钮（唯一开关）；`set_header_widget()` 允许
  用自定义控件替换标题（右坞放"日志/任务"页签）；
- **统一的收起态**：18px 细条 + 竖排内容指示（子类经 ``strip_text()`` 提供，
  如当前线名/当前页签名）+ 展开钮；
- **统一的动画**：QVariantAnimation + OutCubic 220ms，走
  ``ui.motion.animations_enabled()`` 无障碍总闸；
- **开关唯一入口**：``set_collapsed()`` / ``toggle()``；子类可覆写
  ``_on_toggle_clicked()`` 加入状态持久化（如文件树的按页记忆）。

子类契约：内容加进 ``self.body_layout()``；需要感知收/放时覆写
``_on_view_state_changed()``。属性名 ``_expanded_view/_strip_view/
_strip_line_label/_collapsed`` 为既有测试约定，勿改名。
"""
from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, Qt, QSize, QVariantAnimation, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import TransparentToolButton
from qfluentwidgets import FluentIcon as FIF

from ui import constants
from ui.motion import animations_enabled
from ui.theme_helpers import accent_color, font_families_qss

# 标题字号（pt，与 constants 的五档口径一致）。历史上这里写 px，
# 使 SECTION(12) 实际渲染成 12px≈9pt，比设计意图小 25%。
_TITLE_FONT_SIZE = constants.FONT_SIZE_SECTION
_STRIP_FONT_SIZE = constants.FONT_SIZE_BODY


class DockPanel(QWidget):
    """统一坞面板：头 + 主体 + 细条收起态。"""

    collapsed_changed = pyqtSignal(bool)

    def __init__(self, title: str, expanded_width: int,
                 parent: QWidget = None) -> None:
        super().__init__(parent)
        self._title_text = str(title or '')
        self._expanded_width = int(expanded_width)
        self._collapsed = False
        self._width_anim: QVariantAnimation | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(4)

        # ---------------- 展开态视图（头 + 主体）
        self._expanded_view = QWidget(self)
        exp_layout = QVBoxLayout(self._expanded_view)
        exp_layout.setContentsMargins(0, 0, 0, 0)
        exp_layout.setSpacing(4)

        header = QWidget(self._expanded_view)
        self._head_layout = QHBoxLayout(header)
        self._head_layout.setContentsMargins(0, 0, 0, 0)
        self._head_layout.setSpacing(4)
        self._title_label = QLabel(self._title_text)
        self._title_label.setStyleSheet(
            f'font-family: {font_families_qss()}; '
            f'font-size: {_TITLE_FONT_SIZE}pt; font-weight: bold;')
        self._head_layout.addWidget(self._title_label)
        self._head_layout.addStretch(1)
        self._header_widget: QWidget | None = None
        # 透明图标钮：与底部输出面板（日志/任务工具条）同一美术语言——
        # 平时无底色无边框，悬停才显底；14px 图标与全软件工具钮一致
        self._collapse_btn = TransparentToolButton(FIF.LEFT_ARROW, header)
        self._collapse_btn.setIconSize(QSize(*constants.TOOL_BTN_ICON))
        self._collapse_btn.setToolTip('收起为细条（点击细条可展开）')
        self._collapse_btn.setFixedSize(24, 24)
        self._collapse_btn.clicked.connect(self._on_toggle_clicked)
        self._head_layout.addWidget(self._collapse_btn)
        exp_layout.addWidget(header)

        self._body = QWidget(self._expanded_view)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(4)
        exp_layout.addWidget(self._body, 1)

        # ---------------- 细条态视图（展开钮 + 竖排指示）
        self._strip_view = QWidget(self)
        strip_layout = QVBoxLayout(self._strip_view)
        strip_layout.setContentsMargins(0, 0, 0, 4)
        strip_layout.setSpacing(2)
        self._expand_btn = TransparentToolButton(FIF.CHEVRON_RIGHT, self._strip_view)
        self._expand_btn.setIconSize(QSize(*constants.TOOL_BTN_ICON))
        self._expand_btn.setToolTip('展开面板')
        self._expand_btn.setFixedSize(18, 24)
        self._expand_btn.clicked.connect(self._on_toggle_clicked)
        strip_layout.addWidget(self._expand_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        self._strip_line_label = QLabel('')
        self._strip_line_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        strip_layout.addWidget(self._strip_line_label, 0,
                               Qt.AlignmentFlag.AlignHCenter)
        strip_layout.addStretch(1)

        layout.addWidget(self._expanded_view, 1)
        layout.addWidget(self._strip_view, 1)
        self._strip_view.hide()
        self._strip_line_label.hide()
        self.setFixedWidth(self._expanded_width)
        # 只设自身控件的初始配色，**不**调 self.apply_theme()：子类会在
        # apply_theme 里触碰自己尚未构造的属性（如 FileTreePanel 的
        # _project_label），基类构造期调用会 AttributeError。子类负责在
        # __init__ 完成后自行 apply_theme。
        self._apply_own_theme()

    # ------------------------------------------------------------ 主题
    def _apply_own_theme(self) -> None:
        """刷本基类自有控件的配色（不含子类扩展）。"""
        self._title_label.setStyleSheet(
            f'font-family: {font_families_qss()}; '
            f'font-size: {_TITLE_FONT_SIZE}pt; font-weight: bold;')
        self._strip_line_label.setStyleSheet(
            f'font-family: {font_families_qss()}; '
            f'font-size: {_STRIP_FONT_SIZE}pt; color: {accent_color()};')

    def apply_theme(self, dark: bool) -> None:
        """标题与细条指示文字随主题重刷（主窗口 findChildren 遍历调用）。

        细条文字用 :func:`accent_color` 而非写死色：强调色在深色主题下切到
        高亮度变体（``#2dd4bf``），写死浅色主题的 ``#0F6E56`` 在深底上只有
        约 3.2:1，低于 WCAG AA 的 4.5:1。

        子类覆写时应调用 ``super().apply_theme(dark)`` 并追加自身逻辑。
        """
        self._dark = bool(dark)
        self._apply_own_theme()

    # ------------------------------------------------------------ 布局接口
    def body_layout(self) -> QVBoxLayout:
        """主体布局：子类把内容加进这里。"""
        return self._body_layout

    def set_header_widget(self, widget: QWidget) -> None:
        """用自定义控件替换头部标题（如右坞的"日志/任务"页签）。"""
        if self._header_widget is not None:
            self._head_layout.removeWidget(self._header_widget)
            self._header_widget.hide()
        self._header_widget = widget
        self._title_label.hide()
        self._head_layout.insertWidget(1, widget)

    # ------------------------------------------------------------ 子类钩子
    def strip_text(self) -> str:
        """细条竖排指示文字（子类覆写：当前线名/当前页签名…）。"""
        return self._title_text

    def _on_view_state_changed(self) -> None:
        """收/放后视图微调（子类覆写：如文件树的树/空态切换）。"""

    def _on_toggle_clicked(self) -> None:
        """开关点击入口（子类覆写以加入持久化，最后调用 toggle()）。"""
        self.toggle()

    # ------------------------------------------------------------ 公共 API
    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed, animate=True)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool, animate: bool = True) -> None:
        """切换展开/细条；animate=False 用于切页/启动等瞬时场景。"""
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            self._apply_view_state()
            return
        self._collapsed = collapsed
        # 先换视图（不动宽度），再决定宽度是瞬时还是动画过渡
        self._apply_view_state(sync_width=False)
        target_w = (constants.DOCK_COLLAPSED_WIDTH if collapsed
                    else self._expanded_width)
        if animate and animations_enabled():
            self._animate_width(self.width(), target_w)
        else:
            self.setFixedWidth(target_w)
        self.collapsed_changed.emit(collapsed)

    # ------------------------------------------------------------ 内部
    def _apply_view_state(self, sync_width: bool = True) -> None:
        """按（collapsed）刷新两套视图；sync_width=False 供动画过渡场景。"""
        self._expanded_view.setVisible(not self._collapsed)
        self._strip_view.setVisible(self._collapsed)
        if self._collapsed:
            self._update_strip_text()
            self._strip_line_label.setVisible(True)
            if sync_width:
                self.setFixedWidth(constants.DOCK_COLLAPSED_WIDTH)
        else:
            self._strip_line_label.setVisible(False)
            if sync_width:
                self.setFixedWidth(self._expanded_width)
        self._on_view_state_changed()

    def _update_strip_text(self) -> None:
        text = self.strip_text()[:6]
        self._strip_line_label.setText('\n'.join(text) if text else '—')

    def _animate_width(self, start_w: int, target_w: int) -> None:
        if self._width_anim is not None:
            self._width_anim.stop()
        anim = QVariantAnimation(self)
        anim.setStartValue(float(start_w))
        anim.setEndValue(float(target_w))
        anim.setDuration(constants.PANEL_ANIM_DURATION_MS)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(
            lambda v: self.setFixedWidth(int(round(float(v)))))
        self._width_anim = anim
        anim.start()
