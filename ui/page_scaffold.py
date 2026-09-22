# -*- coding: utf-8 -*-
"""页面脚手架（UI 收敛轮）：卡片 / 透明滚动栏 / 表单行 / 下拉与勾选列表填充 /
折叠面板状态持久化。

把 8 个页面里逐字复制的小工厂收敛到这一处纯 Qt 帮助函数（无业务逻辑）：
``make_card``/``card_title``（原 6 份 _create_card/_make_card）、
``make_segment_card``（子标签卡片：标题与 SlimSegment 同行 header）、
``make_scroll_column``（原 2 份 + 散落的透明 QScrollArea QSS）、
``make_form_row``（标签 minWidth=100 + 控件 + stretch 约 30 处）、
``refill_combo``（下拉重建-保持选择，userData 驱动）、
``rebuild_check_list``（勾选列表重建-保持勾选，spatial/delivery 同模式）、
``PanelStateMixin``（processing/spatial 两页折叠状态三件套 + 窄窗自动折叠）。

各页面公开信号/方法签名不变；本模块只做"组装"，不做业务。
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QHBoxLayout, QListWidgetItem, QVBoxLayout, QWidget)
from qfluentwidgets import CaptionLabel, CardWidget, ScrollArea, SubtitleLabel

from ui import constants
from ui.theme_helpers import hint_qss, ui_font

__all__ = [
    'make_card', 'make_segment_card', 'card_title', 'style_transparent_scroll',
    'make_scroll_column', 'wrap_centered', 'make_form_row',
    'HintLabel', 'make_hint',
    'refill_combo', 'rebuild_check_list', 'PanelStateMixin',
]


# ---------------------------------------------------------------- 卡片
def card_title(text: str) -> SubtitleLabel:
    """卡片标题：SubtitleLabel 微软雅黑 10pt Bold（SPEC §1）。

    收敛自 delivery/processing/interpretation/spatial 四份逐字重复的
    私有 ``_card_title``。
    """
    label = SubtitleLabel(text)
    label.setFont(ui_font(constants.FONT_SIZE_BODY, QFont.Weight.Bold))
    return label


def make_card(title: str, *, parent=None, header_action: QWidget | None = None) -> tuple:
    """卡片范式（SPEC §1）：CardWidget + QVBoxLayout(margins=15, spacing=10)，
    首行 10pt Bold 卡片标题。返回 ``(card, layout)``。

    收敛自 home/settings/project 的 ``_create_card`` 与
    processing/spatial 的 ``_make_card``、delivery 的方法版
    ``_make_card``（其 ``CardWidget(self)`` 父参数经 ``parent=`` 传入）。

    ``header_action`` 可选：主操作控件进卡头行右侧（「页签卡 header 放
    动作」模式，P2-3），不再在卡体里孤悬一行；控件可先以页面为父创建，
    加入 header 时 Qt 自动重挂父。
    """
    card = CardWidget(parent)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*constants.CARD_MARGINS)
    layout.setSpacing(constants.CARD_SPACING)
    if header_action is None:
        layout.addWidget(card_title(title))
    else:
        header = QHBoxLayout()
        header.addWidget(card_title(title), 0, Qt.AlignmentFlag.AlignVCenter)
        header.addStretch(1)
        header.addWidget(header_action, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(header)
    return card, layout


def make_segment_card(title: str, segment: QWidget, *, parent=None) -> tuple:
    """子标签卡片范式：单行 header（标题居左 + 瘦页签居右）+ 内容区。

    即 ``make_card`` 的 ``header_action=segment`` 特例（Win11 设置页惯例），
    不再「标题一行 + 页签一行」双 header 浪费中栏垂直空间。返回值与
    :func:`make_card` 同构 ``(card, layout)``。
    """
    return make_card(title, parent=parent, header_action=segment)


# ---------------------------------------------------------------- 透明滚动栏
def style_transparent_scroll(scroll, *, resizable: bool = True) -> None:
    """透明无边框滚动区样式统一入口（整页/面板内 ScrollArea 的散落 QSS）。

    与 :func:`make_scroll_column`（固定宽栏）互补：本函数只设样式，
    不建内容容器（各页自建 container/layout，边距各异）。
    """
    if resizable:
        scroll.setWidgetResizable(True)
    scroll.setStyleSheet(
        'QScrollArea { background-color: transparent; border: none; }')


def make_scroll_column(width: int, *, parent=None,
                       object_name: str = 'pageScrollContent') -> tuple:
    """固定宽透明滚动栏：ScrollArea(固定 width，透明、横向滚动条恒关) +
    内容 widget(固定 width-16，透明) + 内容 QVBoxLayout(0 边距, spacing=15)。
    返回 ``(scroll_area, content_layout)``。

    收敛自 processing/spatial 两份 ``_make_scroll_column``，并统一各页
    散落的 ``QScrollArea{background-color:transparent;border:none;}`` QSS。
    """
    scroll = ScrollArea(parent)
    scroll.setFixedWidth(width)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setStyleSheet(
        'QScrollArea { background-color: transparent; border: none; }')
    content = QWidget(scroll)
    content.setFixedWidth(width - 16)
    content.setObjectName(object_name)
    content.setStyleSheet(
        f'QWidget#{object_name} {{ background-color: transparent; }}')
    layout = QVBoxLayout(content)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(constants.PAGE_SPACING)
    scroll.setWidget(content)
    return scroll, layout


def wrap_centered(content: QWidget,
                  max_width: int = constants.FORM_COLUMN_MAX_WIDTH) -> QWidget:
    """限宽居中列：把内容 widget 包一层，宽屏下居中且不超过 ``max_width``。

    用于设置/交付等单列卡片页——卡片全宽拉满时表单左对齐、右侧留出
    大片空白；限宽居中后各窗宽下阅读节奏一致（网页设置页惯例）。
    """
    content.setMaximumWidth(max_width)
    wrapper = QWidget()
    wrapper.setStyleSheet('background-color: transparent;')
    row = QHBoxLayout(wrapper)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    row.addStretch(1)
    row.addWidget(content)
    row.addStretch(1)
    return wrapper


# ---------------------------------------------------------------- 表单行
def make_form_row(label_text: str, field, *extra,
                  label_min_width: int = constants.FORM_LABEL_MIN_WIDTH,
                  parent=None, trailing_stretch: bool = True) -> QHBoxLayout:
    """表单行：CaptionLabel(minWidth=FORM_LABEL_MIN_WIDTH) + 控件(stretch)
    + 扩展控件 + stretch。

    收敛 settings/project/delivery/home 等页逐字重复的
    「标签(minWidth)+控件+stretch」样板。返回行布局（供 addLayout）。

    标签列统一最小宽（112，容纳六汉字+冒号）：跨卡片/跨页面的值列
    起点对齐（原各处 minWidth=100 散值）。

    :param extra: 行尾附加控件（如「浏览」按钮、提示标签）；
    :param trailing_stretch: 行尾是否加 stretch（以按钮收尾的行传 False）。
    """
    row = QHBoxLayout()
    label = CaptionLabel(label_text, parent)
    label.setMinimumWidth(label_min_width)
    row.addWidget(label)
    row.addWidget(field, 1)
    for widget in extra:
        row.addWidget(widget)
    if trailing_stretch:
        row.addStretch(1)
    return row


# ---------------------------------------------------------------- 提示文字
class HintLabel(CaptionLabel):
    """辅助说明标签（hint）：secondary 色 + SECONDARY 字号，WCAG AA 达标。

    颜色语义经 :func:`ui.theme_helpers.hint_qss` 随主题查表；主窗口主题
    切换遍历（findChildren + apply_theme 鸭子类型）自动重刷，页面无需
    在 apply_theme 里手动重设样式（spatial 等页的手工重刷随之删除）。

    运行期需要改语义色（如预检结果 success/error）时调
    :meth:`set_hint_key`，勿再手拼 QSS。
    """

    def __init__(self, text: str = '', parent=None, key: str = 'secondary'):
        # 走 FluentLabelBase 单参 (parent) 分支再 setText：其 (text, parent)
        # 分支内部会 self.__init__(parent) 重入子类构造（递归直到 TypeError）
        super().__init__(parent)
        self.setText(str(text))
        self._hint_key = str(key)
        self.setStyleSheet(hint_qss(self._hint_key))

    def set_hint_key(self, key: str) -> None:
        """切换语义色键（secondary/success/warning/error/info…）并重刷。"""
        self._hint_key = str(key)
        self.setStyleSheet(hint_qss(self._hint_key))

    def hint_key(self) -> str:
        return self._hint_key

    def apply_theme(self, dark: bool) -> None:
        """主题切换重刷（主窗口全量遍历鸭子类型调用）。"""
        self.setStyleSheet(hint_qss(self._hint_key))


def make_hint(text: str, *, parent=None, key: str = 'secondary') -> HintLabel:
    """辅助说明文字工厂（原 6 页逐字重复的 'color:%s;font-size:11px' 样板）。

    :param key: 语义色键，默认 ``'secondary'``（AA 达标灰）；
        语义状态提示传 success/warning/error/info。
    """
    return HintLabel(text, parent, key)


# ---------------------------------------------------------------- 下拉/勾选列表
def refill_combo(combo, items, text_fn, data_fn=None, *,
                 previous_data=None, prepend=()) -> None:
    """blockSignals 下重建下拉项，并按 ``previous_data`` 找回选择。

    收敛 processing ``set_lines``/``set_artifacts``、interpretation
    ``set_artifacts`` 的「清空-重填-保持选择」簿记——qfluentwidgets 1.8+
    ComboBox 完整支持 userData（addItem(text, userData)/currentData()/
    findData()），平行索引列表簿记因此删除。

    :param text_fn: item → 显示文本；:param data_fn: item → userData
        （返回 None 时该项不传 userData）；
    :param previous_data: 重填前选择（None 表示重填后回退首项；
        不传则自动取重填前的 ``currentData()``）；
    :param prepend: 先填的 ``(text, data)`` 占位项（如 ('原始数据', '')）；
    :return: 无。找回失败一律回退索引 0——各调用方把"最新/默认"项排最前，
        与原各页「默认最新一条 / 原始数据」语义一致。
    """
    if previous_data is None:
        previous_data = combo.currentData()
    combo.blockSignals(True)
    try:
        combo.clear()
        for text, data in prepend:
            combo.addItem(str(text), userData=data)
        for item in items or []:
            text = str(text_fn(item))
            data = data_fn(item) if data_fn is not None else None
            if data is None:
                combo.addItem(text)
            else:
                combo.addItem(text, userData=data)
        index = combo.findData(previous_data) \
            if previous_data not in (None, '') else -1
        if index < 0 and combo.count():
            index = 0
        if index >= 0:
            combo.setCurrentIndex(index)
    finally:
        combo.blockSignals(False)


def rebuild_check_list(widget, items, *, key_fn, text_fn,
                       default_checked: bool = False, icon_fn=None) -> None:
    """重建勾选列表并保持已有勾选状态（spatial 测线列表 / delivery
    测线多选逐字同模式的收敛实现）。

    :param key_fn: item → UserRole 键；:param text_fn: item → 显示文本；
    :param default_checked: 新出现条目的默认勾选态；
    :param icon_fn: item → QIcon 或 None（spatial 的颜色块图标）。
    """
    previous_checked = {}
    for row in range(widget.count()):
        item = widget.item(row)
        previous_checked[str(item.data(Qt.ItemDataRole.UserRole) or '')] = (
            item.checkState() == Qt.CheckState.Checked)
    widget.blockSignals(True)
    try:
        widget.clear()
        for entry in items or []:
            key = str(key_fn(entry) or '')
            if not key:
                continue
            icon = icon_fn(entry) if icon_fn is not None else None
            if icon is None:
                item = QListWidgetItem(str(text_fn(entry)))
            else:
                item = QListWidgetItem(icon, str(text_fn(entry)))
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = previous_checked.get(key, default_checked)
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            widget.addItem(item)
    finally:
        widget.blockSignals(False)


# ---------------------------------------------------------------- 折叠面板状态
# 窄窗自动折叠：中栏（左右面板之间）解析宽度低于该阈值时，自动折叠仍展开的
# 侧栏（优先保中栏可视面积）；窗口恢复到容纳宽度时，仅自动展开"本次自动
# 折叠过"的侧栏——用户手动折叠 / 持久化恢复的侧栏保持尊重用户选择。
_MIDDLE_MIN_PX = 360


class PanelStateMixin:
    """左右折叠面板状态持久化 + 窄窗自动折叠 mixin（processing/spatial）。

    约定子类：
    - 属性 ``_sm``：共享 SettingsManager（主窗口注入，可 None）；
    - 属性 ``_left_panel`` / ``_right_panel``：CollapsiblePanel；
    - 类常量 ``_PANEL_STATE_PREFIX``：SettingsManager 键前缀
      （如 ``'processing'`` → 键 ``processing_left_collapsed``）。

    子类在 ``_connect_internal`` 中把两面板 ``sig_collapsed`` 接到
    ``_on_side_panel_collapsed``；``set_settings_manager`` 注入后调用
    ``self._restore_panel_state()``。

    自动折叠策略（resizeEvent 驱动，解析式算宽度，几何滞后无关）：
    - 中栏可用宽度 = 页宽 − 页边距 − 栏间距×2 − 两侧栏当前占位宽度和；
    - 低于 :data:`_MIDDLE_MIN_PX` 时自动折叠展开中的侧栏（记住是哪些）；
    - 恢复宽度后仅展开记住的侧栏；手动折叠/展开会清除自动痕迹。
    """

    _PANEL_STATE_PREFIX = ''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_collapsed_sides: set[str] = set()
        self._in_auto_panel_change = False

    def panel_states(self) -> dict:
        """当前左右面板折叠状态。"""
        return {
            'left': self._left_panel.is_collapsed(),
            'right': self._right_panel.is_collapsed(),
        }

    def set_panel_collapsed(self, *, left: bool = None, right: bool = None,
                            animate: bool = True) -> None:
        """设置左右面板折叠状态。"""
        if left is not None:
            self._left_panel.set_collapsed(bool(left), animate=animate)
        if right is not None:
            self._right_panel.set_collapsed(bool(right), animate=animate)

    # ------------------------------------------------------------ 窄窗自动折叠
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._auto_collapse_side_panels()

    def _middle_available_px(self) -> int:
        """中栏（左右面板之间）解析可用宽度。

        用 ``footprint_width()`` 而非几何宽度：resizeEvent 内子控件几何
        滞后一帧，折叠/展开动画中途读 width() 会拿到过渡值。
        """
        margins = self.layout().contentsMargins()
        fixed = (margins.left() + margins.right()
                 + 2 * constants.PAGE_SPACING)
        return (self.width() - fixed
                - self._left_panel.footprint_width()
                - self._right_panel.footprint_width())

    def _auto_panel(self, panel, collapsed: bool) -> None:
        """自动折叠/展开单侧栏：抑制持久化（瞬态布局自适应，不写入设置）。"""
        self._in_auto_panel_change = True
        try:
            panel.set_collapsed(collapsed, animate=False)
        finally:
            self._in_auto_panel_change = False

    def _auto_collapse_side_panels(self) -> None:
        """窄窗自动折叠 / 恢复侧栏（策略见类 docstring）。"""
        room = self._middle_available_px()
        for side, panel in (('left', self._left_panel),
                            ('right', self._right_panel)):
            if panel.is_collapsed():
                # 展开该侧栏会消耗 expand−collapse 的中栏宽度，
                # 展开后仍不低于阈值才自动展开，且仅限自动折叠过的。
                cost = panel.expand_width() - panel.footprint_width()
                if (side in self._auto_collapsed_sides
                        and room - cost >= _MIDDLE_MIN_PX):
                    self._auto_panel(panel, False)
                    self._auto_collapsed_sides.discard(side)
                    room -= cost
            elif room < _MIDDLE_MIN_PX:
                # 该侧栏展开占位下中栏过窄 → 自动折叠让位（记住自动痕迹）。
                # 标记在折叠之后落：set_collapsed 同步触发 sig_collapsed →
                # _on_side_panel_collapsed 清理痕迹，随后再补标记。
                self._auto_panel(panel, True)
                self._auto_collapsed_sides.add(side)
                room += panel.expand_width() - panel.footprint_width()

    def _on_side_panel_collapsed(self, side: str, collapsed: bool) -> None:
        """侧栏折叠状态变化：手动操作清除自动痕迹（尊重用户），并持久化。

        自动路径（_in_auto_panel_change）不清痕迹、不写设置——痕迹由
        自动策略自己维护，持久化只记用户/外部程序化的显式变更。
        """
        if self._in_auto_panel_change:
            return
        self._auto_collapsed_sides.discard(side)
        self._save_panel_state()

    # ------------------------------------------------------------ 持久化
    def _save_panel_state(self) -> None:
        """把当前折叠状态写回共享 SettingsManager（未注入时静默跳过）。"""
        sm = self._sm
        if sm is None:
            return
        prefix = self._PANEL_STATE_PREFIX
        sm.set(f'{prefix}_left_collapsed', self._left_panel.is_collapsed())
        sm.set(f'{prefix}_right_collapsed', self._right_panel.is_collapsed())
        sm.save()

    def _restore_panel_state(self) -> None:
        """从共享 SettingsManager 恢复折叠状态（未注入不读盘）。"""
        sm = self._sm
        if sm is None:
            return
        prefix = self._PANEL_STATE_PREFIX
        self._left_panel.blockSignals(True)
        self._right_panel.blockSignals(True)
        try:
            self._left_panel.set_collapsed(
                bool(sm.get(f'{prefix}_left_collapsed', False)), animate=False)
            self._right_panel.set_collapsed(
                bool(sm.get(f'{prefix}_right_collapsed', False)), animate=False)
        finally:
            self._left_panel.blockSignals(False)
            self._right_panel.blockSignals(False)

    # ------------------------------------------------------------ 通用设置读写
    def _persist_setting(self, key: str, value) -> None:
        """写单个设置项：共享实例为唯一写者；未注入（单元测试）时静默跳过。

        各页曾各自复制这段逻辑；收敛到 mixin 单源，避免"某页忘了判 None"
        或"某页忘了 save"这类不一致。
        """
        sm = self._sm
        if sm is None:
            return
        sm.set(key, value)
        sm.save()

    def _restore_setting(self, key: str, default=None):
        """读单个设置项；未注入时返回 default（不读盘）。"""
        sm = self._sm
        if sm is None:
            return default
        return sm.get(key, default)
