"""B-Scan 绘图区下沿的采集标记箭头（透明浮层，零布局占位）。

设计取舍（2026-11-11 定稿）：

- **不额外占一行**。此前试过在 ``_glw`` 下面加 40px 的「标记轨道」，
  绘图区净高被直接吃掉；这一版改成挂在 ``_glw`` 里的透明浮层，几何
  严格等于 ViewBox 的矩形，因此控件与绘图区尺寸零增量。
- **贴下沿、尖朝下**。标记的语义是「第 N 道有事件」，箭头尖端指向道号
  刻度比悬在图像里更自然；ViewBox 自带 2% 留白，箭头下段落在那圈空白里，
  上半截贴住图像末行，像书签夹在页边。
- **倒三角 + 细边**。双肩造型有 6 个顶点和中间 V 形缺口，缩到 9px 后
  缺口只剩毛刺感；3 顶点三角在小尺寸下轮廓完整。描边 0.5px/α48 是
  Qt 抗锯齿能表达的最小一档，只为收边、不抢图像。
- **选中不常驻小标**。道号本来就在正下方刻度上、箭头尖正指着它，常驻
  小标属于重复信息，却必须付出「压住底部弱反射信号」「靠近左右边界被
  夹到边上与箭头脱钩」「标记靠近顶部时画不下直接消失」三个代价。选中
  只做 1.24× 放大 + 贯通虚线，具体道号交给悬浮 tooltip。
- **横向位置不让 UI 猜**：``vb.mapViewToScene`` → ``glw.mapFromScene``
  得到像素 x，缩放 / 平移 / 道号↔里程切换后跟着轴走，不在控件里做
  ``np.interp`` 近似。
- 浮层 ``WA_TransparentForMouseEvents``：十字光标、框选、平移、右键
  菜单全部照旧；命中判定由 :meth:`MarkerPins.hit_test` 提供，交给
  :meth:`BScanView._on_mouse_clicked` 调用。

源记录始终只读：这里只保存副本，选中态也只存副本。
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

__all__ = [
    'MARKER_COLORS', 'MARKER_KINDS', 'MARKER_NAMES', 'MAX_MARKERS', 'MarkerPins',
]

# 三种形式：与 core.cdut_acquisition_import 的白名单严格一致（没有 info）
MARKER_KINDS = ('start', 'general', 'end')

# 每种标记一套双档色 (浅色主题, 深色主题)。单套颜色无法同时兼顾两种底色：
# 箭头压在近白的 ViewBox 留白上，原配色（#35b58a / #39a8ed / #de79bb）的
# 亮度 138/143/159 离白底太近，缩到 9px 后粉色最先糊掉。浅色档因此整体
# 压深一档（对白底差 121~138），深色档反过来提亮（对 #1b1f26 底差 135+）。
# 色相不动，开始绿 / 一般蓝 / 结束品红的辨认习惯保留。
MARKER_COLORS = {
    'start': ('#1a9e73', '#3ecf9b'),
    'general': ('#2b8bd9', '#48a9f7'),
    'end': ('#c85c92', '#f08cc8'),
}
MARKER_NAMES = {'start': '开始', 'general': '一般', 'end': '结束'}

# 渲染上限：采集标记通常个位数，这里只是防御性截断（tooltip 会说明）。
MAX_MARKERS = 500

# 倒三角：设计框 18×18、尖在 (9, 2.6)，控件里整体翻转成尖朝下。
_ARROW_POINTS = ((9.0, 2.6), (14.4, 13.9), (3.6, 13.9))
_TIP = _ARROW_POINTS[0]
_GLYPH_W = 10.8          # (14.4 − 3.6)
_GLYPH_H = 11.3          # (13.9 − 2.6)

ARROW_SCALE = 0.80       # 1× 下 8.64 × 9.04 px，够看清又不压图像
ARROW_SCALE_SELECTED = 1.24
ARROW_SCALE_HOVER = 1.10
EDGE_WIDTH = 0.5         # 细边：只为收边，不形成硬轮廓
EDGE_ALPHA = 48
BOTTOM_GAP = 2.0         # 尖端离绘图区底边的距离
HIT_TOP = 18.0           # 命中带高度（从尖端往上，不随箭头缩小，保住可点性）
HIT_HALF_WIDTH = 8.0


def marker_color(kind: str, dark: bool) -> QColor:
    """某类标记在当前主题下的填充色。"""
    light, night = MARKER_COLORS.get(kind, MARKER_COLORS['general'])
    return QColor(night if dark else light)


def _arrow_path() -> QPainterPath:
    path = QPainterPath()
    path.moveTo(*_ARROW_POINTS[0])
    for point in _ARROW_POINTS[1:]:
        path.lineTo(*point)
    path.closeSubpath()
    return path


class MarkerPins(QWidget):
    """绘图区下沿的标记箭头（透明浮层，无底色、无布局占位）。

    信号:
        sig_selection_changed(object): 选中项变化时发源记录副本；取消选中发 None。
    """

    sig_selection_changed = pyqtSignal(object)

    def __init__(self, parent, plot_item, glw):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setMouseTracking(False)
        self._plot = plot_item
        self._glw = glw
        self._arrow = _arrow_path()
        self._records: list[dict] = []
        self._view_x: list[float] = []
        self._px: list[float] = []
        self._selected = None            # 选中记录的 key
        self._hover = None               # 悬停记录的下标
        self._dark = False
        self._syncing = False
        self._edge = QColor('#0b1017')
        self._edge.setAlpha(EDGE_ALPHA)
        vb = plot_item.vb
        vb.sigXRangeChanged.connect(self._on_view_changed)
        vb.sigYRangeChanged.connect(self._on_view_changed)
        vb.sigResized.connect(self._on_view_changed)
        self.hide()

    # ------------------------------------------------------------------ 数据
    @staticmethod
    def _key(record) -> tuple:
        return (int(record.get('trace_no', 0)),
                str(record.get('marker_type', 'general')),
                str(record.get('label', '')))

    def set_markers(self, records, view_positions) -> None:
        """喂入源记录副本 + 每条的显示坐标 x（列中心，来自 BScanView 的网格换算）。"""
        self._records = [dict(record) for record in records]
        self._view_x = [float(value) for value in view_positions]
        keys = {self._key(record) for record in self._records}
        if self._selected not in keys:
            had = self._selected is not None
            self._selected = None
            if had:
                self.sig_selection_changed.emit(None)
        self._hover = None
        self._sync_geometry()
        self._recompute()
        self.update()

    def marker_count(self) -> int:
        return len(self._records)

    def view_positions(self) -> list[float]:
        """每条标记的显示坐标 x（列中心，未换算像素）。"""
        return list(self._view_x)

    def pixel_positions(self) -> list[float]:
        """当前像素 x（未布局或越界时为 0）。"""
        return list(self._px)

    def shown(self) -> bool:
        """逻辑可见性：只反映「有标记 + 开关打开」，不随祖先窗口是否显示变化。

        ``QWidget.isVisible()`` 在祖先未显示时恒为 False（离屏视图、尚未
        show 的工作区分页都会这样），用它判定会把「开关是否打开」和
        「窗口是否已显示」混在一起。
        """
        return self.isVisibleTo(self.parentWidget())

    def tooltip(self, index: int) -> str:
        """单条标记的悬浮说明（道号以源文件行为准，预览可能降采样）。"""
        record = self._records[index]
        kind = str(record.get('marker_type', 'general'))
        row = int(record.get('trace_no', 0))
        text = f"采集{MARKER_NAMES.get(kind, '一般')}标记 · 原文件第{row}道"
        label = str(record.get('label', '') or '').strip()
        if label:
            text += f'\n{label}'
        text += '\n缩略图位置为显示近似；标记不代表已确认目标'
        return text

    def record_at(self, index: int) -> dict:
        return dict(self._records[index])

    # ------------------------------------------------------------------ 选中
    def selected(self):
        if self._selected is None:
            return None
        for record in self._records:
            if self._key(record) == self._selected:
                return dict(record)
        return None

    def selected_view_position(self):
        """选中项的显示坐标 x（列中心，未换算像素）；无选中返回 None。

        供 BScanView 把这一道带进视野 / 切 A-scan，避免下游再按记录
        反查索引而和浮层内部顺序耦合。
        """
        if self._selected is None:
            return None
        for record, x in zip(self._records, self._view_x):
            if self._key(record) == self._selected:
                return x
        return None

    def select(self, record) -> None:
        """按内容三元组选中（传 None 清空）；不存在的记录等同清空。"""
        key = None if record is None else self._key(record)
        keys = {self._key(item) for item in self._records}
        key = key if key in keys else None
        if key == self._selected:
            return
        self._selected = key
        self.update()
        self.sig_selection_changed.emit(self.selected())

    def set_hover(self, index) -> None:
        if index == self._hover:
            return
        self._hover = index
        self.update()

    # ------------------------------------------------------------------ 命中
    def _to_local(self, scene_pos) -> QPointF:
        return QPointF(self._glw.mapFromScene(scene_pos)) - QPointF(self.x(), self.y())

    def hit_test(self, scene_pos) -> int | None:
        """场景坐标 → 命中的标记下标（命中带贴着绘图区下沿）。"""
        if not self._records or not self.shown():
            return None
        local = self._to_local(scene_pos)
        height = self.height()
        if not (height - HIT_TOP <= local.y() <= height + 2):
            return None
        best, best_dx = None, HIT_HALF_WIDTH
        for index, px in enumerate(self._px):
            dx = abs(local.x() - px)
            if dx <= best_dx:
                best, best_dx = index, dx
        return best

    # ------------------------------------------------------------------ 几何
    def _on_view_changed(self, *_args) -> None:
        if self._syncing:
            return
        self._sync_geometry()
        self._recompute()
        self.update()

    def _sync_geometry(self) -> None:
        """几何严格取 ViewBox 的矩形：与绘图区同宽同高，不多占一像素。"""
        rect = self._plot.vb.sceneBoundingRect()
        top_left = self._glw.mapFromScene(rect.topLeft())
        bottom_right = self._glw.mapFromScene(rect.bottomRight())
        self._syncing = True
        try:
            self.setGeometry(top_left.x(), top_left.y(),
                             max(1, bottom_right.x() - top_left.x()),
                             max(1, bottom_right.y() - top_left.y()))
        finally:
            self._syncing = False

    def _recompute(self) -> None:
        vb = self._plot.vb
        ox = self.x()
        px = []
        for x_view in self._view_x:
            scene = vb.mapViewToScene(QPointF(x_view, 0.0))
            px.append(self._glw.mapFromScene(scene).x() - ox)
        self._px = px

    # ------------------------------------------------------------------ 绘制
    def set_theme(self, dark: bool) -> None:
        self._dark = bool(dark)
        self._edge = QColor('#0b1017')
        self._edge.setAlpha(EDGE_ALPHA if not dark else EDGE_ALPHA + 12)
        self.update()

    def paintEvent(self, event) -> None:
        if not self._records:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        width, height = self.width(), self.height()
        tip_y = height - BOTTOM_GAP
        edge = QPen(self._edge, EDGE_WIDTH)
        edge.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        for index, (record, px) in enumerate(zip(self._records, self._px)):
            if px < -16 or px > width + 16:
                continue
            kind = str(record.get('marker_type', 'general'))
            color = marker_color(kind, self._dark)
            selected = self._selected is not None and self._key(record) == self._selected
            if selected:
                self._draw_guide(painter, px, tip_y, color)
                scale = ARROW_SCALE_SELECTED
            else:
                scale = ARROW_SCALE_HOVER if index == self._hover else ARROW_SCALE
            painter.save()
            painter.translate(px, tip_y)
            painter.scale(scale, -scale)          # 垂直翻转：尖朝下指向道号
            painter.translate(-_TIP[0], -_TIP[1])
            painter.setPen(edge)
            painter.setBrush(color)
            painter.drawPath(self._arrow)
            painter.restore()
            if kind in ('start', 'end'):
                self._draw_bar(painter, px, tip_y, color, kind, scale)
        painter.end()

    @staticmethod
    def _draw_guide(painter: QPainter, px: float, tip_y: float, color: QColor) -> None:
        """选中标记的贯通虚线：把「这个箭头对应整列」说清楚。"""
        pen = QPen(color, 1.0, Qt.PenStyle.DashLine)
        pen.setDashPattern([3, 3])
        painter.setPen(pen)
        painter.drawLine(QPointF(px, 0.0), QPointF(px, tip_y))

    @staticmethod
    def _draw_bar(painter: QPainter, px: float, tip_y: float,
                  color: QColor, kind: str, scale: float) -> None:
        """开始/结束各加一条短杆（左 / 右），信息点保持干净。"""
        half = _GLYPH_W * scale / 2.0 + 1.5
        bar_x = px - half if kind == 'start' else px + half
        top = tip_y - _GLYPH_H * scale + 2.0
        painter.fillRect(QRectF(bar_x - 0.8, top, 1.6, _GLYPH_H * scale - 4.0), color)
