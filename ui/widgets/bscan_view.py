"""BScanView — B-Scan 图像显示控件（SPEC §5.1）。

性能纪律（style_spec §1 / §3.3）：
- 预分配 ImageItem 复用，setImage 不重建
- matrix 约定为 (samples, traces)，row-major 语义下零拷贝直接显示
  （x=道/列、y=采样/行；不再 .T，见 set_matrix 注释）
- ImageItem(axisOrder='row-major') + invertY(True)
- autoLevels=False + 显式 levels=(vmin, vmax)
- ColorBarItem 随行同步

轴单位（Phase1 能力包）：横轴 道↔距离 在 AxisItem 刻度文本层换算（索引
坐标系不动）；纵轴 采样轴↔海拔 是**两种不同的显示网格**——
- 采样轴模式：显示坐标 = 数据索引（x=道、y=采样点），与 pick/overlay/
  降采样换算的既有约定一致；
- 海拔模式：整幅剖面经 :func:`bscan_axes.build_elevation_view` 重采样到
  **共享绝对海拔网格**（每道按自己的地面高程下移，海拔 = 逐道地面高程 −
  深度），顶边即真实地貌，地表以上 NaN → 透明留白。此模式下显示纵坐标
  是「海拔网格行号」，_view_to_data/_data_to_view 负责行号↔采样点换算。
海拔 warp 带「同一份数据只算一次」缓存，且只作用于显示层，原始数据零变异。

十字光标读数：鼠标在图像区移动时显示十字线 + 左下角读数浮层
（道号/距起点/纵轴物理值/幅值；PreviewBundle 带 trace_axis_m /
sample_axis 时显示物理量，降采样数据附"原始约 N"），右键菜单可开关。

右键菜单（RoundMenu）：缩放组（放大/缩小/回到全览/方形/1:1/全屏/铺满）/
轴单位子菜单 / 色标子菜单 / 十字光标 / A-scan 跟随 / 显示模式 / 色阶设置 /
复制 / 导出 PNG。工具条已退役（2026-09-23：4 钮与右键菜单完全重复），
全屏保留为图内左上角悬浮钮，铺满走右键菜单，轴单位与比例策略走设置页
与右键菜单。
接入时已 vb.setMenuEnabled(False) 关闭 pyqtgraph 原生英文菜单
（代价：右键拖拽框选缩放失效，由菜单缩放项补偿），并隐藏 pyqtgraph
自带的「A」自适应钮（其功能由右键菜单「回到全览」承担）。
"""

import math

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from enum import Enum

import pyqtgraph as pg
from PyQt6.QtWidgets import QLabel
from qfluentwidgets import FluentIcon as FIF, ToolButton

from ui import constants
from ui.desktop_backend_facade import compute_display_levels
from ui.theme_helpers import control_palette
from ui.widgets._colormap_data import COLORMAP_DATA
from ui.widgets.bscan_axes import (
    X_AXIS_MODES,
    Y_AXIS_MODES,
    IndexAxis,
    build_elevation_view,
    distance_available,
    distance_tick_strings,
    elevation_tick_strings,
    sample_unit_label,
)
from ui.widgets.bscan_fullscreen import FullscreenHost
from ui.widgets.bscan_levels_dialog import LevelsDialog
from ui.widgets.context_menus import (RoundMenu, add_action,
                                      add_checkable_submenu, make_menu)
from ui.widgets.empty_state import EmptyStateOverlay
from ui.widgets.pg_view_base import GraphicsViewBase, style_plot_item

# 色标缓存：名 → ColorMap（数据查表重建，进程内只构建一次）
_COLORMAP_CACHE = {}


# 显示动态范围默认值：与处理页 p_low/p_high 一致（2% / 98% 百分位）
DEFAULT_P_LOW, DEFAULT_P_HIGH = 2.0, 98.0


def _normalise_levels(levels) -> tuple[float, float] | None:
    """把外部传入的动态范围规整为合法 (low, high)；非法返回 None。

    「非法」= 不是两个数 / 非有限 / 不在 [0,100] / low >= high。跨会话恢复时
    设置文件可能被手改成任意内容，必须在入口收口，否则 0/0 会被
    compute_display_levels 带进退化分支（静默返回 ±1，图看起来"没坏"但色阶
    已经被改错）。返回 None 让调用方自己决定是「回落默认」还是「拒绝本次」。
    """
    try:
        low, high = float(levels[0]), float(levels[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    if not (0.0 <= low < high <= 100.0):
        return None
    return low, high


def _restored_levels(levels) -> tuple[float, float]:
    """跨会话恢复用：非法值静默回落默认百分位（不能让坏设置挡住出图）。"""
    return _normalise_levels(levels) or (DEFAULT_P_LOW, DEFAULT_P_HIGH)


def _lookup_colormap(name: str):
    """按名取 ColorMap：优先查预采样数据（零 matplotlib import）。

    getFromMatplotlib 首次调用会触发 matplotlib 全量 import（实测 ~0.9s），
    主页预览卡构造期即命中，是冷启动大头。九项 SPEC 色标由
    scripts/gen_colormap_data.py 离线采样为 _colormap_data.py，重建结果
    与 getFromMatplotlib 逐点一致（1024 点采样最大通道差 0，脚本自验）。
    数据缺失的色标（新增未重跑生成脚本）回落原路径，正确性不受影响。
    """
    cmap = _COLORMAP_CACHE.get(name)
    if cmap is not None:
        return cmap
    entry = COLORMAP_DATA.get(name)
    if entry is not None:
        cmap = pg.colormap.ColorMap(pos=entry[0], color=entry[1], name=name)
    else:
        cmap = pg.colormap.getFromMatplotlib(name)
    if cmap is not None:
        cmap.name = name
        _COLORMAP_CACHE[name] = cmap
    return cmap


class BScanDisplayMode(Enum):
    """B-scan 显示模式（Phase1 1.2）：灰度 / 变面积 / 波形叠加。"""

    GRAYSCALE = 'grayscale'
    WIGGLE = 'wiggle'          # 变面积：正半轴填充
    WAVEFORM = 'waveform'      # 波形叠加：正负对称双线


def _downsample_map_index(view_index: int, view_count: int,
                          data_count: int) -> int:
    """显示坐标索引 → 原始数据索引（strided 降采样近似线性映射）。

    读数"原始约 N"标注与 pick/overlay 坐标换算共用本函数，避免双实现漂移。
    """
    return int(round(view_index * (data_count - 1)
                     / max(view_count - 1, 1)))


def _downsample_map_index_inverse(data_index, view_count: int,
                                  data_count: int):
    """原始数据索引 → 显示坐标（_downsample_map_index 的逆映射）。"""
    return (data_index * max(view_count - 1, 1)
            / max(data_count - 1, 1))


def format_crosshair_readout(trace: int, sample: int, shape: tuple,
                             amplitude: float, *,
                             trace_axis_m=None, sample_axis=None,
                             sample_axis_label: str = '',
                             trace_count: int = 0,
                             sample_count: int = 0) -> str:
    """十字光标读数文本（纯函数，便于测试）。

    trace/sample 为显示坐标（0 基）；shape=(n_traces, n_display_rows)。
    trace_axis_m/sample_axis 为与显示矩阵等长的物理轴（None 跳过）——
    海拔模式下调用方传共享海拔轴 + '海拔 (m)' 标签，纵轴读数即为海拔。
    trace_count/sample_count 为原始（未降采样）数量，与显示数不同
    时追加"（原始约 N）"（strided 降采样近似线性映射）。
    """
    n_traces, n_rows = shape
    lines = [f'道 {trace + 1}']
    if trace_count and trace_count != n_traces:
        approx = _downsample_map_index(trace, n_traces, trace_count)
        lines[0] += f'（原始约 {approx + 1}）'
    if trace_axis_m is not None and 0 <= trace < len(trace_axis_m):
        lines.append(f'距起点 {float(trace_axis_m[trace]):.3g} m')
    if sample_axis is not None and 0 <= sample < len(sample_axis):
        label = sample_axis_label or '纵轴'
        lines.append(f'{label} {float(sample_axis[sample]):.4g}')
    else:
        text = f'采样 {sample + 1}'
        if sample_count and sample_count != n_rows:
            approx = _downsample_map_index(sample, n_rows, sample_count)
            text += f'（原始约 {approx + 1}）'
        lines.append(text)
    if amplitude is None or not math.isfinite(float(amplitude)):
        lines.append('幅值 —')       # 海拔模式地表以上是 NaN（留白区）
    else:
        lines.append(f'幅值 {float(amplitude):.4g}')
    return '\n'.join(lines)


class BScanView(GraphicsViewBase, QWidget):
    """B-Scan 剖面图像视图。

    缩放/导出/轴主题继承 GraphicsViewBase（pg_view_base）；
    比例策略（方形/铺满/1:1）与十字光标等为本类专属。

    信号:
        sig_point_picked(int, int): pick 模式下鼠标点击发射 (trace_index, sample_index)，
            统一为原始数据坐标（预览降采样时已换算）。
        sig_colormap_changed(str): 右键菜单切换色标时发射（宿主写回设置用）。
    """

    sig_point_picked = pyqtSignal(int, int)
    sig_colormap_changed = pyqtSignal(str)
    # 用户手动切换显示比例时发射（'free'/'square'/'cell'），页面据此持久化
    sig_aspect_changed = pyqtSignal(str)
    # 用户手动切换轴单位时发射（'trace'/'distance'、'sample'/'elevation'）
    sig_x_axis_changed = pyqtSignal(str)
    sig_y_axis_changed = pyqtSignal(str)
    # 用户手动改色阶（低/高百分位）时发射，供宿主写回设置
    sig_levels_changed = pyqtSignal(float, float)
    # 用户手动切换色标显隐时发射，供宿主写回设置
    sig_colorbar_visible_changed = pyqtSignal(bool)
    # 用户手动切换显示增益（'off'/'sec'/'tvg'）时发射，供宿主写回设置
    sig_gain_changed = pyqtSignal(str)
    # TVG 参数（总增益 dB、弯度幂次）经滑条面板调整后持久化（关闭面板时发一次）
    sig_gain_params_changed = pyqtSignal(float, float)
    # 工具条「⤢ 铺满」：本视图不认识页面布局，交给宿主页面折叠侧栏
    sig_expand_requested = pyqtSignal()

    def __init__(self, parent=None, *, with_colorbar: bool = True,
                 default_aspect: str = 'free', default_x_axis: str = 'trace',
                 default_y_axis: str = 'sample',
                 default_levels: tuple | None = None):
        super().__init__(parent)
        self._pick_enabled = False
        self._image_shape = None  # (traces, samples) 显示坐标系尺寸
        # 显示比例策略：'free' 拉伸铺满（默认——实测占空比 85.9%，x/y 拉伸比
        # 1.09 几乎不畸变）/ 'square' 数据盒锁正方形（占空比仅 43.3%，且把
        # 4:1 的真实剖面压成 1:1、双曲线形状失真）/ 'cell' 数据格 1:1。
        # 旧默认 'square' 是"B-Scan 习惯比例"的想当然，实测两项指标都最差，
        # 2026-09-22 改默认为 'free'；方形仍保留在右键菜单随时可切。
        self._aspect_mode = default_aspect if default_aspect in (
            'square', 'free', 'cell') else 'free'
        self._cmap_name = constants.DEFAULT_COLORMAP
        self._cmap = None
        # 轴单位（Phase1 能力包）：用户偏好 + 海拔换算数据
        self._x_axis = default_x_axis if default_x_axis in X_AXIS_MODES else 'trace'
        self._y_axis = default_y_axis if default_y_axis in Y_AXIS_MODES else 'sample'
        # 显示动态范围（百分位裁切）：右键「色阶设置」可改，不改数据
        self._p_low, self._p_high = _restored_levels(default_levels)
        self._matrix = None               # 当前显示矩阵（供重算色阶/刷新用）
        # 色标显隐偏好：与色标对象解耦（with_colorbar=False 的视图无色标，
        # 偏好仍记录，构造处若支持可后续兑现）
        self._colorbar_visible = True
        # 显示增益（SEC/TVG）：显示域逐行缩放（raw 一个字节不动）；增益后
        # 矩阵按 (矩阵身份, 模式+参数) 缓存复用。TVG 参数经滑条面板调整
        self._gain_mode = 'off'
        self._gain_alpha = 0.2
        self._gain_db = 24.0        # TVG 深端总增益（dB）
        self._gain_power = 1.0      # TVG 曲线弯度（幂次：1 线性，>1 集中深部）
        self._gain_cache = None
        self._tvg_dialog = None     # 滑条面板（懒创建，随视图销毁）
        # 「⤢ 铺满」只有接到了宿主页（能折叠侧栏）才有意义，默认隐藏该钮
        self._expand_enabled = False
        # 全屏宿主（懒创建）+ 几何持久化回调（主窗口注入，避免 view 碰文件）
        self._fullscreen = None
        self._geometry_loader = None
        self._geometry_saver = None
        # A-scan 波形跟随浮窗（懒创建；Phase1 1.1）
        self._ascan_popup = None
        self._ascan_follow = False
        # 显示模式（Phase1 1.2）：灰度默认；wiggle/waveform 懒创建 path item
        self.display_mode = BScanDisplayMode.GRAYSCALE
        self._wiggle_item = None
        self._export_title = ''   # PNG 导出文件名用（数据身份，随数据更新）
        # 十字光标读数状态（PreviewBundle 物理轴元数据）
        self._init_axis_state()

        self._init_plot(with_colorbar)
        self._init_readout_overlay()
        self._init_layout()

        # 空态引导浮层（评审 P0-1）：初始无数据即显示，数据到达隐藏
        self._empty_overlay = EmptyStateOverlay(
            self._glw, icon=FIF.PHOTO, title='暂无数据',
            hint='导入测线并选择数据后，此处预览雷达 B-Scan 剖面')

        self._glw.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._glw.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._refresh_axis_state()
        from qfluentwidgets import isDarkTheme
        self.apply_theme(isDarkTheme())

    def _init_axis_state(self) -> None:
        """十字光标读数与海拔换算所需的物理轴元数据（无数据时全空）。"""
        self._crosshair_on = True
        self._trace_axis_m = None
        self._sample_axis = None
        self._sample_axis_label = ''
        self._trace_count = 0
        self._sample_count = 0
        self._ground_elevation_m = None   # 逐道地面高程（m，显示网格）
        self._depth_axis_m = None         # 地表以下深度（m，显示网格）
        # 海拔模式状态：当前显示的是否是 warp 后的矩阵 + 其海拔轴
        self._showing_elevation = False
        self._elev_axis = None            # 降序海拔轴（行 r ↔ elev_axis[r]）
        self._elev_cache = None           # (matrix, ground, depth, warped, axis)

    def _init_plot(self, with_colorbar: bool) -> None:
        """图形区：ImageItem 预分配复用 + 十字线 + 色标 + overlay 散点。"""
        self._glw = pg.GraphicsLayoutWidget(self)
        # 图内标题只承载数据身份（bundle.title，如"测线 L3"）；通用名称
        # "B-Scan图像"由卡头承担，不再双标题（评审 P1-4）。
        # 横/纵轴换成 IndexAxis：索引坐标不变，只在刻度文本层做单位换算
        self._bottom_axis = IndexAxis('bottom', self._convert_bottom_ticks)
        self._left_axis = IndexAxis('left', self._convert_left_ticks)
        self._plot = self._glw.addPlot(
            row=0, col=0,
            axisItems={'bottom': self._bottom_axis, 'left': self._left_axis})
        self._plot_item = self._plot   # GraphicsViewBase 约定属性
        self._plot.setLabel('bottom', '道数')
        self._plot.setLabel('left', '采样点')
        self._plot.invertY(True)
        self._plot.setMouseEnabled(x=True, y=True)
        # pyqtgraph 自带的「A」自适应钮（英文语境的 autoscaling）由右键
        # 菜单「回到全览」取代，这里直接隐藏避免两套入口
        self._plot.hideButtons()
        # 关闭 pyqtgraph 原生英文右键菜单（右拖框选缩放随之失效，
        # 缩放操作由自定义 RoundMenu 提供）
        self._plot.vb.setMenuEnabled(False)

        # 预分配并复用
        self._image_item = pg.ImageItem(axisOrder='row-major')
        self._image_item.setLevels((0.0, 1.0))
        self._plot.addItem(self._image_item)

        # 十字光标线（默认隐藏，鼠标进入图像区显示）
        self._vline = pg.InfiniteLine(angle=90, movable=False)
        self._hline = pg.InfiniteLine(angle=0, movable=False)
        for line in (self._vline, self._hline):
            line.setVisible(False)
            self._plot.addItem(line, ignoreBounds=True)

        self._colorbar = None
        if with_colorbar:
            # 瘦身：去掉竖排「幅度」label、刻度 7pt 小字——值条语义自明
            # （幅度由上下文与读数浮层表达），每图省 ~25px 宽还给画布。
            self._colorbar = pg.ColorBarItem(interactive=False)
            self._colorbar.setImageItem(self._image_item,
                                        insert_in=self._plot)
            self._colorbar.axis.setStyle(
                tickFont=QFont(constants.FONT_FAMILY, 7))

        # overlay 标注散点（解释页）
        self._scatter = pg.ScatterPlotItem(
            size=8, pen=None,
            brush=pg.mkBrush(constants.CHART_OVERLAY_COLOR))
        self._plot.addItem(self._scatter)
        self.set_colormap('seismic')

    @staticmethod
    def _compact_title_html(title: str) -> str:
        """图内标题 8pt 小字 + 超长截短（空间瘦身双保险）。

        - 8pt：标题行高约减半（pyqtgraph 默认字号太占纵向）；
        - 截短：**title 会参与 PlotItem 布局的列最小宽**（实测每字符
          ~4px @8pt），长成果名会把右侧色标列推出视口（444px 卡上
          30 字符即出界，实探针）；截到 20 字符（保头尾，中略）后
          窄栏双图也稳。空标题时 pyqtgraph 自动隐藏。
        """
        text = str(title or '')
        if not text:
            return ''
        if len(text) > 20:
            text = text[:12] + '…' + text[-7:]
        text = (text.replace('&', '&amp;')
                .replace('<', '&lt;').replace('>', '&gt;'))
        return f'<span style="font-size:8pt">{text}</span>'

    def _init_readout_overlay(self) -> None:
        """十字光标读数浮层（左下角，半透明底白字，深浅主题通用）。"""
        self._readout = QLabel(self)
        self._readout.setStyleSheet(
            'QLabel { background-color: rgba(0, 0, 0, 150); '
            'color: white; border-radius: 4px; padding: 6px 10px; '
            f'font-size: {constants.FONT_SIZE_SECONDARY}pt; '
            'line-height: 1.5; }')
        self._readout.setWordWrap(False)
        self._readout.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._readout.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._readout.hide()
        # 读数文本 30ms 节流：移动中十字线即时跟手（setPos 便宜），
        # 文本 setText/adjustSize 触发布局合并到定时器刷新（首次出现即时）。
        self._readout_timer = QTimer(self)
        self._readout_timer.setSingleShot(True)
        self._readout_timer.setInterval(30)
        self._readout_timer.timeout.connect(self._flush_crosshair_readout)
        self._pending_readout = None   # (trace, sample, amplitude)

    def _init_layout(self) -> None:
        """图形区占满整控件（工具条已退役，能力分流右键菜单/悬浮钮）。

        2026-09-23 评审：工具条 4 钮与右键菜单缩放组完全重复，每视图
        常驻吃一行 ~40px（dual/quad/free 多画布时成倍）——缩放走右键
        与滚轮，「铺满」并入右键菜单；全屏高频，保留为图内左上角悬浮
        钮（与左下读数浮层对称，不与居中图内标题、右侧色标冲突）。
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._glw, 1)

        self._fullscreen_btn = ToolButton(FIF.FULL_SCREEN, self)
        self._fullscreen_btn.setToolTip('全屏：在独立窗口中展开（Esc 退出）')
        self._fullscreen_btn.setFixedSize(*constants.TOOL_BTN_COMPACT)
        self._fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        self._fullscreen_btn.move(6, 6)
        self._fullscreen_btn.raise_()
        self._fullscreen_btn.show()

    # ------------------------------------------------------------------ 轴单位
    def x_axis_mode(self) -> str:
        """横轴单位偏好：'trace' 道号 / 'distance' 里程 (m)。"""
        return self._x_axis

    def y_axis_mode(self) -> str:
        """纵轴单位偏好：'sample' 采样轴（时间/深度） / 'elevation' 海拔 (m)。"""
        return self._y_axis

    def effective_y_axis_mode(self) -> str:
        """实际生效的纵轴单位：偏好是海拔但 warp 没成立时回落 'sample'。

        以 ``_showing_elevation``（当前显示的是否是海拔网格矩阵）为准，
        而不是重新检查数据可用性——warp 是一次性结论，重查既慢又可能与
        已显示内容不一致。
        """
        if self._y_axis == 'elevation' and not self._showing_elevation:
            return 'sample'
        return self._y_axis

    def distance_available(self) -> bool:
        """横轴能否切成里程：需与显示矩阵等长的逐道里程轴。"""
        if self._image_shape is None:
            return False
        return distance_available(self._trace_axis_m, self._image_shape[0])

    def elevation_available(self) -> bool:
        """纵轴能否切成海拔：逐道地面高程（与道数等长）+ 深度轴齐备。"""
        if self._matrix is None:
            return False
        if self._showing_elevation:
            return True
        if self._ground_elevation_m is None or self._depth_axis_m is None:
            return False
        if not len(self._depth_axis_m):
            return False
        return len(self._ground_elevation_m) >= self._matrix.shape[1]

    def set_x_axis_mode(self, mode: str, *, notify: bool = False) -> None:
        """切换横轴单位（'trace' / 'distance'）；无里程轴时拒绝切到距离。

        :param notify: True 时发 sig_x_axis_changed（供页面写回设置）；
            恢复持久化阶段应传 False。
        """
        if mode not in X_AXIS_MODES:
            return
        if mode == 'distance' and not self.distance_available():
            return
        self._x_axis = mode
        self._apply_axis_modes()
        if notify:
            self.sig_x_axis_changed.emit(mode)

    def set_y_axis_mode(self, mode: str, *, notify: bool = False) -> None:
        """切换纵轴单位（'sample' / 'elevation'）；数据不支持时不接受海拔。

        接受后由 ``_apply_axis_modes`` 完成真正的换图（整幅 warp 到共享
        海拔网格）。偏好语义不变：只有当前数据确实不支持时才拒绝，
        支持与否由 ``elevation_available`` 判定。
        """
        if mode not in Y_AXIS_MODES:
            return
        if mode == 'elevation' and not self.elevation_available():
            return
        self._y_axis = mode
        self._apply_axis_modes()
        if notify:
            self.sig_y_axis_changed.emit(mode)

    def set_axis_modes(self, x_mode: str, y_mode: str, *,
                       notify: bool = False) -> None:
        """一次性恢复横/纵轴单位（二维为一处调用，避免中间态闪一下）。"""
        self.set_x_axis_mode(x_mode, notify=notify)
        self.set_y_axis_mode(y_mode, notify=notify)

    def _apply_axis_modes(self) -> None:
        """按当前单位刷新显示矩阵、轴标签与刻度文本（切换/来数据共用）。"""
        swapped = self._refresh_elevation_image()
        self._plot.setLabel('bottom', self._bottom_label())
        self._plot.setLabel('left', self._left_label())
        self._refresh_axis_state()
        if swapped:
            # 显示网格变了（海拔模式行数 ≠ 采样数）：按当前比例策略重铺，
            # 且波形纵坐标映射变了（行号↔海拔），非灰度模式要按新网格重画
            self._fit_current_mode()
            if self.display_mode is not BScanDisplayMode.GRAYSCALE:
                self._apply_display_mode(self.display_mode, reset_view=False)

    def _refresh_axis_state(self) -> None:
        """刷新刻度画面（轴标签保持调用方/set_bundle 的既有结论）。"""
        self._bottom_axis.invalidate()
        self._left_axis.invalidate()

    def _bottom_label(self) -> str:
        return ('距离 (m)' if self._x_axis == 'distance'
                and self.distance_available() else '道数')

    def _left_label(self) -> str:
        """纵轴标签随单位变：采样轴用 bundle 的标签，海拔轴固定带单位。"""
        if self.effective_y_axis_mode() == 'elevation':
            return '海拔 (m)'
        return self._sample_axis_label or '采样点'

    def _convert_bottom_ticks(self, values, scale, spacing):
        """横轴刻度换算：距离模式换算成里程，否则回落默认（道索引）。"""
        if self._x_axis != 'distance':
            return None
        return distance_tick_strings(values, self._trace_axis_m)

    def _convert_left_ticks(self, values, scale, spacing):
        """纵轴刻度换算：海拔模式行号直接查共享海拔轴，否则回落默认。"""
        if not self._showing_elevation:
            return None
        return elevation_tick_strings(values, self._elev_axis)

    # ------------------------------------------------------------------ 缩放 / 导出
    def _export_grab_target(self):
        """PNG 导出/复制抓取目标：图形画布（不含悬浮钮与读数浮层）。"""
        return self._glw

    def _fit_view(self) -> None:
        self.fit_to_data()

    def aspect_mode(self) -> str:
        """当前比例策略：'free' / 'square' / 'cell'。"""
        return self._aspect_mode

    def set_aspect_mode(self, mode: str, *, notify: bool = False) -> None:
        """按名切换比例策略（外部恢复持久化设置用）。

        :param notify: True 时发 sig_aspect_changed（供页面写回设置）；
            恢复阶段应传 False，避免"读设置→写设置"回环。
        """
        if mode == 'free':
            self.fit_to_data(notify=notify)
        elif mode == 'cell':
            self.reset_1to1(notify=notify)
        elif mode == 'square':
            self.fit_square(notify=notify)

    def fit_to_data(self, *, notify: bool = True) -> None:
        """自适应窗口：显示全部数据（解除纵横锁定，拉伸铺满）。"""
        self._aspect_mode = 'free'
        self._plot.vb.setAspectLocked(False)
        self._plot.vb.autoRange()
        if notify:
            self.sig_aspect_changed.emit('free')

    def fit_square(self, *, notify: bool = True) -> None:
        """近似方形显示：锁定纵横比，使数据包围盒在屏幕上接近正方形。

        pyqtgraph setAspectLocked 的 ratio 是「一个 x 单位在屏幕上的宽度 /
        一个 y 单位在屏幕上的高度」。数据盒宽 n_traces、高 n_samples，
        要让它显示为正方形：n_traces * ratio = n_samples → ratio = 高/宽。

        注意：本模式会把整个数据盒拉成正方形，与数据自身的长宽比无关——
        真实的 4:1 剖面会被横向压缩 4 倍，双曲线形状失真。仅作对照用，
        默认已改为 'free'（见 __init__ 注释）。
        """
        self._aspect_mode = 'square'
        if self._image_shape is None:
            self._plot.vb.setAspectLocked(False)
            self._plot.vb.autoRange()
        else:
            n_traces, n_samples = self._image_shape
            vb = self._plot.vb
            vb.setAspectLocked(False)
            vb.setRange(xRange=(0, n_traces), yRange=(0, n_samples),
                        padding=0.02)
            vb.setAspectLocked(
                True, ratio=float(n_samples) / max(float(n_traces), 1.0))
        if notify:
            self.sig_aspect_changed.emit('square')

    def reset_1to1(self, *, notify: bool = True) -> None:
        """恢复像素 1:1 显示（x/y 轴等比例）。"""
        self._aspect_mode = 'cell'
        if self._image_shape is None:
            self._plot.vb.setAspectLocked(False)
            self._plot.vb.autoRange()
        else:
            n_traces, n_samples = self._image_shape
            self._plot.vb.setRange(
                xRange=(0, n_traces), yRange=(0, n_samples), padding=0.0)
            self._plot.vb.setAspectLocked(True, ratio=1.0)
        if notify:
            self.sig_aspect_changed.emit('cell')

    def _fit_current_mode(self) -> None:
        """按当前比例策略铺满视野（新数据到达时调用，不重置用户选择）。

        notify=False：这是数据驱动重排，不是用户切换，不该触发持久化。
        """
        # 注意三个分支都直接置 _aspect_mode 后铺满，不重复发信号
        mode = self._aspect_mode
        if mode == 'free':
            self.fit_to_data(notify=False)
        elif mode == 'cell':
            self.reset_1to1(notify=False)
        else:
            self.fit_square(notify=False)

    # ------------------------------------------------------------------ 数据
    def set_bundle(self, bundle) -> None:
        """接收 PreviewBundle（鸭子类型，不 import core.gui_rendering）。"""
        self.set_matrix(
            bundle.matrix, bundle.vmin, bundle.vmax,
            title=str(getattr(bundle, 'title', '') or ''),
            x_label=getattr(bundle, 'x_label', '道数'),
            y_label=getattr(bundle, 'y_label', '采样点'),
        )
        # 十字光标读数用物理轴元数据（可选；与显示矩阵同降采样）
        self._trace_axis_m = getattr(bundle, 'trace_axis_m', None)
        self._sample_axis = getattr(bundle, 'sample_axis', None)
        self._sample_axis_label = str(
            getattr(bundle, 'sample_axis_label', '') or '')
        self._trace_count = int(getattr(bundle, 'trace_count', 0) or 0)
        self._sample_count = int(getattr(bundle, 'sample_count', 0) or 0)
        # 海拔纵轴的两份原料（均已降到显示网格）：逐道地面高程 + 米制深度
        self._ground_elevation_m = getattr(bundle, 'trace_elevation_m', None)
        self._depth_axis_m = getattr(bundle, 'depth_axis_m', None)
        # 数据换了 → 轴可用性可能变（偏好若是海拔，新的数据支持则自动恢复）
        self._apply_axis_modes()
        # 增益开启时 set_matrix 阶段只能用行号近似曲线（物理轴此刻才就位），
        # 按物理轴补一次整体重渲染；海拔路径已在 _refresh_elevation_image
        # 内通过 _resolve_display_target 拿到增益后目标，无需重复。
        if self._gain_mode != 'off' and not self._showing_elevation:
            self._refresh_gain_render()

    def set_matrix(self, matrix, vmin, vmax, *,
                   title="", x_label="道数", y_label="采样点") -> None:
        """matrix 为 (samples, traces)，零拷贝视图直接显示。

        row-major 语义下 array[r, c] → (x=c, y=r)，故 (samples, traces)
        矩阵本身就是显示布局：x=道数（列）、y=采样点（行），无需转置拷贝。
        （SPEC §5.1 字面要求"显示 .T 转置视图"系沿袭师兄缓冲 (traces,
        samples) 约定；本控件输入约定为 (samples, traces)，若再 .T 会
        使 x/y 轴互换，与 x_label=道数 / y_label=采样点 及
        sig_point_picked(trace, sample) 契约矛盾，故按契约语义实现。）

        本函数只送**原始（未变形）矩阵**的增益视图上屏（_matrix 本体保留
        raw，供十字读数/波形/海拔 warp 使用）：海拔 warp 的原料（逐道高程/
        深度轴）由 set_bundle 在本函数之后才赋值，故海拔模式的换图统一由
        _apply_axis_modes → _refresh_elevation_image 兜底（set_bundle 末尾
        必然调用），这里不重复做。
        """
        import numpy as np

        mat = np.asarray(matrix)
        if mat.ndim != 2:
            raise ValueError('B-Scan 矩阵必须是二维 (samples, traces)')
        view = mat  # 零拷贝；row-major 下 y=采样(行)、x=道(列)
        self._matrix = view
        # 直接 set_matrix 的调用方没有物理轴元数据，读数退回索引显示
        self._trace_axis_m = None
        self._sample_axis = None
        self._sample_axis_label = ''
        self._trace_count = 0
        self._sample_count = 0
        # 无 bundle 元数据 → 无物理轴：海拔/距离不可用（回落采样轴显示）
        self._ground_elevation_m = None
        self._depth_axis_m = None
        self._showing_elevation = False
        self._elev_axis = None
        new_shape = (view.shape[1], view.shape[0])   # (traces, samples)
        shape_changed = new_shape != self._image_shape
        self._image_shape = new_shape
        # 上屏的是增益后矩阵（_matrix 保留 raw 供读数/波形/海拔 warp）；
        # 此刻物理轴元数据尚未由 set_bundle 赋值，增益曲线先行退行号——
        # set_bundle 末尾在增益开启时会按物理轴整体重渲染一次。
        self._image_item.setImage(self._gain_applied(view), autoLevels=False,
                                  levels=(float(vmin), float(vmax)))
        if shape_changed:
            # 新数据尺寸变化时按当前比例策略铺满视野，避免换测线后图像跑出可视区
            self._fit_current_mode()
        # 色阶偏好（可能是跨会话恢复或用户在无数据时设的）应用到新数据上：
        # 优先于调用方给的 vmin/vmax，因为那只是数据的默认裁切。
        # 注意下面还会统一重渲染一次波形，故这里让它跳过重绘（见参数注释）。
        self._apply_levels_to_render(redraw_waveform=False)
        if self._colorbar is not None:
            self._colorbar.setLevels(self._image_item.levels)
        # 非灰度模式下按当前显示模式重渲染波形；新数据到达保持用户视野，
        # 不重置缩放（reset_view=False，模式切换才重置）
        if self.display_mode is not BScanDisplayMode.GRAYSCALE:
            self._apply_display_mode(self.display_mode, reset_view=False)
        # 图内标题只留数据身份（如"测线 L3"）；空标题时 pyqtgraph 自动隐藏。
        # 8pt 小字（空间瘦身）：身份保留、标题行高约减半，默认字号太占纵向。
        self._export_title = str(title or '')
        self._plot.setTitle(self._compact_title_html(self._export_title))
        self._plot.setLabel('bottom', x_label)
        self._plot.setLabel('left', y_label)
        # 数据换了 → 轴可用性与刻度都要重算（单位切换不动，标签由调用方给定）
        self._refresh_axis_state()
        self._empty_overlay.setVisible(False)

    def _resolve_display_target(self) -> tuple:
        """按纵轴偏好解析应显示的矩阵：海拔偏好且可 warp → (warped, axis)。

        增益是显示链第一级：先对 raw 矩阵逐行缩放（每个采样点拿到自己
        深度的精确增益），海拔 warp 的原料即增益后矩阵——warp 的逐列
        线性重采样不破坏行增益语义。增益关闭时 src 即 raw（零拷贝）。
        """
        src = self._gain_applied(self._matrix)
        if self._y_axis == 'elevation':
            warped, axis = self._warped_elevation(src)
            if warped is not None:
                return warped, axis
        return src, None

    def _warped_elevation(self, src) -> tuple:
        """海拔 warp（缓存：同一份数据/高程/深度只算一次，按对象身份判重）。

        缓存键用对象身份而不是 id()：id 在对象被回收后可能被新对象复用，
        身份比较（is）则绝对可靠。增益开关/参数变化会生成新的 src 对象
        （身份必变），warp 缓存随之自动失效重算。

        :param src: 增益后矩阵（见 _resolve_display_target）。
        """
        if src is None:
            return None, None
        cache = self._elev_cache
        if (cache is not None
                and cache[0] is src
                and cache[1] is self._ground_elevation_m
                and cache[2] is self._depth_axis_m):
            return cache[3], cache[4]
        warped, axis = build_elevation_view(
            src, self._ground_elevation_m, self._depth_axis_m)
        self._elev_cache = (src, self._ground_elevation_m,
                            self._depth_axis_m, warped, axis)
        return warped, axis

    def _refresh_elevation_image(self) -> bool:
        """按纵轴偏好换显示矩阵（海拔↔采样轴切换共用）。返回是否换图。"""
        if self._matrix is None:
            return False
        want = self._y_axis == 'elevation'
        target, elev_axis = None, None
        if want:
            target, elev_axis = self._resolve_display_target()
            if elev_axis is None:
                want = False          # 数据不支持，回落采样轴显示
        if want == self._showing_elevation:
            return False
        if not want:
            target, elev_axis = self._matrix, None
        levels = self._image_item.levels
        self._image_item.setImage(target, autoLevels=False, levels=levels)
        self._showing_elevation = want
        self._elev_axis = elev_axis
        self._image_shape = (target.shape[1], target.shape[0])
        return True

    def set_colormap(self, name: str) -> None:
        """按 matplotlib 名取 LUT（九项见 SPEC §1，默认 seismic）。"""
        self._cmap_name = str(name)
        self._cmap = _lookup_colormap(str(name))
        self._image_item.setColorMap(self._cmap)
        if self._colorbar is not None:
            self._colorbar.setColorMap(self._cmap)

    def display_levels(self) -> tuple[float, float]:
        """当前显示动态范围的百分位 ``(low, high)``。"""
        return self._p_low, self._p_high

    def set_display_levels(self, p_low, p_high, *, notify: bool = False) -> bool:
        """按百分位重算并应用显示色阶（只改显示，不动数据）。

        **偏好与渲染分离**：百分位本身是纯状态，永远记下来；只有「算 vmin/vmax
        并推给 ImageItem」这一步需要数据。跨会话恢复发生在**还没有数据**的
        时刻（主窗构造期），若此时整体放弃，用户存的 5/95 会被静默丢掉、
        直到他手动再设一次——`set_matrix` 会用 `_p_low/_p_high` 重算，
        所以这里记下来就够。

        :param notify: True 时发 sig_levels_changed（供宿主写回设置）；
            恢复持久化阶段传 False。
        :return: 是否真的重算了渲染（无数据时返回 False，但偏好已记下）
        """
        normalised = _normalise_levels((p_low, p_high))
        if normalised is None:
            return False
        low, high = normalised
        self._p_low, self._p_high = low, high
        self._apply_levels_to_render()
        if notify:
            self.sig_levels_changed.emit(low, high)
        return self._matrix is not None

    def _apply_levels_to_render(self, *, redraw_waveform: bool = True) -> None:
        """把当前 ``_p_low/_p_high`` 换算成 vmin/vmax 推到图像与色标（无数据则跳过）。

        :param redraw_waveform: 波形模式的振幅归一取自 levels，故色阶变了一般要
            重绘；但 ``set_matrix`` 会在随后统一重绘一次，传 False 避免同一帧
            画两遍（波形重绘是整幅遍历，不便宜）。
        """
        if self._matrix is None:
            return
        # 色阶在**增益后**矩阵上取百分位：增益与百分位正交（增益纵向拉平、
        # 百分位横向裁剪），顺序为先增益、再取百分位
        vmin, vmax = compute_display_levels(
            self._gain_applied(self._matrix),
            p_low=self._p_low, p_high=self._p_high)
        self._image_item.setLevels((float(vmin), float(vmax)))
        if self._colorbar is not None:
            self._colorbar.setLevels((float(vmin), float(vmax)))
        if (redraw_waveform
                and self.display_mode is not BScanDisplayMode.GRAYSCALE):
            # 波形模式的振幅归一取自 levels，色阶变了要按新 levels 重画
            self._apply_display_mode(self.display_mode, reset_view=False)

    def colorbar_visible(self) -> bool:
        """色标显隐偏好（与色标对象是否存在解耦）。"""
        return self._colorbar_visible

    # ------------------------------------------------------------------ 显示增益
    def gain_mode(self) -> str:
        """当前显示增益模式：'off' / 'sec' / 'tvg'。"""
        return self._gain_mode

    def set_gain(self, mode: str, *, alpha: float | None = None,
                 db: float | None = None, power: float | None = None,
                 notify: bool = False) -> None:
        """切换显示增益；显示域逐行缩放，raw 一个字节不动。

        :param mode: 'off' 关闭 / 'sec' SEC 补偿（扩散∝深度 + 指数吸收）/
            'tvg' TVG 补偿（用户滑条调参的幂次曲线）。
        :param alpha: SEC 衰减补偿系数（dB/采样轴单位：深度轴为 dB/m，
            时间轴为 dB/ns）；None 保持现值。
        :param db: TVG 深端总增益（dB，浅端恒为 0 增益）；None 保持现值。
        :param power: TVG 曲线弯度幂次（1 线性，>1 增益集中深部）；None 保持。
        :param notify: True 时发 sig_gain_changed（TVG 另发
            sig_gain_params_changed 供宿主持久化参数）；恢复持久化阶段
            传 False。
        """
        mode = str(mode)
        if mode not in ('off', 'sec', 'tvg'):
            raise ValueError(f'未知增益模式: {mode!r}')
        changed = (mode != self._gain_mode
                   or (alpha is not None and float(alpha) != self._gain_alpha)
                   or (db is not None and float(db) != self._gain_db)
                   or (power is not None
                       and float(power) != self._gain_power))
        self._gain_mode = mode
        if alpha is not None:
            self._gain_alpha = float(alpha)
        if db is not None:
            self._gain_db = float(db)
        if power is not None:
            self._gain_power = float(power)
        if changed:
            self._gain_cache = None
            self._refresh_gain_render()
        if notify:
            self.sig_gain_changed.emit(self._gain_mode)
            if mode == 'tvg':
                self.sig_gain_params_changed.emit(self._gain_db,
                                                  self._gain_power)

    def _refresh_gain_render(self) -> None:
        """增益变了：按当前解析目标整体换图 + 重算色阶/波形。

        与 _refresh_elevation_image 的差异：无论海拔状态是否变化都要换图
        （增益改的是像素值本身），色阶随增益后矩阵重算（见
        _apply_levels_to_render）。
        """
        if self._matrix is None:
            return
        target, elev_axis = self._resolve_display_target()
        levels = self._image_item.levels
        self._image_item.setImage(target, autoLevels=False, levels=levels)
        self._showing_elevation = elev_axis is not None
        self._elev_axis = elev_axis
        self._image_shape = (target.shape[1], target.shape[0])
        self._apply_levels_to_render()

    def _gain_applied(self, mat):
        """显示域增益变换：off 原样返回（零拷贝）；sec/tvg 返回逐行缩放结果。

        缓存按 (矩阵身份, 模式+参数) 判重——与海拔 warp 缓存同一纪律
        （身份比较，不用 id()）。增益关闭时必须原样返回以保持「零拷贝 +
        身份不变」语义（海拔 warp 缓存、回落判定都依赖它）。
        """
        if mat is None or self._gain_mode == 'off':
            return mat
        cache = self._gain_cache
        if (cache is not None and cache[0] is mat
                and cache[1] == self._gain_mode
                and cache[2] == self._gain_alpha
                and cache[3] == self._gain_db
                and cache[4] == self._gain_power):
            return cache[5]
        import numpy as np
        if self._gain_mode == 'sec':
            gain = self._sec_row_gain(mat.shape[0])
        else:
            gain = self._tvg_row_gain(mat.shape[0])
        out = np.asarray(mat) * gain[:, None]
        self._gain_cache = (mat, self._gain_mode, self._gain_alpha,
                            self._gain_db, self._gain_power, out)
        return out

    def _phys_depth_axis(self, n_rows: int):
        """物理采样轴（米/纳秒）；无轴/长度不符/含 NaN 时返回 None。

        返回原数组（不拷贝），浅端归零与归一化由各增益曲线自行处理。
        """
        import numpy as np
        axis = self._sample_axis
        if axis is None:
            return None
        arr = np.asarray(axis, dtype=np.float64)
        if (arr.ndim == 1 and arr.size == n_rows
                and np.isfinite(arr).all() and n_rows >= 2):
            return arr
        return None

    def _sec_row_gain(self, n_rows: int):
        """SEC 行增益曲线（长度 n_rows，浅→深单调放大）。

        g(d) = (d + d0)/d0 × 10^(α·d/20)：扩散项 ∝ 深度（1/r 幅值补偿）
        + 指数吸收补偿。d 取物理采样轴（米/纳秒，α 单位跟随轴）；无轴或
        轴长不符时退行号（set_matrix 阶段物理轴尚未就位的近似，set_bundle
        末尾会按物理轴重渲染）。浅端参考 d0 取轴量程 2% 防 1/r 发散；
        总增益限幅 1000×（60 dB）防深部噪声底被抬满。
        """
        import numpy as np
        arr = self._phys_depth_axis(n_rows)
        if arr is not None:
            d = np.abs(arr - arr[0])     # 浅端归零（兼容升/降序轴）
        else:
            d = np.arange(n_rows, dtype=np.float64)
        d0 = max(float(np.ptp(d)) * 0.02, 1e-6)
        gain = ((d + d0) / d0) * np.power(10.0, self._gain_alpha * d / 20.0)
        return np.clip(gain, 1.0, 1000.0)

    def _tvg_row_gain(self, n_rows: int):
        """TVG 行增益曲线：g(u) = 10^(db·u^power / 20)，u 为归一化深度。

        u∈[0,1]（浅端 0、深端 1，取物理轴，无轴退行号）；浅端恒 1×，
        深端恰为 10^(db/20)。power=1 线性递增，>1 增益向深部集中，
        <1 向浅部集中（滑条范围限 0.3~3.0）。限幅同 SEC（1000×）。
        """
        import numpy as np
        arr = self._phys_depth_axis(n_rows)
        if arr is not None:
            span = float(np.ptp(arr))
            u = (np.abs(arr - arr[0]) / span) if span > 0 else \
                np.linspace(0.0, 1.0, n_rows)
        else:
            u = (np.arange(n_rows, dtype=np.float64) / max(n_rows - 1, 1))
        power = min(max(self._gain_power, 0.1), 10.0)
        gain = np.power(10.0, self._gain_db * np.power(u, power) / 20.0)
        return np.clip(gain, 1.0, 1000.0)

    def _on_gain_menu(self, mode: str) -> None:
        """右键「显示增益」子菜单动作：切模式；选 TVG 时弹滑条面板。"""
        self.set_gain(mode, notify=True)
        if mode == 'tvg':
            self._open_tvg_dialog()

    def _open_tvg_dialog(self) -> None:
        """打开 TVG 滑条面板（非模态，随视图销毁；重复调用只唤起）。"""
        if self._tvg_dialog is not None:
            self._tvg_dialog.show()
            self._tvg_dialog.raise_()
            self._tvg_dialog.activateWindow()
            return
        from ui.widgets.bscan_tvg_dialog import TvgGainDialog
        self._tvg_dialog = TvgGainDialog(self)
        self._tvg_dialog.show()

    def set_colorbar_visible(self, visible: bool, *, notify: bool = False) -> None:
        """切换右侧色标条的显示；隐藏后其宽度完整归还画布。

        探针实证（``_probe_colorbar_toggle.py``）：ColorBarItem 插在 plot
        右列，纯 ``setVisible`` 即可——隐藏后 vb.width 644→716（72px 完整
        归还），再显示恢复 644，免动 GraphicsLayout。with_colorbar=False
        构造的视图没有色标对象，此调用只记偏好不动渲染。

        :param notify: True 时发 sig_colorbar_visible_changed（供宿主写回
            设置）；恢复持久化阶段传 False。
        """
        visible = bool(visible)
        self._colorbar_visible = visible
        if self._colorbar is not None:
            self._colorbar.setVisible(visible)
        if notify:
            self.sig_colorbar_visible_changed.emit(visible)

    def _edit_levels(self) -> None:
        """右键「色阶设置…」：弹窗取低/高百分位后应用。"""
        dialog = LevelsDialog(self.window(), self._p_low, self._p_high)
        if dialog.exec() != LevelsDialog.DialogCode.Accepted:
            return
        chosen = dialog.percentiles()
        if chosen is None:
            return
        self.set_display_levels(*chosen, notify=True)

    def _choose_colormap(self, name: str) -> None:
        """右键菜单选色标：应用到本视图并发信号让页面同步控件。"""
        self.set_colormap(name)
        self.sig_colormap_changed.emit(name)

    # ------------------------------------------------------------------ 视图范围
    def set_expand_enabled(self, enabled: bool) -> None:
        """由宿主页决定右键是否露「铺满窗口」（页面能折叠侧栏才有意义）。"""
        self._expand_enabled = bool(enabled)

    def set_geometry_store(self, loader=None, saver=None) -> None:
        """注入全屏窗口几何的读写回调（view 自己不碰文件/设置）。

        :param loader: ``() -> QByteArray | None``，回放历史几何；
        :param saver:  ``(QByteArray) -> None``，窗口关闭时保存。
        """
        self._geometry_loader = loader
        self._geometry_saver = saver

    def request_expand(self) -> None:
        """「铺满」：本视图不认识页面布局，发信号让宿主页面去折叠侧栏。"""
        self.sig_expand_requested.emit()

    def is_fullscreen(self) -> bool:
        """当前是否处于独立窗口展开态。"""
        return self._fullscreen is not None and self._fullscreen.isVisible()

    def toggle_fullscreen(self) -> None:
        """进出独立窗口展开（重复调用即切换）。"""
        if self.is_fullscreen():
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def enter_fullscreen(self) -> None:
        """把本控件临时改嫁到独立窗口（同一实例，交互状态完全连续）。

        几何回放走宿主注入的 loader；没有历史几何时给一个合理的初始尺寸
        （按屏幕的 85%，保证足够大的同时不遮死其它窗口）。
        """
        if self._fullscreen is None:
            self._fullscreen = FullscreenHost(self.window())
            self._fullscreen.closed.connect(self._on_fullscreen_closed)
        host = self._fullscreen
        host.take(self)
        restored = False
        if self._geometry_loader is not None:
            try:
                restored = host.restore_saved_geometry(self._geometry_loader())
            except Exception:  # noqa: BLE001 - 几何回放失败不该挡住全屏
                restored = False
        if not restored:
            screen = self.screen()
            available = screen.availableGeometry() if screen is not None else None
            if available is not None:
                host.resize(int(available.width() * 0.85),
                            int(available.height() * 0.85))
        host.show()
        host.raise_()
        host.activateWindow()
        self._fit_current_mode()

    def exit_fullscreen(self) -> None:
        """退出独立窗口展开（窗口自行 close，closed 信号里做收尾）。"""
        if self._fullscreen is not None:
            self._fullscreen.close()

    def _on_fullscreen_closed(self) -> None:
        """窗口关闭后：几何存回宿主，画布按当前比例重新铺满。"""
        if self._geometry_saver is not None and self._fullscreen is not None:
            try:
                self._geometry_saver(self._fullscreen.captured_geometry())
            except Exception:  # noqa: BLE001 - 存几何失败不该报错给用户
                pass
        self._fit_current_mode()

    # ------------------------------------------------------------------ 右键菜单
    def _show_context_menu(self, event) -> None:
        menu = make_menu(self)
        self._add_menu_zoom(menu)
        menu.addSeparator()
        self._add_menu_axes(menu)
        menu.addSeparator()
        add_checkable_submenu(menu, '色标', constants.COLORMAPS,
                              self._cmap_name, self._choose_colormap)
        add_action(menu, None, '色阶设置…', self._edit_levels,
                   enabled=self._image_shape is not None)
        self._add_menu_toggles(menu)
        menu.addSeparator()
        add_action(menu, FIF.COPY, '复制图像', self._copy_image,
                   enabled=self._image_shape is not None)
        add_action(menu, FIF.SAVE, '导出 PNG…',
                   lambda: self.export_png(
                       title=self._export_title, prefix='bscan'),
                   enabled=self._image_shape is not None)
        menu.exec(event.screenPos().toPoint())

    def _add_menu_zoom(self, menu) -> None:
        """缩放与视野组：缩放 / 回到全览 / 比例 / 全屏 / 铺满。

        「铺满」（收起页面侧栏）只在宿主页接管后出现——原工具条按钮
        退役后，这里是除快捷语义外的唯一入口。
        """
        add_action(menu, FIF.ZOOM_IN, '放大', self.zoom_in)
        add_action(menu, FIF.ZOOM_OUT, '缩小', self.zoom_out)
        add_action(menu, FIF.FIT_PAGE, '回到全览（铺满窗口）', self.fit_to_data)
        add_action(menu, None, '方形显示', self.fit_square)
        add_action(menu, None, '1:1（数据格等比）', self.reset_1to1)
        add_action(menu, FIF.FULL_SCREEN, '全屏浏览', self.enter_fullscreen)
        if self._expand_enabled:
            add_action(menu, FIF.VIEW, '铺满（收起页面侧栏）',
                       self.request_expand)

    def _add_menu_axes(self, menu) -> None:
        """轴单位组：工具条精简后这里是切换主入口（另一处是设置页）。"""
        x_menu = RoundMenu('横轴单位', menu)
        for mode, label in (('trace', '道数'), ('distance', '距离 (m)')):
            x_menu.addAction(self._axis_menu_action(
                label, self._x_axis == mode,
                mode != 'distance' or self.distance_available(),
                lambda m=mode: self.set_x_axis_mode(m, notify=True)))
        menu.addMenu(x_menu)
        y_menu = RoundMenu('纵轴单位', menu)
        y_menu.addAction(self._axis_menu_action(
            sample_unit_label(self._sample_axis_label),
            self.effective_y_axis_mode() == 'sample', self._image_shape is not None,
            lambda: self.set_y_axis_mode('sample', notify=True)))
        y_menu.addAction(self._axis_menu_action(
            '海拔 (m)', self.effective_y_axis_mode() == 'elevation',
            self.elevation_available(),
            lambda: self.set_y_axis_mode('elevation', notify=True)))
        menu.addMenu(y_menu)

    @staticmethod
    def _axis_menu_action(label: str, checked: bool, enabled: bool, slot):
        from qfluentwidgets import Action
        act = Action(label)
        act.setCheckable(True)
        act.setChecked(bool(checked))
        act.setEnabled(bool(enabled))
        act.triggered.connect(lambda _checked=False: slot())
        return act

    def _add_menu_toggles(self, menu) -> None:
        """交互开关组：十字光标 / A-scan 跟随 / 色标显隐 / 显示模式。"""
        from qfluentwidgets import Action
        crosshair_action = Action('十字光标读数')
        crosshair_action.setCheckable(True)
        crosshair_action.setChecked(self._crosshair_on)
        crosshair_action.triggered.connect(self._toggle_crosshair)
        menu.addAction(crosshair_action)
        ascan_action = Action('A-scan 波形跟随')
        ascan_action.setCheckable(True)
        ascan_action.setChecked(self._ascan_follow)
        ascan_action.triggered.connect(self.set_ascan_follow)
        menu.addAction(ascan_action)
        if self._colorbar is not None:
            colorbar_action = Action('显示色标')
            colorbar_action.setCheckable(True)
            colorbar_action.setChecked(self._colorbar_visible)
            colorbar_action.triggered.connect(
                lambda checked=False: self.set_colorbar_visible(
                    checked, notify=True))
            menu.addAction(colorbar_action)
        mode_submenu = RoundMenu('显示模式', menu)
        for mode in BScanDisplayMode:
            mode_label = {
                BScanDisplayMode.GRAYSCALE: '灰度',
                BScanDisplayMode.WIGGLE: '变面积（Wiggle）',
                BScanDisplayMode.WAVEFORM: '波形叠加',
            }[mode]
            act = Action(mode_label)
            act.setCheckable(True)
            act.setChecked(self.display_mode is mode)
            act.triggered.connect(
                lambda _checked=False, m=mode: self.set_display_mode(m))
            mode_submenu.addAction(act)
        menu.addMenu(mode_submenu)
        gain_submenu = RoundMenu('显示增益', menu)
        for mode, label in (('off', '关闭'), ('sec', 'SEC 补偿'),
                            ('tvg', 'TVG 补偿（滑条调参）')):
            act = Action(label)
            act.setCheckable(True)
            act.setChecked(self._gain_mode == mode)
            act.triggered.connect(
                lambda _checked=False, m=mode: self._on_gain_menu(m))
            gain_submenu.addAction(act)
        menu.addMenu(gain_submenu)

    # ------------------------------------------------------------------ 交互
    def set_pick_enabled(self, enabled: bool) -> None:
        """开启后鼠标点击把图像坐标换算成 (trace, sample) 发 sig_point_picked。"""
        self._pick_enabled = bool(enabled)

    def _toggle_crosshair(self, checked: bool) -> None:
        self._crosshair_on = bool(checked)
        if not self._crosshair_on:
            self._hide_crosshair()

    def set_display_mode(self, mode) -> None:
        """切换灰度/变面积/波形叠加三态；模式切换时重置视野（用户主动切换）。"""
        if isinstance(mode, str):
            mode = BScanDisplayMode(mode)
        if not isinstance(mode, BScanDisplayMode):
            raise ValueError(f'未知显示模式: {mode!r}')
        self.display_mode = mode
        self._apply_display_mode(mode, reset_view=True)

    def _apply_display_mode(self, mode, *, reset_view: bool) -> None:
        """应用显示模式；reset_view 仅用户切换模式时为 True。

        set_matrix 新数据到达时以 reset_view=False 重渲染波形，保持用户
        当前缩放/视野（原实现无条件 autoRange，新数据会冲掉手动缩放）。
        波形数据源固定用未变形的 ``_matrix``（套显示增益，与灰度图一致）：
        海拔模式下 ImageItem 里是 warp 后的矩阵，直接取列会把波形纵向
        重采样；正确做法是值取原矩阵（增益后）、纵坐标按海拔映射
        （见 _render_wiggle 的 y_context）。
        """
        img = self._image_item.image
        if img is None or self._matrix is None:
            return
        if mode is BScanDisplayMode.GRAYSCALE:
            if self._wiggle_item is not None:
                self._wiggle_item.hide()
            self._image_item.show()
            return
        if mode is BScanDisplayMode.WIGGLE:
            self._image_item.hide()
        else:
            self._image_item.show()
        self._render_wiggle(self._gain_applied(self._matrix),
                            filled=(mode is BScanDisplayMode.WIGGLE),
                            symmetric=(mode is BScanDisplayMode.WAVEFORM),
                            reset_view=reset_view,
                            y_context=self._wiggle_y_context())

    def _wiggle_y_context(self):
        """海拔模式下波形的纵坐标映射参数；采样轴模式返回 None。

        返回 ``(elev_axis, ground, depth)``：波形每一点的纵坐标 =
        (最高海拔 − (该道地面高程 − 深度)) / 行距，即海拔网格行号——
        每道波形从自己的地面高程处起画，随地形起伏。
        """
        if not self._showing_elevation or self._elev_axis is None:
            return None
        return self._elev_axis, self._ground_elevation_m, self._depth_axis_m

    def _render_wiggle(self, data, *, filled: bool, symmetric: bool,
                       reset_view: bool = False, y_context=None) -> None:
        """变面积/波形叠加：pg.arrayToQPath C 层批量构建（向量化，非逐道循环）。

        填充（变面积）用每道"上升沿正包络+基线回程"闭合；波形叠加为
        全波形双线。connect 数组在每道末尾断开，避免跨道连线。

        :param y_context: None → 纵坐标 = 采样点行号（采样轴模式）；
            (elev_axis, ground, depth) → 纵坐标 = 海拔网格行号（海拔模式）。
        """
        import numpy as np
        from PyQt6.QtGui import QColor

        data = np.asarray(data, dtype=np.float64)
        n_samples, n_traces = data.shape
        levels = self._image_item.levels
        if levels is None or len(levels) < 2 or levels[0] == levels[1]:
            levels = (-1.0, 1.0)
        vmax = max(abs(float(levels[0])), abs(float(levels[1]))) or 1.0
        # 显示层降采样：波形形态用 ≤500 点/道已足够（峰值包络不失真），
        # 全样本会让 QPainterPath 剖析+绘制逼近秒级（实测 2000×1000 全点
        # ~1.1s，500 点 ~40ms）。
        step = max(1, n_samples // 500)
        if step > 1:
            data = data[::step, :]
        n_samples_ds = data.shape[0]
        # 振幅归一：每道正峰占相邻道间距的 0.45（Wiggle 惯例不重叠）
        gain = 0.45 / vmax
        bases = np.arange(n_traces, dtype=np.float64) + 0.5  # 每道基线

        amp = data * gain                      # (samples_ds, traces)
        if filled:
            xs = bases[None, :] + np.maximum(amp, 0.0)
        else:
            xs = bases[None, :] + amp
        ys_col, ys_bottom = self._wiggle_y_arrays(
            n_samples, n_samples_ds, step, n_traces, y_context)
        # 坐标序列：道1全部点 → 道1基线回程点 → 道2... （每道 2 段闭合）
        xs_t = np.concatenate([xs, bases[None, :]], axis=0)      # (n_ds+1, traces)
        ys_t = np.concatenate([ys_col, ys_bottom[None, :]], axis=0)
        # 列优先展开为点序列，connect=0 断开道间
        x_flat = xs_t.T.ravel()
        y_flat = ys_t.T.ravel()
        connect = np.ones(x_flat.size, dtype=np.int32)
        connect[np.arange(n_traces) * (n_samples_ds + 1) + n_samples_ds] = 0
        connect[-1] = 0

        path = pg.arrayToQPath(x_flat, y_flat, connect)
        path.setFillRule(Qt.FillRule.OddEvenFill if filled
                         else Qt.FillRule.WindingFill)

        from pyqtgraph import GraphicsItem  # noqa: F401 - 确保 pg 图层初始化
        from PyQt6.QtWidgets import QGraphicsPathItem
        if self._wiggle_item is None:
            self._wiggle_item = QGraphicsPathItem()
            self._plot.addItem(self._wiggle_item)
        self._wiggle_item.setPath(path)
        # 填充色取当前色表正端；描边为其深色
        lut = (self._cmap.getLookupTable(0.0, 1.0, 2)
               if self._cmap is not None else None)
        rgb = (int(lut[1][0]), int(lut[1][1]), int(lut[1][2])) \
            if lut is not None else (30, 30, 30)
        fill = QColor(*rgb, 160)
        self._wiggle_item.setPen(pg.mkPen(QColor(*rgb).darker(140), width=1))
        self._wiggle_item.setBrush(fill if filled else pg.mkBrush(None))
        self._wiggle_item.show()
        if reset_view:
            self._plot.getViewBox().autoRange()

    def _wiggle_y_arrays(self, n_samples: int, n_samples_ds: int,
                         step: int, n_traces: int, y_context):
        """波形的纵坐标数组（降采样后每点一行号 + 每道基线回程行号）。

        采样轴模式：行号 = 降采样索引 × 步距，基线回程 = 最深采样行。
        海拔模式：行号 = (最高海拔 − (地面高程 − 深度)) / 行距，每道
        各自从自己的地面高程起画（地形起伏直接体现在波形起点上）。
        """
        import numpy as np

        if y_context is None:
            ys_col = np.repeat(
                np.arange(n_samples_ds, dtype=np.float64)[:, None] * step,
                n_traces, axis=1)
            ys_bottom = np.full(n_traces, float(n_samples - 1))
            return ys_col, ys_bottom
        elev_axis, ground, depth = y_context
        top = float(elev_axis[0])
        step_m = (top - float(elev_axis[-1])) / max(len(elev_axis) - 1, 1)
        if step_m <= 0:
            ys_col = np.zeros((n_samples_ds, n_traces), dtype=np.float64)
            return ys_col, np.zeros(n_traces, dtype=np.float64)
        sample_idx = np.clip(
            np.arange(n_samples_ds, dtype=np.float64) * step,
            0, len(depth) - 1)
        d = np.asarray(depth, dtype=np.float64)[sample_idx.astype(np.int64)]
        g = np.asarray(ground, dtype=np.float64)[:n_traces]
        # 行号 = (top − (ground − depth)) / step_m；depth ≥ 0 → 行号 ≥ 0，
        # 且 top = 最高地面 → 任何道的最浅点行号也不会越出网格顶。
        ys_col = (top - g[None, :] + d[:, None]) / step_m
        ys_bottom = (top - g + float(depth[-1])) / step_m
        return ys_col, ys_bottom

    def set_ascan_follow(self, enabled: bool) -> None:
        """开关"A-scan 波形跟随"：懒创建浮窗并同步 pick 模式（Phase1 1.1）。"""
        from ui.widgets.ascan_popup import AScanPopup

        self._ascan_follow = bool(enabled)
        if enabled:
            if self._ascan_popup is None:
                self._ascan_popup = AScanPopup(self.window())
                self._ascan_popup.closed.connect(self._on_ascan_popup_closed)
            self._ascan_popup.show()
            self.set_pick_enabled(True)
        elif self._ascan_popup is not None:
            self._ascan_popup.hide()

    def _on_ascan_popup_closed(self) -> None:
        """浮窗被用户关闭（仅隐藏、实例复用）：回落跟随标志。

        与菜单取消勾选同效——不动 pick 模式（set_ascan_follow(False)
        同样只隐藏浮窗），仅让下次右键菜单的勾选态与浮窗实际状态一致。
        """
        self._ascan_follow = False

    def _on_mouse_moved(self, pos) -> None:
        """鼠标在图像区移动：十字线跟手 + 左下角读数浮层（文本 30ms 合并刷新）。"""
        if not self._crosshair_on or self._image_shape is None:
            self._hide_crosshair()
            return
        if not self._plot.sceneBoundingRect().contains(pos):
            self._hide_crosshair()
            return
        view_point = self._plot.vb.mapSceneToView(pos)
        trace, sample = int(view_point.x()), int(view_point.y())
        n_traces, n_samples = self._image_shape
        if not (0 <= trace < n_traces and 0 <= sample < n_samples):
            self._hide_crosshair()
            return
        # ImageItem 像素中心在半整数处，十字线对到像素中心
        self._vline.setPos(trace + 0.5)
        self._hline.setPos(sample + 0.5)
        self._vline.setVisible(True)
        self._hline.setVisible(True)
        self._pending_readout = (trace, sample,
                                 float(self._image_item.image[sample, trace]))
        if self._readout.isVisible():
            if not self._readout_timer.isActive():
                self._readout_timer.start()
        else:
            self._flush_crosshair_readout()   # 首次出现即时，不等节流
        self._readout.setVisible(True)

    def _flush_crosshair_readout(self) -> None:
        """节流刷新读数文本 + 浮层定位（_readout_timer 与首次出现共用）。"""
        pending = self._pending_readout
        if pending is None:
            return
        self._pending_readout = None
        trace, sample, amplitude = pending
        # 海拔模式下纵轴读数 = 行号直查共享海拔轴（显示坐标已是行号）
        showing_elevation = self._showing_elevation
        self._readout.setText(format_crosshair_readout(
            trace, sample, self._image_shape, amplitude,
            trace_axis_m=self._trace_axis_m,
            sample_axis=(self._elev_axis if showing_elevation
                         else self._sample_axis),
            sample_axis_label=('海拔 (m)' if showing_elevation
                               else self._sample_axis_label),
            trace_count=self._trace_count,
            sample_count=self._sample_count))
        self._position_readout()

    def _hide_crosshair(self) -> None:
        self._readout_timer.stop()
        self._pending_readout = None
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._readout.hide()

    def _position_readout(self) -> None:
        self._readout.adjustSize()
        margin = 10
        self._readout.move(
            margin, self.height() - self._readout.height() - margin)

    def leaveEvent(self, event) -> None:
        self._hide_crosshair()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._readout.isVisible():
            self._position_readout()

    def set_overlay_points(self, points, color: str = constants.CHART_OVERLAY_COLOR) -> None:
        """解释页标注散点：points 为 [(trace, sample), ...]（原始数据坐标系）。

        预览矩阵可能经 strided 降采样（>900×1800），此处把原始坐标
        映射回显示坐标再绘制，保证标注落在正确的图像位置上。
        """
        spots = [{'pos': self._data_to_view(t, s), 'brush': pg.mkBrush(color)}
                 for t, s in (points or [])]
        self._scatter.setData(spots)

    # ------------------------------------------------------------------ 坐标映射
    def _view_to_data(self, trace: int, sample: int) -> tuple:
        """显示坐标 → 原始数据坐标（strided 降采样近似线性映射）。

        采样轴模式：y 显示坐标 = 采样点索引（降采样时线性映射回原始）。
        海拔模式：y 显示坐标 = 海拔网格行号 → 该道地面高程处的深度 →
        深度轴查表得采样点（行号越界夹取；地表以上/最深处以下夹到端点）。
        无物理轴元数据（直接 set_matrix）或未降采样时为恒等映射。
        """
        if self._image_shape is None:
            return int(trace), int(sample)
        n_traces, n_rows = self._image_shape
        t, r = int(trace), int(sample)
        r = max(0, min(n_rows - 1, r))
        if self._showing_elevation:
            s = self._row_to_sample(t, r)
        else:
            s = r
            if self._sample_count and self._sample_count != n_rows:
                s = _downsample_map_index(s, n_rows, self._sample_count)
        if self._trace_count and self._trace_count != n_traces:
            t = _downsample_map_index(t, n_traces, self._trace_count)
        return t, s

    def _row_to_sample(self, trace_display: int, row: int) -> int:
        """海拔网格行 → 采样点索引（该道地面高程 − 行海拔 = 深度 → 查深度轴）。"""
        import numpy as np

        ground, depth = self._ground_elevation_m, self._depth_axis_m
        if ground is None or depth is None or self._elev_axis is None:
            return 0
        if not len(depth) or not len(ground):
            return 0
        ti = max(0, min(len(ground) - 1, int(trace_display)))
        ri = max(0, min(len(self._elev_axis) - 1, int(row)))
        depth_at_row = float(ground[ti]) - float(self._elev_axis[ri])
        idx = int(np.searchsorted(np.asarray(depth, dtype=np.float64),
                                  depth_at_row))
        return max(0, min(len(depth) - 1, idx))

    def _data_to_view(self, trace, sample) -> tuple:
        """原始数据坐标 → 显示坐标（_view_to_data 的逆映射）。

        海拔模式下返回行号（浮点，overlay 散点可落在行间）。
        """
        if self._image_shape is None:
            return float(trace), float(sample)
        n_traces, n_rows = self._image_shape
        t, s = float(trace), float(sample)
        if self._trace_count and self._trace_count != n_traces:
            t = _downsample_map_index_inverse(t, n_traces, self._trace_count)
        if self._showing_elevation:
            r = self._sample_to_row(t, s)
        else:
            r = s
            if self._sample_count and self._sample_count != n_rows:
                r = _downsample_map_index_inverse(s, n_rows, self._sample_count)
        return t, r

    def _sample_to_row(self, trace_display: float, sample_display: float) -> float:
        """采样点 → 海拔网格行（浮点）：行 = (最高海拔 − 该点海拔) / 行距。"""
        ground, depth, elev = (self._ground_elevation_m,
                               self._depth_axis_m, self._elev_axis)
        if ground is None or depth is None or elev is None or len(elev) < 2:
            return float(sample_display)
        ti = max(0, min(len(ground) - 1, int(round(trace_display))))
        si = max(0, min(len(depth) - 1, int(round(sample_display))))
        elevation = float(ground[ti]) - float(depth[si])
        top = float(elev[0])
        step = (top - float(elev[-1])) / (len(elev) - 1)
        if step <= 0:
            return 0.0
        return (top - elevation) / step

    def _on_mouse_clicked(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            if self._plot.sceneBoundingRect().contains(event.scenePos()):
                self._show_context_menu(event)
            return
        if not self._pick_enabled or self._image_shape is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        view_point = self._plot.vb.mapSceneToView(pos)
        trace, sample = int(view_point.x()), int(view_point.y())
        n_traces, n_samples = self._image_shape
        if 0 <= trace < n_traces and 0 <= sample < n_samples:
            # sig_point_picked 契约是原始数据坐标：预览可能降采样，
            # 后端会话（界面标注/双曲线拾取）全部在原始坐标系工作。
            data_trace, data_sample = self._view_to_data(trace, sample)
            self._emit_point_picked(data_trace, data_sample, view_trace=trace)

    def _emit_point_picked(self, trace: int, sample: int, *, view_trace: int | None = None) -> None:
        """统一 pick 发射口：跟随浮窗消费 + 原信号照常发出。

        trace/sample 为原始数据坐标（信号契约）；view_trace 是点击处的
        显示坐标道号——跟随浮窗画的波形取自当前预览矩阵，必须用显示
        索引取列，否则降采样预览下波形与十字线错位。
        """
        if (self._ascan_follow and self._ascan_popup is not None
                and self._image_shape is not None):
            import numpy as np

            # 波形取自未变形的预览矩阵：海拔模式下 ImageItem 里是 warp 后
            # 的列（纵向重采样过），不是原始 A-scan，弹窗必须用 _matrix。
            image = self._matrix
            if image is None:
                self.sig_point_picked.emit(trace, sample)
                return
            image = np.asarray(image)
            display_trace = self._image_shape[0] - 1 if view_trace is None else int(view_trace)
            if 0 <= display_trace < image.shape[1]:
                dist = (self._trace_axis_m[display_trace]
                        if self._trace_axis_m is not None
                        and display_trace < len(self._trace_axis_m) else None)
                self._ascan_popup.show_trace(
                    image[:, display_trace], trace_index=display_trace, distance_m=dist)
        self.sig_point_picked.emit(trace, sample)

    # ------------------------------------------------------------------ 其它
    def clear(self) -> None:
        self._image_item.clear()
        self._scatter.setData([])
        self._image_shape = None
        self._matrix = None
        self._showing_elevation = False
        self._elev_axis = None
        self._elev_cache = None
        self._hide_crosshair()
        # 空态引导浮层（评审 P0-1）：替代原"只剩坐标轴 + 图内标题文案"
        self._export_title = ''
        self._plot.setTitle('')
        self._refresh_axis_state()
        self._empty_overlay.setVisible(True)

    def apply_theme(self, dark: bool) -> None:
        """深色 bg 'k'/文字 'w'；浅色 bg 'w'/文字 'k'；轴/色标/悬浮钮/十字光标同步。

        轴 pen/textPen/标签/标题走 pg_view_base.style_plot_item 统一循环；
        悬浮钮等控件配色走 control_palette 单源。
        """
        self._dark = bool(dark)
        palette = control_palette(dark)
        self._glw.setBackground(palette['plot_bg'])
        style_plot_item(
            self._plot, dark,
            colorbar_axis=(self._colorbar.axis
                           if self._colorbar is not None else None))
        self._refresh_control_palette(dark)
        # 十字光标：深色主题黄 / 浅色主题深红（图像与白底上均醒目）
        crosshair_pen = pg.mkPen(
            '#ffe135' if dark else '#c8000a',
            style=Qt.PenStyle.DashLine, width=1)
        self._vline.setPen(crosshair_pen)
        self._hline.setPen(crosshair_pen)

    def _refresh_control_palette(self, dark: bool) -> None:
        """悬浮钮配色（control_palette 单源，主题切换时重刷）。"""
        palette = control_palette(dark)
        # 悬浮钮：紧凑尺寸保留，颜色随主题（硬编码浅色会在深色下突兀）
        btn_qss = (
            f'PushButton {{ background-color: {palette["button_bg"]}; '
            f'color: {palette["button_text"]}; '
            f'border: 1px solid {palette["border"]}; border-radius: 4px; '
            f'padding: 2px 8px; '
            f'font-size: {constants.FONT_SIZE_SECONDARY}pt; }}'
            f'PushButton:hover {{ background-color: {palette["hover"]}; }}'
        )
        self._fullscreen_btn.setStyleSheet(btn_qss)
