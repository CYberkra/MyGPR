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


from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QVBoxLayout, QWidget

import pyqtgraph as pg
from PyQt6.QtWidgets import QLabel
from qfluentwidgets import FluentIcon as FIF, ToolButton

from ui import constants
from ui.desktop_backend_facade import compute_display_levels  # noqa: F401  (历史路径，tests 可能经此导入)
from ui.theme_helpers import control_palette
from ui.widgets.bscan_axes import (
    X_AXIS_MODES,
    Y_AXIS_MODES,
    IndexAxis,
    distance_available,
    distance_tick_strings,
    elevation_tick_strings,
)
from ui.widgets.bscan_view_data import (BScanViewDataMixin, DEFAULT_P_LOW as DEFAULT_P_LOW, DEFAULT_P_HIGH as DEFAULT_P_HIGH, _restored_levels as _restored_levels)
from ui.widgets.bscan_view_fullscreen import BScanViewFullscreenMixin
from ui.widgets.bscan_view_interaction import (BScanDisplayMode,
                                               BScanViewInteractionMixin)
from ui.widgets.bscan_view_interaction import (format_crosshair_readout
                                               as format_crosshair_readout)
from ui.widgets.empty_state import EmptyStateOverlay
from ui.widgets.pg_view_base import GraphicsViewBase, style_plot_item



class BScanView(BScanViewFullscreenMixin, BScanViewDataMixin, BScanViewInteractionMixin, GraphicsViewBase, QWidget):
    """B-Scan 剖面图像视图。

    缩放/导出/轴主题继承 GraphicsViewBase（pg_view_base）；
    比例策略（方形/铺满/1:1）为本类专属；右键菜单/拾取/十字光标/坐标
    映射在 :class:`BScanViewInteractionMixin`（2026-09-24 组件化拆分），
    显示模式枚举亦随迁（本模块 re-export）。

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
        # 显式主题色：不写 color 时 QLabel 富文本用调色板文字色，实测在
        # 浅色主题下 GraphicsView 内的白字几乎不可见（视觉验收抓到）
        from qfluentwidgets import isDarkTheme
        color = '#e0e0e0' if isDarkTheme() else '#333333'
        return f'<span style="font-size:8pt;color:{color}">{text}</span>'

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
