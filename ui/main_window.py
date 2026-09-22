# -*- coding: utf-8 -*-
"""MyGPRMainWindow(FluentWindow) 组装器（SPEC §6.1，style_spec §2 复刻）。

主窗口只做「创建 → 布局 → connect」：
    _init_window → _create_controllers → _create_pages
    → _build_ui → _connect_signals → _init_state

页面 / 控制器 / OutputPanel 均以 try/except 导入：缺失时页面降级为
PlaceholderPage（QLabel '页面建设中' 居中），controller / 面板为 None（connect 判空）。
"""
import logging

from PyQt6.QtCore import (
    Qt, QSize, QTimer, pyqtSignal,
)
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QHBoxLayout, QLabel, QPushButton, QSplitter,
    QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    FluentIcon as FIF, FluentWindow, InfoBar, InfoBarPosition,
    LineEdit, NavigationItemPosition, PrimaryPushButton, PushButton,
    SplashScreen, TransparentToolButton, isDarkTheme,
)

from ui import constants, file_dialogs
from ui.page_coordinator import PageCoordinator
from ui.logger_config import setup_logger
from ui.settings_manager import SettingsManager
from ui.theme_helpers import apply_theme, control_palette
from ui.widgets.bscan_container import BScanContainer
from ui.widgets.bscan_view import BScanView
from ui.widgets.file_tree_panel import FileTreePanel
from ui.widgets.segment_tabs import SlimSegment

logger = setup_logger('mygpr_window', 'logs/mygpr_window.log', level=logging.DEBUG)

# 上格宿主最小高：splitter 分配下限（页签条 36 + 页面区保底 ~260）。
# 不覆写时 minimumSizeHint 透传 QStackedWidget 的「所有页最大 min」
# （实测 835px > splitter 总高），QSplitter 永远把下格 OutputPanel 压到
# 自身最小——表现为输出面板拖不动/一拖就消失、页面区视觉补位；且布局
# 常驻溢出态，任何页面数据刷新引发的 min 抖动都会造成整窗闪动。
_CONTENT_HOST_MIN_H = 300


class _PageHostWidget(QWidget):
    """内容行宿主：min hint 不透传页面栈全页最大值（不可见页不钳制布局）。"""

    def minimumSizeHint(self):   # Qt 虚函数命名（CamelCase 是 Qt 约定，非本仓风格）
        return QSize(0, _CONTENT_HOST_MIN_H)


# ------------------------------------------------------------ 页面（[A4]/[A5] 提供，缺失降级占位）
def _import_page_class(module_path: str, class_name: str):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as e:  # ImportError 及其他导入期异常 → 占位页
        logger.warning('页面 %s.%s 导入失败，使用占位页: %s', module_path, class_name, e)
        return None


HomePage = _import_page_class('ui.pages.home_page', 'HomePage')
ProjectPage = _import_page_class('ui.pages.project_page', 'ProjectPage')
ProcessingPage = _import_page_class('ui.pages.processing_page', 'ProcessingPage')
InterpretationPage = _import_page_class('ui.pages.interpretation_page', 'InterpretationPage')
SpatialPage = _import_page_class('ui.pages.spatial_page', 'SpatialPage')
DeliveryPage = _import_page_class('ui.pages.delivery_page', 'DeliveryPage')
JobsPage = _import_page_class('ui.pages.jobs_page', 'JobsPage')
SettingsPage = _import_page_class('ui.pages.settings_page', 'SettingsPage')

# ------------------------------------------------------------ 控制器（[A3] 提供，缺失为 None）
def _import_controller_class(class_name: str):
    return _import_page_class(f'ui.controllers.{_camel_to_snake(class_name)}', class_name)


def _camel_to_snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append('_')
        out.append(ch.lower())
    return ''.join(out)


BackendController = _import_controller_class('BackendController')
ProjectController = _import_controller_class('ProjectController')
ProcessingController = _import_controller_class('ProcessingController')
InterpretationController = _import_controller_class('InterpretationController')
DeliveryController = _import_controller_class('DeliveryController')

# ------------------------------------------------------------ OutputPanel（底部输出面板，缺失为 None）
OutputPanel = _import_page_class('ui.widgets.output_panel', 'OutputPanel')


class PlaceholderPage(QWidget):
    """占位页：QLabel '页面建设中' 居中（A1 骨架阶段）。"""

    def __init__(self, title: str = '', parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel('页面建设中', self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        if title:
            self.setToolTip(title)


class MyGPRMainWindow(FluentWindow):
    """主窗口组装器（SPEC §6.1）。"""

    _log_signal = pyqtSignal(str)   # 类级信号：跨线程日志安全转发

    # 首屏构造的页面（其余延后到首帧渲染后空闲分片预热，见 _warmup_next_page）。
    # home = 初始可见页；settings = 接线/状态回放必需（_connect_signals 取
    # theme_changed、_init_state 回放开局设置）。
    FIRST_PAINT_PAGES = ('homeInterface', 'settingsInterface')

    def __init__(self, settings: SettingsManager = None, parent=None):
        super().__init__(parent)
        self.settings = settings or SettingsManager()
        self.spacing = constants.PAGE_SPACING
        self._restoring_settings = False
        self.pages = {}             # objectName -> page widget

        # ---- 窗口自身状态 ----
        self._backend_ready = False
        self._backend_error_bar = None      # 后端初始化失败的常驻错误横幅

        self._init_window()
        self._create_controllers()
        self._create_pages()
        # 跨页业务信号链与运行态统一由 PageCoordinator 持有（任务 F 候选 1）
        self.page_coordinator = PageCoordinator(self)
        self._build_ui()
        self._setup_global_shortcuts()
        self._connect_signals()
        self._init_state()
        # 注入设置值到页面（设置页更改后下次启动生效）——依赖 project/spatial，
        # 二者属延后页，实际注入发生在预热收尾 _finish_warmup()。
        # 首屏后空闲分片构造剩余页面（启动提速：把 6 个非首屏页的构造挪出首帧）
        self._start_warmup()

    def _inject_page_settings(self) -> None:
        """注入设置页保存的默认值到各页面控件（非信号驱动，首次启动 / 重启生效）。"""
        project = self._page('projectInterface')
        if hasattr(project, 'set_default_dielectric'):
            project.set_default_dielectric(
                self.settings.get('default_dielectric', constants.DEFAULT_DIELECTRIC))
        spatial = self._page('spatialInterface')
        if hasattr(spatial, 'set_auto_prefetch_enabled'):
            spatial.set_auto_prefetch_enabled(
                bool(self.settings.get('auto_prefetch_basemap', True)))

    # ============================================================ 组装
    def _init_window(self) -> None:
        self.setWindowTitle(constants.APP_NAME)
        # 切页零动画：库默认 PopUpAni 是新页从 76px 下方滑入（"上下跳变"）；
        # 透明度淡入方案对重页面（B-Scan/表格）要整页离屏渲染，切换时卡顿。
        # 苹果系专业工具（System Settings/Xcode）与 VS Code 均为瞬时切换。
        self.stackedWidget.setAnimationEnabled(False)
        # 禁用 Mica 亚克力背景：Win11 24H2 上浅色 Mica Backdrop 渲染失效，
        # 会导致标题栏/导航栏整片透明、透出桌面（浅色主题"基本不可用"的根因）。
        # 禁用后窗口自行绘制实色背景（浅 #f0f4f9 / 深 #202020），
        # 深浅主题都能稳定渲染，且在 Win10/Win11 各版本表现一致。
        self.setMicaEffectEnabled(False)
        # 屏幕自适应：目标 1450×850，但不超过可用桌面的 92%；最小尺寸同样
        # 受屏幕约束，避免小屏/高缩放机器上窗口比屏幕大、按钮被挤出可视区。
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        if available is not None and available.width() > 0:
            max_w = int(available.width() * 0.92)
            max_h = int(available.height() * 0.92)
        else:
            max_w, max_h = constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT
        self.resize(min(constants.WINDOW_WIDTH, max_w),
                    min(constants.WINDOW_HEIGHT, max_h))
        self.setMinimumSize(min(constants.WINDOW_MIN_WIDTH, max_w),
                            min(constants.WINDOW_MIN_HEIGHT, max_h))
        # Windows 默认级联放置会把新窗口不断往右下推；窗口接近屏宽时
        # 右缘会伸出屏幕，屏外部分不渲染（整块黑），看起来像"界面被压缩"。
        # 显式居中到可用桌面，保证窗口完整落在屏内。
        if available is not None and available.width() > 0:
            frame = self.frameGeometry()
            frame.moveCenter(available.center())
            self.move(frame.topLeft())
        self.setWindowIcon(QIcon(constants.APP_ICON_PATH))
        # 竖导航已隐藏（顶部横排 Pivot 页签替代），不再配置展开宽度

        # 开屏画面：图标 256×256，删除关闭按钮，600ms 后关闭
        self.splashScreen = SplashScreen(QIcon(constants.APP_ICON_PATH), self)
        self.splashScreen.titleBar.closeBtn.hide()
        self.splashScreen.show()
        QTimer.singleShot(constants.SPLASH_DURATION_MS, self.splashScreen.close)

    def _create_controllers(self) -> None:
        """控制器缺失（A3 未交付）时为 None，connect 时判空。"""
        self.backend_controller = BackendController(self) if BackendController else None
        self.project_controller = ProjectController(self) if ProjectController else None
        self.processing_controller = ProcessingController(self) if ProcessingController else None
        self.interpretation_controller = (
            InterpretationController(self) if InterpretationController else None)
        self.delivery_controller = DeliveryController(self) if DeliveryController else None

        # 注入 backend
        if self.backend_controller is not None:
            for ctrl in (self.project_controller, self.processing_controller,
                         self.interpretation_controller, self.delivery_controller):
                if ctrl is not None and hasattr(ctrl, 'set_backend'):
                    ctrl.set_backend(self.backend_controller)

    def _create_pages(self) -> None:
        page_specs = [
            # objectName, 页面类, 图标, 文本, position
            ('homeInterface', HomePage, FIF.HOME, '主页', NavigationItemPosition.TOP),
            ('projectInterface', ProjectPage, FIF.FOLDER, '项目', NavigationItemPosition.TOP),
            ('processingInterface', ProcessingPage, FIF.DEVELOPER_TOOLS, '处理',
             NavigationItemPosition.TOP),
            ('interpretationInterface', InterpretationPage, FIF.EDIT, '解释',
             NavigationItemPosition.TOP),
            ('spatialInterface', SpatialPage, FIF.GLOBE, '空间信息',
             NavigationItemPosition.TOP),
            ('deliveryInterface', DeliveryPage, FIF.SEND, '成果', NavigationItemPosition.TOP),
            ('jobsInterface', JobsPage, FIF.SYNC, '任务', NavigationItemPosition.TOP),
            ('settingsInterface', SettingsPage, FIF.SETTING, '设置',
             NavigationItemPosition.BOTTOM),
        ]
        self._nav_specs = [(name, text) for name, _cls, _icon, text, _pos in page_specs]
        # 首屏只构造 FIRST_PAINT_PAGES；其余登记到 _deferred_specs，首帧后
        # 由 QTimer.singleShot(0) 每帧构造一个（见 _warmup_next_page）。
        self._deferred_specs: list = []
        self._pages_warmed = False
        for spec in page_specs:
            if spec[0] in self.FIRST_PAINT_PAGES:
                self._register_page(*spec)
            else:
                self._deferred_specs.append(spec)

    def _register_page(self, object_name, page_class, icon, text, position) -> None:
        """构造单个页面并注册路由 / 注入共享 SettingsManager。"""
        page = page_class(self) if page_class else PlaceholderPage(text, self)
        # 注入共享 SettingsManager：页面不再各自构造实例，避免读-改-写
        # 互相覆盖（共享实例是唯一写者）
        if hasattr(page, 'set_settings_manager'):
            page.set_settings_manager(self.settings)
        # B-Scan 显示偏好（比例/轴单位/色阶/全屏几何）：四页都有 BScanView，
        # 统一在此恢复与持久化，避免在四个页面里各写一遍（漏一个就是"某页记不住"）
        self._wire_bscan_preferences(page)
        page.setObjectName(object_name)
        self.addSubInterface(page, icon, text, position=position)
        self.pages[object_name] = page

    # ---------------------------------------------------------- B-Scan 显示偏好
    def _wire_bscan_preferences(self, page) -> None:
        """把页面内所有 BScanView 的显示偏好接到持久化设置（跨会话记住）。

        覆盖：比例模式 / 横纵轴单位 / 色阶百分位 / 全屏窗口几何。恢复阶段
        一律 ``notify=False``，否则形成「读设置 → 写设置」回环。
        """
        views = page.findChildren(BScanView)
        if not views:
            return
        aspect = self._setting_choice('bscan_aspect_mode', ('free', 'square', 'cell'), 'free')
        x_axis = self._setting_choice('bscan_x_axis', ('trace', 'distance'), 'trace')
        y_axis = self._setting_choice('bscan_y_axis', ('sample', 'elevation'), 'sample')
        p_low, p_high = self._setting_levels()
        for view in views:
            view.set_aspect_mode(aspect, notify=False)
            view.set_axis_modes(x_axis, y_axis, notify=False)
            view.set_display_levels(p_low, p_high, notify=False)
            view.set_geometry_store(loader=self._load_fullscreen_geometry,
                                    saver=self._save_fullscreen_geometry)
            self._connect_bscan_signals(view)
            self._wire_bscan_expand(view)
        # 容器级偏好：预览布局（面板集合随布局变，轴/比例/色阶已逐面板恢复）
        layout_mode = self._setting_choice(
            'bscan_layout_mode', ('auto', 'single', 'dual', 'quad'), 'auto')
        for container in self._iter_bscan_containers(page):
            container.set_layout_mode(layout_mode, notify=False)
            container.sig_layout_changed.connect(lambda mode:
                self._mirror_bscan_setting('bscan_layout_mode', str(mode)))

    @staticmethod
    def _iter_bscan_containers(page):
        """页面里全部 BScanContainer（哑组件，偏好按容器整体恢复/持久化）。"""
        yield from page.findChildren(BScanContainer)

    def _connect_bscan_signals(self, view) -> None:
        """把用户操作（而非恢复动作）接到设置写盘。

        ⚠️ 必须**同时回写设置页控件**（``_mirror_bscan_setting``）：设置页的
        ComboBox/SpinBox 是 B-Scan 偏好的第二份副本，而 ``closeEvent`` 会把
        ``settings_page.settings()`` 整体回写设置文件。若工具条改了偏好却不
        同步到设置页，关窗时就会被那份**过期的默认值**覆盖掉——表现为「在
        工具条切了方形，重开又变回铺满」。这是排查过的真实缺陷，别只写盘。
        """
        view.sig_aspect_changed.connect(
            lambda mode: self._mirror_bscan_setting('bscan_aspect_mode', str(mode)))
        view.sig_x_axis_changed.connect(
            lambda mode: self._mirror_bscan_setting('bscan_x_axis', str(mode)))
        view.sig_y_axis_changed.connect(
            lambda mode: self._mirror_bscan_setting('bscan_y_axis', str(mode)))
        view.sig_levels_changed.connect(
            lambda low, high: self._mirror_bscan_levels(low, high))

    def _mirror_bscan_setting(self, key: str, value) -> None:
        """写盘 + 把设置页控件同步到同一值（避免关窗回写覆盖用户选择）。"""
        self._persist_setting(key, value)
        self._sync_settings_page_bscan({key: value})

    def _mirror_bscan_levels(self, low: float, high: float) -> None:
        """色阶两键一并写盘与同步（避免中途崩溃留下 low > high）。"""
        self._persist_bscan_levels(low, high)
        self._sync_settings_page_bscan({
            'bscan_p_low': float(low), 'bscan_p_high': float(high)})

    def _sync_settings_page_bscan(self, values: dict) -> None:
        """把 B-Scan 偏好同步进设置页控件（blockSignals 防信号回环）。"""
        settings_page = self._page('settingsInterface')
        syncer = getattr(settings_page, 'sync_bscan_view_settings', None)
        if callable(syncer):
            syncer(values)

    def _wire_bscan_expand(self, view) -> None:
        """「⤢ 铺满」：只有能折叠侧栏的页面才露出该钮，并接上折叠动作。"""
        toggler = getattr(view.parent(), 'toggle_side_panels', None)
        host = view.parent()
        while host is not None and not callable(toggler):
            host = host.parent()
            toggler = getattr(host, 'toggle_side_panels', None)
        if not callable(toggler):
            return
        view.set_expand_enabled(True)
        view.sig_expand_requested.connect(toggler)

    def _setting_choice(self, key: str, allowed: tuple, fallback: str) -> str:
        """读枚举型设置：越界/损坏一律回落默认（不让坏设置影响出图）。"""
        value = str(self.settings.get(key, fallback) or fallback)
        return value if value in allowed else fallback

    def _iter_bscan_views(self):
        """本会话所有 BScanView（含尚未构造的延迟页之外的已建页面）。"""
        for page in list(self.pages.values()):
            yield from page.findChildren(BScanView)

    def _iter_all_bscan_containers(self):
        """全部页面里的 BScanContainer（设置页统一下发布局模式用）。"""
        for page in list(self.pages.values()):
            yield from page.findChildren(BScanContainer)

    def _on_settings_bscan_changed(self) -> None:
        """设置页改了 B-Scan 视图项：即时下发到所有视图并写盘（无需重启）。"""
        settings_page = self._page('settingsInterface')
        getter = getattr(settings_page, 'bscan_view_settings', None)
        if not callable(getter):
            return
        values = getter()
        for key, value in values.items():
            self.settings.set(key, value)
        self.settings.save()
        for view in self._iter_bscan_views():
            view.set_aspect_mode(values['bscan_aspect_mode'], notify=False)
            view.set_axis_modes(values['bscan_x_axis'], values['bscan_y_axis'],
                                notify=False)
            view.set_display_levels(values['bscan_p_low'],
                                    values['bscan_p_high'], notify=False)
        for container in self._iter_all_bscan_containers():
            # notify=True：切到/切出 auto 时页面要重分发数据。由此触发的
            # sig_layout_changed 会把同值写回设置（幂等），无行为副作用。
            container.set_layout_mode(values['bscan_layout_mode'], notify=True)

    def _setting_levels(self) -> tuple[float, float]:
        """读色阶百分位；非法值回落 BScanView 默认（2 / 98）。"""
        try:
            low = float(self.settings.get('bscan_p_low', 2.0))
            high = float(self.settings.get('bscan_p_high', 98.0))
        except (TypeError, ValueError):
            return 2.0, 98.0
        if not 0.0 <= low < high <= 100.0:
            return 2.0, 98.0
        return low, high

    def _persist_setting(self, key: str, value) -> None:
        """用户操作触发的一次写盘（共享 SettingsManager 唯一写者）。"""
        self.settings.set(key, value)
        self.settings.save()

    def _persist_bscan_levels(self, low: float, high: float) -> None:
        """色阶两个键必须同一次写盘，否则中途崩溃会留下 low>high 的坏状态。"""
        self.settings.set('bscan_p_low', float(low))
        self.settings.set('bscan_p_high', float(high))
        self.settings.save()

    def _load_fullscreen_geometry(self):
        """回放 B-Scan 全屏窗口几何；从未存过（空串）返回 None 走默认尺寸。"""
        from PyQt6.QtCore import QByteArray
        raw = self.settings.get('bscan_fullscreen_geometry', '')
        if not raw:
            return None
        try:
            return QByteArray.fromBase64(str(raw).encode('ascii'))
        except (ValueError, UnicodeEncodeError):
            return None

    def _save_fullscreen_geometry(self, blob) -> None:
        """把 QByteArray 几何转 base64 存入 JSON 设置（空几何不覆盖旧值）。"""
        if blob is None or blob.isEmpty():
            return
        payload = bytes(blob.toBase64().data()).decode('ascii')
        self._persist_setting('bscan_fullscreen_geometry', payload)

    # ---------------------------------------------------------- 首屏后预热
    def _start_warmup(self) -> None:
        """首帧渲染后开始分片构造剩余页面（每帧一个，避免一次性长阻塞）。"""
        QTimer.singleShot(0, self._warmup_next_page)

    def _warmup_next_page(self) -> None:
        if not self._deferred_specs:
            self._finish_warmup()
            return
        self._register_page(*self._deferred_specs.pop(0))
        QTimer.singleShot(0, self._warmup_next_page)

    def _finish_warmup(self) -> None:
        """全部页面就位后的收尾（幂等）：跨页接线 + 设置注入 + 后端门控补齐。"""
        if self._pages_warmed:
            return
        self._pages_warmed = True
        # 清空预热队列：ensure_pages_ready 已同步构造完所有页，此时若还有
        # QTimer.singleShot(0, _warmup_next_page) 留在事件队列里，它会在下面
        # 遍历 pages 期间的嵌套事件循环中执行 _register_page → self.pages[...]
        # 写入，触发「dictionary changed size during iteration」（CI Linux
        # offscreen 上表现为 test_main_window_titlebar 偶发 ERROR）。
        self._deferred_specs.clear()
        # 跨页业务信号链需要 8 页全部存在，因此延后到此处（接线代码本身不改）
        self.page_coordinator.connect_all()
        # 后端先于本收尾就绪的场景：_on_backend_ready 当时因接线未完成而跳过
        # load_methods（methods_loaded 发出时还没有接收者）→ 这里补加载。
        # 反向时序（后端后就绪）由 _on_backend_ready 正常加载，二者恰好互斥不重复。
        if self._backend_ready and self.processing_controller is not None:
            self.processing_controller.load_methods()
        self._inject_page_settings()
        if not self._backend_ready:
            # 预热期间后端可能仍未就绪：补齐延后页的禁用态
            # 遍历用 list() 快照：setEnabled 会走 Qt 事件/布局，理论上可能
            # 重入本方法或被其它路径改 pages；快照让遍历对字典变更免疫。
            for object_name, page in list(self.pages.items()):
                if object_name not in self.FIRST_PAINT_PAGES:
                    page.setEnabled(False)

    def ensure_pages_ready(self) -> None:
        """同步构造全部剩余页面并完成收尾接线（预热竞态兜底 / 测试 / 冒烟）。"""
        while self._deferred_specs:
            self._register_page(*self._deferred_specs.pop(0))
        self._finish_warmup()

    def _build_ui(self) -> None:
        """顶部横排页签 + 左坞文件树 + 底部输出面板（OutputPanel：日志/任务）。

        根布局由 FluentWindow 的 ``导航 | widgetLayout`` 两层改为
        ``导航(隐藏) | 右列[ 页签条 / 内容行(widgetLayout: 文件树|页面) / OutputPanel ]``。
        """
        # 竖导航退役：隐藏 FluentWindow 自带 NavigationInterface。addSubInterface
        # 仍经它注册路由（_onCurrentInterfaceChanged/qrouter 保持自洽），
        # switchTo 只动 stackedWidget，不依赖导航可见性；页签改用顶部横排
        # Pivot（qfluentwidgets 现成 Fluent 组件），左列整列让给文件树。
        self.navigationInterface.hide()

        # 底部输出面板：日志/任务（常驻头部栏 + 可收展内容区，状态记忆）
        self.output_panel = OutputPanel(self) if OutputPanel else None
        if self.output_panel is not None and hasattr(self.output_panel, 'set_settings_manager'):
            self.output_panel.set_settings_manager(self.settings)

        # 右列容器：页签条（内含标题栏 48px 净空）/ 竖向 QSplitter[
        # 内容行(widgetLayout: 文件树|页面，stretch 1) / 输出面板（高度可拖）]
        self._right_area = QWidget(self)
        right_col = QVBoxLayout(self._right_area)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(0)
        self.hBoxLayout.removeItem(self.widgetLayout)
        right_col.addWidget(self._create_top_nav_bar())
        # 标题栏净空由页签条承担，内容行不再预留顶部 48px
        self.widgetLayout.setContentsMargins(0, 0, 0, 0)
        # QSplitter 只收 widget，把内容行 layout 包进宿主 widget
        # （_PageHostWidget：min hint 固定 300，见类注释——透传全页最大
        # min 会导致输出面板被 QSplitter 压死，问题 1/2 同根因）
        content_host = _PageHostWidget(self)
        content_host.setMinimumHeight(_CONTENT_HOST_MIN_H)
        host_col = QVBoxLayout(content_host)
        host_col.setContentsMargins(0, 0, 0, 0)
        host_col.setSpacing(0)
        host_col.addLayout(self.widgetLayout, 1)
        self._v_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._v_splitter.setChildrenCollapsible(False)
        self._v_splitter.addWidget(content_host)
        if self.output_panel is not None:
            self._v_splitter.addWidget(self.output_panel)
            self._v_splitter.setStretchFactor(0, 1)
            self._v_splitter.setStretchFactor(1, 0)
            self.output_panel.sig_open_toggled.connect(
                self._on_output_open_toggled)
            self._v_splitter.splitterMoved.connect(
                self._on_output_splitter_moved)
            # 几何就绪后恢复上次拖出的面板高度
            QTimer.singleShot(0, self._restore_output_panel_height)
        right_col.addWidget(self._v_splitter, 1)
        self.hBoxLayout.addWidget(self._right_area, 1)
        # _right_area 是有实体的全高 widget（顶部 48px 标题栏留白由
        # widgetLayout 的 margin 承担，不在它身上），又创建于 titleBar 之后，
        # z-order 压过标题栏右半——最小化/最大化/关闭按钮的鼠标事件会被它
        # 吞掉（曾表现为"三个按钮点了没反应"）。标题栏必须重新抬到最顶层。
        self.titleBar.raise_()

        # 左侧常驻文件树（页签条 | 文件树 | 页面），面板常驻；
        # 未打开项目时显示空态而不是整树消失（入口通用性）。
        # 注意插 widgetLayout（页签条已承担标题栏净空），不是 hBoxLayout
        # （后者从窗口最顶开始，会把面板顶进标题栏）。
        # 宽度由面板自管（展开 232 / 细条 18），切页感知经 stackedWidget.currentChanged
        self._file_tree_panel = FileTreePanel(self)
        self._file_tree_panel.set_settings_manager(self.settings)
        self.widgetLayout.insertWidget(0, self._file_tree_panel, 0)
        self.stackedWidget.currentChanged.connect(self._on_page_switched)
        # currentChanged 不为"启动即所在的初始页"触发，补一次初始页应用，
        # 否则初始页（主页）在打开项目前一直停留在面板默认的展开态
        self._on_page_switched(self.stackedWidget.currentIndex())

    def _create_top_nav_bar(self) -> QWidget:
        """顶部横排页签条：SlimSegment 药丸分段（与日志/任务、处理页预览同款控件）
        套圆角轨道，外加标题栏净空。

        与 Pivot 同属一套信号语义（SegmentedWidget 继承 Pivot）：
        setCurrentItem 与 currentItemChanged 双向同步 stack 页面——点击页签
        → switchTo；程序化 switchTo（快捷键/主页快捷操作/冒烟脚本）
        → _on_page_switched 回写选中态。两侧 setter 均在状态不变时提前返回，
        信号环自然终止。
        """
        bar = QWidget(self)
        bar.setObjectName('topNavBar')
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 48, 16, 0)
        bar.setFixedHeight(48 + constants.NAV_TOP_BAR_HEIGHT)
        self._top_nav_bar = bar
        # 药丸轨道：圆角浅底容器把 8 个分段收拢成一个控件（示意图方案 B）
        track = QWidget(bar)
        track.setObjectName('topNavTrack')
        track_layout = QHBoxLayout(track)
        track_layout.setContentsMargins(4, 4, 4, 4)
        track_layout.setSpacing(0)
        self._top_nav_track = track
        self._top_nav = SlimSegment(track)
        for object_name, text in self._nav_specs:
            self._top_nav.addItem(object_name, text)
        self._top_nav.currentItemChanged.connect(self._on_top_nav_changed)
        track_layout.addWidget(self._top_nav)
        # 药丸居中（评审 P1-5）：两侧对称 stretch，消除"重心偏左、
        # 右半条空白"的失衡感
        layout.addStretch(1)
        layout.addWidget(track, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch(1)
        # 右端双开关：文件树 (Ctrl+B) / 输出面板 (Ctrl+J)，快捷键入口对等曝光
        self._file_tree_btn = TransparentToolButton(FIF.LAYOUT, bar)
        self._file_tree_btn.setIconSize(QSize(*constants.TOOL_BTN_ICON))
        self._file_tree_btn.setToolTip('文件树 (Ctrl+B)')
        self._file_tree_btn.setFixedSize(*constants.TOOL_BTN_SIZE)
        self._file_tree_btn.clicked.connect(self._toggle_file_tree)
        layout.addWidget(self._file_tree_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self._output_btn = TransparentToolButton(FIF.CODE, bar)
        self._output_btn.setIconSize(QSize(*constants.TOOL_BTN_ICON))
        self._output_btn.setToolTip('输出面板：日志 / 任务 (Ctrl+J)')
        self._output_btn.setFixedSize(*constants.TOOL_BTN_SIZE)
        if self.output_panel is not None:
            self._output_btn.clicked.connect(self.output_panel.toggle_panel)
        layout.addWidget(self._output_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self._apply_top_nav_style()
        # 初始选中第一个页面（与 addSubInterface 的初始 stack 页一致）
        if self._nav_specs:
            self._top_nav.setCurrentItem(self._nav_specs[0][0])
        return bar

    def _on_top_nav_changed(self, route_key: str) -> None:
        page = self.pages.get(str(route_key))
        if page is None and self._deferred_specs:
            # 预热未完成就点页签：同步构造（与 _goto_page 同一兜底）
            self.ensure_pages_ready()
            page = self.pages.get(str(route_key))
        if page is not None and self.stackedWidget.currentWidget() is not page:
            self.switchTo(page)

    def _apply_top_nav_style(self) -> None:
        """页签条底部分隔线 + 药丸轨道底色，深浅主题跟随（control_palette 单源）。"""
        pal = control_palette(isDarkTheme())
        line = pal['nav_line']
        track = pal['nav_track']
        if getattr(self, '_top_nav_bar', None) is not None:
            self._top_nav_bar.setStyleSheet(
                f'#topNavBar {{ border-bottom: 1px solid {line}; }}'
                f'#topNavTrack {{ background: {track}; border-radius: 15px; }}')

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        # FluentWindow.resizeEvent 为竖导航收起钮把标题栏右移 46px；竖导航
        # 已隐藏，标题栏拉满全宽（横排页签在其下方行，不重叠）。
        self.titleBar.move(0, 0)
        self.titleBar.resize(self.width(), self.titleBar.height())

    # ------------------------------------------------------------ 输出面板高度（竖向 QSplitter）
    def _output_panel_target_height(self) -> int:
        """展开态目标高度：用户上次拖出的值（settings），否则常量默认。"""
        default = (constants.OUTPUT_PANEL_HEIGHT
                   + constants.OUTPUT_PANEL_HEADER_HEIGHT)
        saved = self.settings.get('output_panel_height') if self.settings else None
        try:
            return max(80, int(saved)) if saved else default
        except (TypeError, ValueError):
            return default

    def _set_output_panel_height(self, target: int) -> None:
        """把 splitter 下格调整到 target，上格吃掉余量（保下限防挤没）。

        收/展会 ``setVisible`` 切换下格 min hint（37 ↔ 97），其触发的
        child invalidate 会在事件循环里让 QSplitter 按 stretch 重算、
        把这里的显式分配洗掉（下格 stretch=0 被压回 min）——故事件循环
        排空后再重申一次分配。
        """
        total = sum(self._v_splitter.sizes()) or self._v_splitter.height()
        if total <= 0:
            return
        target = min(target, max(120, total - 200))
        sizes = [max(200, total - target), target]
        self._v_splitter.setSizes(sizes)
        QTimer.singleShot(0, lambda: self._v_splitter.setSizes(sizes))

    def _restore_output_panel_height(self) -> None:
        """启动恢复：按开合态把 splitter 下格设为展开高/头部栏高。"""
        if self.output_panel is None:
            return
        if sum(self._v_splitter.sizes()) <= 0:
            # 构造函数末尾几何尚未分配，事件循环起来后重试一次
            QTimer.singleShot(100, self._restore_output_panel_height)
            return
        if self.output_panel.is_open():
            self._set_output_panel_height(self._output_panel_target_height())
        else:
            self._set_output_panel_height(
                constants.OUTPUT_PANEL_HEADER_HEIGHT + 2)

    def _on_output_open_toggled(self, open_: bool) -> None:
        """输出面板收/展：下格在「展开高 ↔ 仅头部栏」间切换。"""
        if open_:
            self._set_output_panel_height(self._output_panel_target_height())
        else:
            self._set_output_panel_height(
                constants.OUTPUT_PANEL_HEADER_HEIGHT + 2)

    def _on_output_splitter_moved(self, _pos: int, _index: int) -> None:
        """用户拖拽改变面板高度 → 记忆（仅展开态；收起时下格是头部栏高）。

        收起态拖 handle = 用户想要更高面板：内容区恢复显示，高度以用户
        拖出的为准（本面板 min hint 恒定，显隐切换不会洗掉当前分配）。
        """
        if self.output_panel is None:
            return
        if not self.output_panel.is_open():
            if self._v_splitter.sizes()[1] > constants.OUTPUT_PANEL_HEADER_HEIGHT + 8:
                self.output_panel.set_open(True)
            return
        if self.settings is not None:
            self.settings.set('output_panel_height',
                              self._v_splitter.sizes()[1])
            self.settings.save()

    def _toggle_file_tree(self) -> None:
        """页签条右端钮 / Ctrl+B：收/展文件树（面板按页记忆持久化）。"""
        panel = getattr(self, '_file_tree_panel', None)
        if panel is not None:
            panel.toggle_panel()

    def _page(self, object_name: str):
        """按 objectName 取页面（占位页返回原样，调用方自行判接口）。"""
        return self.pages.get(object_name)

    def _setup_global_shortcuts(self) -> None:
        """全局快捷键：Ctrl+1~8 切页签，Ctrl+B 文件树，Ctrl+J 输出面板，F1 清单。"""
        self._page_shortcuts = [
            ('Ctrl+1', 'homeInterface', '主页'),
            ('Ctrl+2', 'projectInterface', '项目'),
            ('Ctrl+3', 'processingInterface', '处理'),
            ('Ctrl+4', 'interpretationInterface', '解释'),
            ('Ctrl+5', 'spatialInterface', '空间信息'),
            ('Ctrl+6', 'deliveryInterface', '成果'),
            ('Ctrl+7', 'jobsInterface', '任务'),
            ('Ctrl+8', 'settingsInterface', '设置'),
        ]
        for seq, obj_name, _title in self._page_shortcuts:
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(lambda checked=False, on=obj_name: self._goto_page(on))
        if self.output_panel is not None:
            self._panel_toggle = QShortcut(QKeySequence('Ctrl+J'), self)
            self._panel_toggle.setContext(Qt.ShortcutContext.WindowShortcut)
            self._panel_toggle.activated.connect(self.output_panel.toggle_panel)
        self._file_tree_toggle = QShortcut(QKeySequence('Ctrl+B'), self)
        self._file_tree_toggle.setContext(Qt.ShortcutContext.WindowShortcut)
        self._file_tree_toggle.activated.connect(self._toggle_file_tree)
        self._shortcuts_help = QShortcut(QKeySequence("F1"), self)
        self._shortcuts_help.setContext(Qt.ShortcutContext.WindowShortcut)
        self._shortcuts_help.activated.connect(self._show_shortcuts_dialog)

    def _connect_signals(self) -> None:
        """窗口自身接线 + 委托 PageCoordinator 完成全量业务接线（SPEC §7）。"""
        self._log_signal.connect(self._on_log_message)

        settings_page = self._page('settingsInterface')

        # ---------------- 控制器日志转发（三级通道之一）
        controllers = (self.backend_controller, self.project_controller,
                       self.processing_controller, self.interpretation_controller,
                       self.delivery_controller)
        for ctrl in controllers:
            if ctrl is not None and hasattr(ctrl, 'log_message'):
                ctrl.log_message.connect(self._log_signal)

        # ---------------- 后端生命周期
        if self.backend_controller is not None:
            if hasattr(self.backend_controller, 'backend_ready'):
                self.backend_controller.backend_ready.connect(self._on_backend_ready)
            if hasattr(self.backend_controller, 'backend_failed'):
                self.backend_controller.backend_failed.connect(self._on_backend_failed)

        # ---------------- 设置页（SPEC §6.4）
        if hasattr(settings_page, 'theme_changed'):
            settings_page.theme_changed.connect(self._on_theme_changed)
        if hasattr(settings_page, 'bscan_view_changed'):
            settings_page.bscan_view_changed.connect(
                self._on_settings_bscan_changed)

        # ---------------- 跨页业务信号链（项目/测线/导入/处理/解释/成果/任务）
        # 延后到预热收尾：connect_all() 需要 8 页全部存在（见 _finish_warmup）

    def _init_state(self) -> None:
        # 恢复主题设置（回放期间抑制副作用）
        self._restoring_settings = True
        try:
            self._on_theme_changed(self.settings.get('theme', constants.THEME_LIGHT))
        finally:
            self._restoring_settings = False

        # 回放设置到设置页控件（blockSignals，不触发 theme_changed）
        settings_page = self._page('settingsInterface')
        if hasattr(settings_page, 'load_settings'):
            settings_page.load_settings(self.settings.get_all())

        # 后端就绪门控：除主页/设置外页面禁用，backend_ready 后 enable
        if self.backend_controller is not None:
            self._set_backend_ready(False)
            self.backend_controller.start(
                max_workers=int(self.settings.get('max_workers', constants.MAX_WORKERS)))
            self.log_message('INFO 后端初始化中…')
        else:
            self.log_message('WARNING BackendController 未就绪（骨架阶段），全部页面保持启用')

        self.log_message('INFO MyGPR 探地雷达数据处理软件启动成功')

    # ============================================================ 日志 / 主题 / 后端门控
    def log_message(self, msg: str) -> None:
        """统一日志入口（三级通道之一：底部输出面板日志 tab）。"""
        self._log_signal.emit(msg)

    def _on_log_message(self, msg: str) -> None:
        if self.output_panel is not None:
            self.output_panel.append_log(msg)

    def _on_theme_changed(self, theme: str) -> None:
        """主题切换槽：setTheme + 输出面板换肤 + 各 View.apply_theme + 全量重绘。

        全窗控件单轮遍历同时完成 update() 与 apply_theme（原两轮 findChildren
        遍历在视图树大时构成重绘风暴）；页面/视图树运行期基本不变，
        无缓存列表的需求，单轮已够。
        """
        dark = str(theme) == constants.THEME_DARK
        apply_theme(theme)
        # qfluentwidgets 1.11 的 CardWidget 等纯 paintEvent 控件在主题切换时
        # 不会自动触发重绘（浅色底 + 深色文字的"半套主题"问题），这里强制
        # 全量 update()，保证深浅主题即时、完整地生效。
        # 日志框换肤：浅深主题都跟随——全宽面板下浅主题配深底大色块过于突兀
        if self.output_panel is not None:
            self.output_panel.apply_theme(dark)
        self._apply_top_nav_style()
        # 单轮遍历：重绘 + 主题应用一次完成（所有 BScanView/AScanView 等
        # 鸭子类型实现 apply_theme 的控件；output_panel 已在上文单独换肤）
        for widget in self.findChildren(QWidget):
            widget.update()
            apply_fn = getattr(widget, 'apply_theme', None)
            if callable(apply_fn) and widget is not self.output_panel:
                try:
                    apply_fn(dark)
                except Exception as e:
                    logger.debug('apply_theme 调用失败: %s', e)
        self.update()
        # 设置页主题 ComboBox 回写（blockSignals 防循环）
        settings_page = self._page('settingsInterface')
        if hasattr(settings_page, 'set_theme_text'):
            settings_page.set_theme_text(str(theme))
        self.settings.set('theme', str(theme))
        if not self._restoring_settings:
            self.settings.save()

    def _set_backend_ready(self, ready: bool) -> None:
        """后端未就绪时禁用除主页/设置外的页面，就绪后恢复。"""
        self._backend_ready = bool(ready)
        # list() 快照：本方法可能在预热期间被后端就绪回调触发，此时
        # _warmup_next_page 仍会往 pages 里加页 —— 直接遍历原字典会在
        # 嵌套事件循环中抛「dictionary changed size during iteration」。
        for object_name, page in list(self.pages.items()):
            if object_name in ('homeInterface', 'settingsInterface'):
                continue
            page.setEnabled(ready)
        if ready:
            self.log_message('SUCCESS 后端初始化完成')
            logger.info('backend ready')

    def _on_backend_failed(self, error: str) -> None:
        self.log_message(f'ERROR 后端初始化失败: {error}')
        logger.error('backend init failed: %s', error)
        # P0-2：失败必须可见——常驻错误横幅（含重试），而非静默禁用页面
        if self._backend_error_bar is not None:
            self._backend_error_bar.close()
        retry_btn = PushButton('重试')
        retry_btn.clicked.connect(self._retry_backend)
        bar = InfoBar.error(
            title='后端未就绪',
            content=str(error or '后端初始化失败，部分功能不可用'),
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=-1,  # 常驻，直到重试成功或用户手动关闭
            parent=self)
        bar.addWidget(retry_btn)
        self._backend_error_bar = bar

    def _retry_backend(self) -> None:
        """后端失败后重试初始化；成功后由 _on_backend_ready 恢复页面。"""
        if self._backend_error_bar is not None:
            self._backend_error_bar.close()
            self._backend_error_bar = None
        if self.backend_controller is not None:
            self.backend_controller.start(
                max_workers=int(self.settings.get('max_workers', constants.MAX_WORKERS)))

    def _on_backend_ready(self) -> None:
        """backend_ready：接通 JobBridge → 任务控件；加载方法库与预设。"""
        self._set_backend_ready(True)
        # 控制器注入 backend_controller（构造期已注入，此处幂等兜底）
        for ctrl in (self.project_controller, self.processing_controller,
                     self.interpretation_controller, self.delivery_controller):
            if ctrl is not None and hasattr(ctrl, 'set_backend'):
                ctrl.set_backend(self.backend_controller)
        bridge = getattr(self.backend_controller, 'job_bridge', None)
        if bridge is not None:
            # 任务信号槽在 PageCoordinator（候选 1 重构迁移到接线器）；
            # 由接线器显式接线：槽位不存在会立刻 AttributeError 暴露，
            # 而非 hasattr 探测静默跳过（历史事故：load_methods 不执行 → 方法库为空）。
            if self.page_coordinator is not None:
                self.page_coordinator.connect_job_bridge(bridge)
        if self.processing_controller is not None and self._pages_warmed:
            # 仅在预热收尾（connect_all 已接通 methods_loaded → 方法库）后加载。
            # 后端就绪若先于收尾，信号会发进空气（历史事故同款：方法库空白），
            # 由 _finish_warmup 接线完成后补加载，见下。
            self.processing_controller.load_methods()

    # ============================================================ 项目对话框（窗口 UI）
    def _show_new_project_dialog(self) -> None:
        """新建项目对话框：选目录 + 项目名 + 可选元数据 → create_project。"""
        if not self._backend_ready:
            self._infobar('warning', '新建项目', '后端尚未就绪，请稍后再试')
            return
        if self.project_controller is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('新建项目')
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        form.setSpacing(10)

        root_row = QHBoxLayout()
        root_edit = LineEdit(dialog)
        root_edit.setText(str(self.settings.get(
            'project_root', constants.DEFAULT_PROJECT_ROOT)))
        browse_btn = PushButton('浏览', dialog)
        browse_btn.setFixedWidth(70)

        def _browse() -> None:
            path = file_dialogs.getExistingDirectory(
                dialog, '选择项目根目录', root_edit.text().strip()
                or constants.DEFAULT_PROJECT_ROOT)
            if path:
                root_edit.setText(path)

        browse_btn.clicked.connect(_browse)
        root_row.addWidget(root_edit, 1)
        root_row.addWidget(browse_btn)
        form.addRow('项目根目录:', root_row)

        name_edit = LineEdit(dialog)
        name_edit.setPlaceholderText('例如: 新区道路探测')
        form.addRow('项目名称:', name_edit)

        # P1-9：6 个可选元数据折叠进"项目详情"，首屏只留根目录+名称
        detail_btn = QToolButton(dialog)
        detail_btn.setText('项目详情（可选）')
        detail_btn.setCheckable(True)
        detail_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        detail_btn.setArrowType(Qt.ArrowType.RightArrow)
        form.addRow(detail_btn)

        detail_widget = QWidget(dialog)
        detail_form = QFormLayout(detail_widget)
        detail_form.setContentsMargins(0, 0, 0, 0)
        detail_form.setSpacing(8)
        meta_edits = {}
        for key, label in (('location', '位置(可选):'), ('operator', '操作员(可选):'),
                           ('project_no', '项目编号(可选):'),
                           ('device_model', '设备型号(可选):'),
                           ('coordinate_system', '坐标系(可选):'),
                           ('vertical_datum', '高程基准(可选):')):
            edit = LineEdit(dialog)
            detail_form.addRow(label, edit)
            meta_edits[key] = edit
        detail_widget.setVisible(False)
        form.addRow(detail_widget)

        def _toggle_detail(checked: bool) -> None:
            detail_widget.setVisible(checked)
            detail_btn.setArrowType(
                Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

        detail_btn.toggled.connect(_toggle_detail)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        ok_btn = PrimaryPushButton('创建', dialog)
        cancel_btn = PushButton('取消', dialog)
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        def _accept() -> None:
            name = name_edit.text().strip()
            root = root_edit.text().strip()
            if not root:
                self._infobar('warning', '新建项目', '项目根目录不能为空')
                return
            if not name:
                self._infobar('warning', '新建项目', '项目名称不能为空')
                return
            meta = {key: edit.text().strip()
                    for key, edit in meta_edits.items() if edit.text().strip()}
            dialog.accept()
            self.project_controller.create_project(root, name, meta)

        ok_btn.clicked.connect(_accept)
        dialog.exec()

    def _open_project_dialog(self) -> None:
        """打开项目：目录对话框 → open_project。"""
        if not self._backend_ready:
            self._infobar('warning', '打开项目', '后端尚未就绪，请稍后再试')
            return
        if self.project_controller is None:
            return
        root = file_dialogs.getExistingDirectory(
            self, '选择项目目录', str(self.settings.get(
                'project_root', constants.DEFAULT_PROJECT_ROOT)))
        if root:
            self.project_controller.open_project(root)

    # ============================================================ 通用辅助
    def _infobar(self, level: str, title: str, content: str,
                 duration: int = None) -> None:
        """统一 InfoBar 用户反馈（success/info/warning/error）。"""
        fn = {'success': InfoBar.success, 'info': InfoBar.info,
              'warning': InfoBar.warning, 'error': InfoBar.error}.get(
            level, InfoBar.info)
        if duration is None:
            duration = {'success': 2000, 'info': 2000,
                        'warning': 3000, 'error': 15000}[level]  # 错误常驻更久，避免来不及读
        fn(title=title, content=str(content),
           orient=Qt.Orientation.Horizontal, isClosable=True,
           position=InfoBarPosition.TOP, duration=duration, parent=self)

    def _goto_page(self, object_name: str) -> None:
        """按 objectName 切导航页（HomePage.goto_page / 快速操作）。"""
        page = self.pages.get(str(object_name))
        if page is None and self._deferred_specs:
            # 预热未完成就切到延后页：立即同步构造全部剩余页面，避免空白页
            self.ensure_pages_ready()
            page = self.pages.get(str(object_name))
        if page is not None:
            self.switchTo(page)

    def _on_page_switched(self, index: int) -> None:
        """页面切换 → 顶部页签选中态同步 + 文件树按该页记忆的展开态收/放。"""
        widget = self.stackedWidget.widget(index)
        name = str(widget.objectName() or '') if widget is not None else ''
        top_nav = getattr(self, '_top_nav', None)
        if name and top_nav is not None:
            top_nav.setCurrentItem(name)
        if name and self._file_tree_panel is not None:
            self._file_tree_panel.apply_page(name)

    def _show_shortcuts_dialog(self) -> None:
        """F1：弹出当前支持的快捷键清单。"""
        dialog = QDialog(self)
        dialog.setWindowTitle('快捷键')
        dialog.setMinimumSize(360, 300)
        layout = QVBoxLayout(dialog)
        text = QTextBrowser(dialog)
        text.setOpenExternalLinks(False)
        rows = ['<h3>快捷键清单</h3><ul>']
        for seq, _obj, title in self._page_shortcuts:
            rows.append(f'<li><b>{seq}</b>：切换到{title}</li>')
        rows.extend([
            '<li><b>Ctrl+B</b>：收起/展开左侧文件树</li>',
            '<li><b>Ctrl+J</b>：收起/展开底部输出面板（日志/任务）</li>',
            '<li><b>Ctrl+R</b>：处理页运行处理链</li>',
            '<li><b>Ctrl+L</b>：处理页加载测线数据</li>',
            '<li><b>Delete</b>：处理链删除选中步骤 / 项目页删除选中测线</li>',
            '</ul><p>提示：处理链步骤行支持右键菜单（上移/下移/删除）。</p>',
        ])
        text.setHtml(''.join(rows))
        layout.addWidget(text)
        btn = QPushButton('关闭', dialog)
        btn.clicked.connect(dialog.close)
        layout.addWidget(btn)
        dialog.exec()

    def _current_project_id(self):
        if self.project_controller is None:
            return None
        return self.project_controller.current_project_id

    def _require_project(self) -> bool:
        """无项目门控（SPEC §7）：未打开项目时提示并返回 False。"""
        if not self._backend_ready:
            self._infobar('warning', '提示', '后端尚未就绪，请稍后再试')
            return False
        if not self._current_project_id():
            self._infobar('warning', '提示', '请先在主页打开或新建项目')
            return False
        return True

    def _require_line(self) -> str:
        """返回当前测线号；无项目/无测线时提示并返回 ''。

        当前测线的写入点在 PageCoordinator（候选 1 状态迁移），
        此处只读——须委托 coordinator，读 self 会 AttributeError
        且被 Qt 信号槽吞掉（表现为"运行处理链没反应"）。
        """
        if not self._require_project():
            return ''
        pc = self.page_coordinator
        if pc is None:
            return ''
        line_id = pc.current_line_id()   # coordinator 拥有该状态（候选 1 迁移）
        if not line_id:
            self._infobar('warning', '提示', '请先在项目页导入并选择测线')
            return ''
        return line_id

    def _job_bridge(self):
        if self.backend_controller is None:
            return None
        return getattr(self.backend_controller, 'job_bridge', None)

    # ============================================================ 关闭
    def closeEvent(self, event) -> None:
        # 1) 先关解释标注会话（可能涉及后端会话资源）
        if self.interpretation_controller is not None:
            try:
                self.interpretation_controller.close_session()
            except Exception as e:  # noqa: BLE001
                logger.warning('关闭标注会话异常（已吞掉）: %s', e)
        # 2) 持久化设置（含设置页当前控件值）
        settings_page = self._page('settingsInterface')
        if hasattr(settings_page, 'settings'):
            try:
                for key, value in settings_page.settings().items():
                    self.settings.set(key, value)
            except Exception as e:  # noqa: BLE001
                logger.warning('设置页状态回写失败（已吞掉）: %s', e)
        self.settings.save()
        # 3) 后端 shutdown
        if self.backend_controller is not None:
            try:
                self.backend_controller.shutdown()
            except Exception as e:
                logger.warning('backend shutdown 异常（已吞掉）: %s', e)
        super().closeEvent(event)
