# -*- coding: utf-8 -*-
"""MyGPRMainWindow(FluentWindow) 组装器（SPEC §6.1，style_spec §2 复刻）。

主窗口只做「创建 → 布局 → connect」：
    _init_window → _create_controllers → _create_pages
    → _build_ui → _connect_signals → _init_state

页面 / 控制器 / LogPanel 均以 try/except 导入：缺失时页面降级为
PlaceholderPage（QLabel '页面建设中' 居中），controller 为 None（connect 判空）。
"""
import datetime
import logging

from PyQt6.QtCore import (
    QEasingCurve, QPropertyAnimation, Qt, QTimer, pyqtSignal,
)
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QTextBrowser, QTextEdit, QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    CardWidget, FluentIcon as FIF, FluentWindow, InfoBar, InfoBarPosition,
    NavigationItemPosition, PushButton,
    SegmentedWidget, SplashScreen,
)

from ui.window_mixins import (
    _DeliveryMixin,
    _ImportPreflightMixin,
    _InterpretationMixin,
    _JobCenterMixin,
    _LineArtifactMixin,
    _ProcessingMixin,
    _ProjectLifecycleMixin,
)

from ui import constants
from ui.logger_config import setup_logger
from ui.settings_manager import SettingsManager
from ui.theme_helpers import apply_theme, log_panel_qss

logger = setup_logger('mygpr_window', 'logs/mygpr_window.log', level=logging.DEBUG)

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

# ------------------------------------------------------------ LogPanel（[A2] 提供，缺失用内置回退）
LogPanel = _import_page_class('ui.widgets.log_panel', 'LogPanel')


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


class _FallbackLogPanel(CardWidget):
    """LogPanel 内置回退（A2 交付前使用）。

    CardWidget 容器 max 380/min 0，margins 6/spacing 6；
    顶部 SegmentedWidget('日志','任务') + QStackedWidget；
    日志 tab：QTextEdit 只读 + 按钮行('清空' 60px)。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(0)
        self.setMaximumWidth(constants.PANEL_MAX_WIDTH)
        self.setMinimumHeight(constants.PANEL_MIN_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*constants.PANEL_MARGINS)
        layout.setSpacing(constants.PANEL_SPACING)

        self.segmented = SegmentedWidget(self)
        self.stacked = QStackedWidget(self)
        self.stacked.setMinimumWidth(364)

        # 日志 tab
        log_tab = QWidget(self)
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(constants.PANEL_SPACING)
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(log_panel_qss('terminal'))
        btn_row = QHBoxLayout()
        self.clear_btn = PushButton('清空', self)
        self.clear_btn.setFixedWidth(60)
        self.clear_btn.clicked.connect(self.log_text.clear)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch(1)
        log_layout.addWidget(self.log_text, 1)
        log_layout.addLayout(btn_row)

        # 任务 tab（A2 MiniJobList 交付前为占位）
        jobs_tab = QWidget(self)
        jobs_layout = QVBoxLayout(jobs_tab)
        jobs_layout.setContentsMargins(0, 0, 0, 0)
        jobs_label = QLabel('暂无任务', jobs_tab)
        jobs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        jobs_layout.addWidget(jobs_label)

        log_tab.setObjectName('logTab')
        jobs_tab.setObjectName('jobsTab')
        self.stacked.addWidget(log_tab)
        self.stacked.addWidget(jobs_tab)
        self.segmented.addItem('logTab', '日志',
                               onClick=lambda: self.stacked.setCurrentWidget(log_tab))
        self.segmented.addItem('jobsTab', '任务',
                               onClick=lambda: self.stacked.setCurrentWidget(jobs_tab))
        self.segmented.setCurrentItem('logTab')

        layout.addWidget(self.segmented)
        layout.addWidget(self.stacked, 1)

    # ------------------------------------------------------------ 对外接口（与 A2 LogPanel 对齐）
    def append_log(self, msg: str) -> None:
        """自动加 [HH:MM:SS] 前缀 + 级别着色 + 滚到底（style_spec §2.5）。"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        color = None
        if any(k in msg for k in ('ERROR', '失败', '错误')):
            color = constants.LOG_COLOR_ERROR     # #dc3545
        elif any(k in msg for k in ('WARNING', '警告')):
            color = constants.LOG_COLOR_WARNING   # #ffc107
        elif any(k in msg for k in ('SUCCESS', '成功', '完成')):
            color = constants.LOG_COLOR_SUCCESS   # #28a745
        elif 'INFO' in msg:
            color = constants.LOG_COLOR_INFO      # #17a2b8
        text = f'[{timestamp}] {msg}'
        if color:
            text = f'<span style="color:{color}">{text}</span>'
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def mini_jobs(self):
        return None

    def apply_theme(self, dark: bool) -> None:
        """日志框随主题换肤（style_spec §2.5：深色 #1e1e1e / 浅色 #f5f5f5）。"""
        self.log_text.setStyleSheet(log_panel_qss('dark' if dark else 'light'))


class MyGPRMainWindow(FluentWindow, _ProjectLifecycleMixin, _LineArtifactMixin,
                      _ImportPreflightMixin, _ProcessingMixin,
                      _InterpretationMixin, _DeliveryMixin, _JobCenterMixin):
    """主窗口组装器（SPEC §6.1）。"""

    _log_signal = pyqtSignal(str)   # 类级信号：跨线程日志安全转发

    def __init__(self, settings: SettingsManager = None, parent=None):
        super().__init__(parent)
        self.settings = settings or SettingsManager()
        self.spacing = constants.PAGE_SPACING
        self._restoring_settings = False
        self._panel_animating = False
        self.pages = {}             # objectName -> page widget

        # ---- 业务接线状态（SPEC §7）----
        self._backend_ready = False
        self._current_line_id = ''          # 当前测线（项目/处理/解释页共用）
        self._known_job_ids = set()         # 已 upsert 到任务控件的任务
        self._import_job_ids = set()        # 测线导入/传感器同步任务（完成后刷新测线）
        self._spatial_job_ids = set()       # 空间成果任务（完成后刷新空间成果表）
        self._processing_job_id = ''        # 处理页当前运行任务
        self._processing_line_id = ''       # 运行提交时的测线（防运行中切测线竞态）
        self._processing_cancel_requested = False  # 用户主动取消（区别于失败）
        self._preview_newest_artifact = False  # 处理完成后自动预览最新成果
        self._show_run_completion_notice = False  # 处理完成后提示一次
        self._pending_select_line_id = ''        # 导入完成后要选中的测线
        self._backend_error_bar = None           # 后端初始化失败的常驻错误横幅

        self._init_window()
        self._create_controllers()
        self._create_pages()
        self._build_ui()
        self._setup_global_shortcuts()
        self._connect_signals()
        self._init_state()
        # 注入设置值到页面（设置页更改后下次启动生效）
        self._inject_page_settings()

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
        self.setWindowIcon(QIcon(constants.APP_ICON_PATH))
        self.navigationInterface.setExpandWidth(constants.NAV_EXPAND_WIDTH)

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
        for object_name, page_class, icon, text, position in page_specs:
            page = page_class(self) if page_class else PlaceholderPage(text, self)
            page.setObjectName(object_name)
            self.addSubInterface(page, icon, text, position=position)
            self.pages[object_name] = page

    def _build_ui(self) -> None:
        """右侧全局可折叠面板（折叠按钮 + LogPanel 承载）。"""
        # 折叠按钮：chevron 图标 + 主题色淡底长条（与 CollapsiblePanel 统一视觉）
        from ui.widgets import collapse_button_qss
        self.fold_button = PushButton('', self)
        self.fold_button.setFixedSize(constants.FOLD_BUTTON_WIDTH,
                                      constants.FOLD_BUTTON_HEIGHT)
        self.fold_button.setIcon(FIF.CHEVRON_RIGHT_MED.icon())
        self.fold_button.setToolTip('收起右侧面板')
        self.fold_button.setStyleSheet(collapse_button_qss())
        self.fold_button.clicked.connect(self._toggle_panel)

        # 面板容器：LogPanel（A2）或内置回退
        self.log_panel = LogPanel(self) if LogPanel else _FallbackLogPanel(self)
        self.log_panel.setMinimumWidth(0)
        self.log_panel.setMaximumWidth(constants.PANEL_MAX_WIDTH)
        self.log_panel.setMinimumHeight(constants.PANEL_MIN_HEIGHT)

        # 追加到 FluentWindow 根布局（导航 | 页面 | 折叠钮 | 面板）
        self.hBoxLayout.addWidget(self.fold_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self.hBoxLayout.addWidget(self.log_panel)

        # 折叠动画：maximumWidth 220ms OutCubic 0↔380
        self.panel_animation = QPropertyAnimation(self.log_panel, b'maximumWidth', self)
        self.panel_animation.setDuration(constants.PANEL_ANIM_DURATION_MS)
        self.panel_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.panel_animation.finished.connect(self._on_panel_animation_finished)
        self._panel_collapsed = False

    def _page(self, object_name: str):
        """按 objectName 取页面（占位页返回原样，调用方自行判接口）。"""
        return self.pages.get(object_name)

    def _setup_global_shortcuts(self) -> None:
        """全局快捷键：Ctrl+1~8 切换页签，F1 显示快捷键清单。"""
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
        self._shortcuts_help = QShortcut(QKeySequence("F1"), self)
        self._shortcuts_help.setContext(Qt.ShortcutContext.WindowShortcut)
        self._shortcuts_help.activated.connect(self._show_shortcuts_dialog)

    def _connect_signals(self) -> None:
        """全量业务接线（SPEC §7）：Page 信号 → Controller 槽；Controller 信号 → Page set_xxx。"""
        self._log_signal.connect(self._on_log_message)

        home = self._page('homeInterface')
        project = self._page('projectInterface')
        processing = self._page('processingInterface')
        interpretation = self._page('interpretationInterface')
        spatial = self._page('spatialInterface')
        delivery = self._page('deliveryInterface')
        jobs = self._page('jobsInterface')
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

        # ---------------- 主页（SPEC §6.2）
        if hasattr(home, 'new_project_requested'):
            home.new_project_requested.connect(self._show_new_project_dialog)
            home.open_project_requested.connect(self._open_project_dialog)
            home.import_line_requested.connect(
                lambda: self._goto_page('projectInterface'))
            home.goto_page.connect(self._goto_page)
            home_jobs = home.mini_jobs()
            if home_jobs is not None and hasattr(home_jobs, 'cancel_requested'):
                home_jobs.cancel_requested.connect(self._cancel_job)

        # ---------------- 项目页（SPEC §6.3）
        if hasattr(project, 'import_requested'):
            project.import_requested.connect(self._on_import_requested)
            project.sync_requested.connect(self._on_sync_requested)
            project.line_selected.connect(self._on_line_selected)
            project.line_process_requested.connect(self._on_line_process_requested)
            project.line_delete_requested.connect(self._on_line_delete_requested)
            project.artifact_preview_requested.connect(self._on_artifact_preview_requested)
            project.close_project_requested.connect(self._on_close_project_requested)
            # 测线表右键"复制路径/打开所在文件夹"的路径查询回调
            project.set_source_path_resolver(
                self.project_controller.line_source_path)

        # ---------------- 处理页（SPEC §6.5）
        if hasattr(processing, 'run_requested'):
            processing.run_requested.connect(self._on_run_requested)
            processing.cancel_requested.connect(self._on_processing_cancel)
            processing.autotune_requested.connect(self._on_autotune_requested)
            processing.line_load_requested.connect(self._on_line_load_requested)
            processing.line_changed.connect(self._on_processing_line_changed)
            processing.artifact_selected.connect(self._on_processing_artifact_selected)

        # ---------------- 解释页（SPEC §6.6）
        if hasattr(interpretation, 'open_session_requested'):
            interpretation.open_session_requested.connect(self._on_open_session_requested)
            interpretation.points_changed.connect(self._on_points_changed)
            if self.interpretation_controller is not None:
                interpretation.auto_trace_requested.connect(
                    self.interpretation_controller.auto_trace)
                interpretation.snap_requested.connect(self.interpretation_controller.snap)
                interpretation.smooth_requested.connect(
                    self.interpretation_controller.smooth)
                interpretation.undo_requested.connect(self.interpretation_controller.undo)
                interpretation.redo_requested.connect(self.interpretation_controller.redo)
                interpretation.save_requested.connect(self.interpretation_controller.save)

        # ---------------- 空间信息页
        if hasattr(spatial, 'current_line_requested'):
            spatial.current_line_requested.connect(self._on_spatial_current_line)

        # ---------------- 成果页（SPEC §6.7）
        if hasattr(delivery, 'spatial_requested'):
            delivery.spatial_requested.connect(self._on_spatial_requested)
            delivery.report_requested.connect(self._on_report_requested)
            delivery.backup_requested.connect(self._on_backup_requested)
            delivery.restore_requested.connect(self._on_restore_requested)

        # ---------------- 任务页（SPEC §6.8）
        if hasattr(jobs, 'cancel_requested'):
            jobs.cancel_requested.connect(self._cancel_job)
            jobs.prune_requested.connect(self._on_prune_jobs)

        # ---------------- 设置页（SPEC §6.4）
        if hasattr(settings_page, 'theme_changed'):
            settings_page.theme_changed.connect(self._on_theme_changed)

        # ---------------- 右侧日志面板任务取消
        if hasattr(self.log_panel, 'cancel_job_requested'):
            self.log_panel.cancel_job_requested.connect(self._cancel_job)

        # ---------------- 控制器 → 页面
        pc = self.project_controller
        if pc is not None:
            pc.project_opened.connect(self._on_project_opened)
            pc.project_closed.connect(self._on_project_closed)
            pc.open_failed.connect(self._on_open_failed)
            pc.lines_updated.connect(self._on_lines_updated)
            pc.artifacts_updated.connect(self._on_artifacts_updated)
            pc.dataset_preview_ready.connect(self._on_dataset_preview)
            pc.artifact_preview_ready.connect(self._on_artifact_preview)
            pc.preflight_ready.connect(self._on_preflight_ready)
            pc.preflight_failed.connect(self._on_preflight_failed)
            if hasattr(spatial, 'set_tracks') and hasattr(pc, 'spatial_tracks_ready'):
                pc.spatial_tracks_ready.connect(spatial.set_tracks)
            if hasattr(project, 'set_busy'):
                pc.busy_changed.connect(project.set_busy)

        prc = self.processing_controller
        if prc is not None:
            if hasattr(processing, 'set_methods'):
                prc.methods_loaded.connect(self._on_methods_loaded)
            prc.run_finished.connect(self._on_run_finished)
            prc.autotune_finished.connect(self._on_autotune_finished)
            prc.autotune_failed.connect(self._on_autotune_failed)

        ic = self.interpretation_controller
        if ic is not None and hasattr(interpretation, 'set_points'):
            ic.session_opened.connect(self._on_session_opened)
            ic.session_updated.connect(self._on_session_updated)
            ic.session_failed.connect(self._on_session_failed)
            ic.saved.connect(self._on_annotation_saved)
            ic.busy_changed.connect(interpretation.set_busy)

        dc = self.delivery_controller
        if dc is not None and hasattr(delivery, 'set_spatial_results'):
            dc.spatial_results_updated.connect(delivery.set_spatial_results)
            dc.report_generated.connect(self._on_report_generated)

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

        # 恢复全局日志面板折叠状态
        if bool(self.settings.get('log_panel_collapsed', True)):
            self._panel_collapsed = True
            self.log_panel.hide()
            self._set_fold_button_look(collapsed=True)

        # 后端就绪门控：除主页/设置外页面禁用，backend_ready 后 enable
        if self.backend_controller is not None:
            self._set_backend_ready(False)
            self.backend_controller.start(
                max_workers=int(self.settings.get('max_workers', constants.MAX_WORKERS)))
            self.log_message('INFO 后端初始化中…')
        else:
            self.log_message('WARNING BackendController 未就绪（骨架阶段），全部页面保持启用')

        self.log_message('INFO MyGPR 探地雷达数据处理软件启动成功')

    # ============================================================ 右侧面板
    def _toggle_panel(self) -> None:
        """折叠/展开右侧面板（220ms OutCubic，动画防重入）。"""
        if self._panel_animating:
            return
        self._panel_animating = True
        self.panel_animation.stop()
        if self._panel_collapsed:
            # 展开：0 → 380
            self.log_panel.setMaximumWidth(0)
            self.log_panel.show()
            self.panel_animation.setStartValue(0)
            self.panel_animation.setEndValue(constants.PANEL_MAX_WIDTH)
        else:
            # 收起：380 → 0，完成后 hide() 并复位 maximumWidth
            self.panel_animation.setStartValue(constants.PANEL_MAX_WIDTH)
            self.panel_animation.setEndValue(0)
        self.panel_animation.start()

    def _on_panel_animation_finished(self) -> None:
        self._panel_animating = False
        if self._panel_collapsed:
            # 本次是展开完成
            self._panel_collapsed = False
            self._set_fold_button_look(collapsed=False)
        else:
            # 本次是收起完成：hide 并复位 maximumWidth
            self._panel_collapsed = True
            self.log_panel.hide()
            self.log_panel.setMaximumWidth(constants.PANEL_MAX_WIDTH)
            self._set_fold_button_look(collapsed=True)
        self.settings.set('log_panel_collapsed', self._panel_collapsed)
        self.settings.save()

    def _set_fold_button_look(self, collapsed: bool) -> None:
        """全局日志面板折叠按钮的图标与提示随状态切换。"""
        from ui.widgets import chevron_left_icon
        if collapsed:
            self.fold_button.setIcon(chevron_left_icon())
            self.fold_button.setToolTip('展开右侧面板')
        else:
            self.fold_button.setIcon(FIF.CHEVRON_RIGHT_MED.icon())
            self.fold_button.setToolTip('收起右侧面板')

    # ============================================================ 日志 / 主题 / 后端门控
    def log_message(self, msg: str) -> None:
        """统一日志入口（三级通道之一：右侧日志面板）。"""
        self._log_signal.emit(msg)

    def _on_log_message(self, msg: str) -> None:
        if hasattr(self.log_panel, 'append_log'):
            self.log_panel.append_log(msg)

    def _on_theme_changed(self, theme: str) -> None:
        """主题切换槽：setTheme + LogPanel 换肤 + 各 View.apply_theme + pg 背景。"""
        dark = str(theme) == constants.THEME_DARK
        apply_theme(theme)
        # qfluentwidgets 1.11 的 CardWidget 等纯 paintEvent 控件在主题切换时
        # 不会自动触发重绘（浅色底 + 深色文字的"半套主题"问题），这里强制
        # 全量 update()，保证深浅主题即时、完整地生效。
        for widget in self.findChildren(QWidget):
            widget.update()
        self.update()
        # 日志框换肤（style_spec §2.5）：启动回放浅色主题时保留初始 #2b2b2b 深底
        if not self._restoring_settings or dark:
            if hasattr(self.log_panel, 'apply_theme'):
                self.log_panel.apply_theme(dark)
            elif hasattr(self.log_panel, 'set_theme'):
                self.log_panel.set_theme(theme)
        # 所有 BScanView/AScanView（鸭子类型，A2 交付后自动生效）
        for widget in self.findChildren(QWidget):
            apply_fn = getattr(widget, 'apply_theme', None)
            if callable(apply_fn) and widget is not self.log_panel:
                try:
                    apply_fn(dark)
                except Exception as e:
                    logger.debug('apply_theme 调用失败: %s', e)
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
        for object_name, page in self.pages.items():
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
            bridge.progress_changed.connect(self._on_job_progress)
            bridge.status_changed.connect(self._on_job_status)
            bridge.job_completed.connect(self._on_job_completed)
        if self.processing_controller is not None:
            self.processing_controller.load_methods()

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
        if page is not None:
            self.switchTo(page)

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
        """返回当前测线号；无项目/无测线时提示并返回 ''。"""
        if not self._require_project():
            return ''
        if not self._current_line_id:
            self._infobar('warning', '提示', '请先在项目页导入并选择测线')
            return ''
        return self._current_line_id

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
