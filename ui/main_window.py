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
import os
import threading

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QStackedWidget,
    QTextEdit, QVBoxLayout, QWidget,
)
from qfluentwidgets import (
    CardWidget, FluentIcon as FIF, FluentWindow, InfoBar, InfoBarPosition,
    LineEdit, NavigationItemPosition, PrimaryPushButton, PushButton,
    SegmentedWidget, SplashScreen,
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


class MyGPRMainWindow(FluentWindow):
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
        self._preview_newest_artifact = False  # 处理完成后自动预览最新成果

        self._init_window()
        self._create_controllers()
        self._create_pages()
        self._build_ui()
        self._connect_signals()
        self._init_state()

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
        # 折叠按钮：宽18 高60，去圆角去边框（style_spec §2.5 逐字 QSS）
        self.fold_button = PushButton('◀', self)
        self.fold_button.setFixedSize(constants.FOLD_BUTTON_WIDTH,
                                      constants.FOLD_BUTTON_HEIGHT)
        self.fold_button.setToolTip('收起/展开右侧面板')
        self.fold_button.setStyleSheet(
            'PushButton { border: none; border-radius: 0; font-size: 10px; padding: 0; }')
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
            self.fold_button.setText('▶')

        # 后端就绪门控：除主页/设置外页面禁用，backend_ready 后 enable
        if self.backend_controller is not None:
            self._set_backend_ready(False)
            self.backend_controller.start()
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
            self.fold_button.setText('◀')
        else:
            # 本次是收起完成：hide 并复位 maximumWidth
            self._panel_collapsed = True
            self.log_panel.hide()
            self.log_panel.setMaximumWidth(constants.PANEL_MAX_WIDTH)
            self.fold_button.setText('▶')
        self.settings.set('log_panel_collapsed', self._panel_collapsed)
        self.settings.save()

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
                        'warning': 3000, 'error': 5000}[level]
        fn(title=title, content=str(content),
           orient=Qt.Orientation.Horizontal, isClosable=True,
           position=InfoBarPosition.TOP, duration=duration, parent=self)

    def _goto_page(self, object_name: str) -> None:
        """按 objectName 切导航页（HomePage.goto_page / 快速操作）。"""
        page = self.pages.get(str(object_name))
        if page is not None:
            self.switchTo(page)

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

    # ============================================================ 主页：新建/打开项目
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
            path = QFileDialog.getExistingDirectory(
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

        meta_edits = {}
        for key, label in (('location', '位置(可选):'), ('operator', '操作员(可选):'),
                           ('project_no', '项目编号(可选):'),
                           ('device_model', '设备型号(可选):'),
                           ('coordinate_system', '坐标系(可选):'),
                           ('vertical_datum', '高程基准(可选):')):
            edit = LineEdit(dialog)
            form.addRow(label, edit)
            meta_edits[key] = edit
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
        root = QFileDialog.getExistingDirectory(
            self, '选择项目目录', str(self.settings.get(
                'project_root', constants.DEFAULT_PROJECT_ROOT)))
        if root:
            self.project_controller.open_project(root)

    def _on_close_project_requested(self) -> None:
        if self.project_controller is not None and self._require_project():
            self.project_controller.close_current()

    # ============================================================ 项目生命周期 → 各页
    def _on_project_opened(self, summary) -> None:
        """project_opened → 主页卡片 + 项目页信息 + 最近项目 + 成果页刷新。"""
        home = self._page('homeInterface')
        project = self._page('projectInterface')
        if hasattr(home, 'set_current_project'):
            home.set_current_project(summary)
        if hasattr(project, 'set_project_info'):
            project.set_project_info(summary)
        root = str(getattr(summary, 'root_path', '') or '')
        if root:
            self.settings.add_recent_project(root)
            self.settings.save()
        self._current_line_id = ''
        self._update_line_labels()
        # 测线列表由 controller 在 project_opened 后自行 refresh_lines
        project_id = self._current_project_id()
        if project_id and self.delivery_controller is not None:
            self.delivery_controller.refresh_spatial(project_id)
        # 空间信息页：项目打开后加载空间轨迹
        if self.project_controller is not None and hasattr(
                self.project_controller, 'load_spatial_tracks'):
            self.project_controller.load_spatial_tracks()
        name = getattr(summary, 'name', '')
        self._infobar('success', '项目', f'项目已打开：{name}')
        self.log_message(f'SUCCESS 当前项目：{name}（{root}）')

    def _on_project_closed(self) -> None:
        """project_closed → 清空各页项目态并恢复无项目门控。"""
        self._current_line_id = ''
        home = self._page('homeInterface')
        project = self._page('projectInterface')
        processing = self._page('processingInterface')
        interpretation = self._page('interpretationInterface')
        delivery = self._page('deliveryInterface')
        if hasattr(home, 'set_current_project'):
            home.set_current_project(None)
            home.set_preview_bundle(None)
        if hasattr(project, 'set_project_info'):
            project.set_project_info(None)   # 页面内部禁用操作按钮 + 顶部提示
            project.set_lines([])
            project.set_artifacts([])
            project.set_preview_bundle(None)
        if hasattr(processing, 'set_line_label'):
            processing.set_line_label('')
            processing.set_original_bundle(None)
            processing.set_result_bundle(None)
        if hasattr(interpretation, 'set_line_label'):
            interpretation.set_line_label('')
            interpretation.set_session_info('未打开会话')
            interpretation.set_points([])
        if hasattr(delivery, 'set_lines'):
            delivery.set_lines([])
            delivery.set_spatial_results([])
        spatial = self._page('spatialInterface')
        if hasattr(spatial, 'set_tracks'):
            spatial.set_tracks([])
            spatial.set_lines([])
        self.log_message('INFO 项目已关闭，相关页面恢复未打开项目状态')

    def _on_open_failed(self, message: str) -> None:
        self._infobar('error', '项目', message)

    # ============================================================ 测线 / 成果 / 预览
    def _update_line_labels(self) -> None:
        """当前测线标签：处理页 / 解释页共用。"""
        processing = self._page('processingInterface')
        interpretation = self._page('interpretationInterface')
        if hasattr(processing, 'set_line_label'):
            processing.set_line_label(self._current_line_id)
        if hasattr(interpretation, 'set_line_label'):
            interpretation.set_line_label(self._current_line_id)

    def _on_lines_updated(self, lines: list) -> None:
        """lines_updated → 项目页测线表（自动选中首行）+ 成果页/空间页测线多选。"""
        lines = list(lines or [])
        project = self._page('projectInterface')
        delivery = self._page('deliveryInterface')
        spatial = self._page('spatialInterface')
        if hasattr(project, 'set_lines'):
            project.set_lines(lines)   # 非空时自动选中首行 → line_selected
        if hasattr(delivery, 'set_lines'):
            delivery.set_lines(lines)
        if hasattr(spatial, 'set_lines'):
            spatial.set_lines(lines)
        # 测线集合变化后空间轨迹同步重载（导入/同步完成均触发 lines_updated）
        if self.project_controller is not None and hasattr(
                self.project_controller, 'load_spatial_tracks'):
            self.project_controller.load_spatial_tracks()
        valid_ids = [str(getattr(line, 'line_id', '') or '') for line in lines]
        if self._current_line_id not in valid_ids:
            self._current_line_id = valid_ids[0] if valid_ids else ''
            self._update_line_labels()
        if not lines:
            if hasattr(project, 'set_artifacts'):
                project.set_artifacts([])

    def _on_line_selected(self, line_id: str) -> None:
        """line_selected → 记录当前测线 + 预览 + 刷新成果列表。"""
        line_id = str(line_id or '')
        if not line_id:
            return
        self._current_line_id = line_id
        self._update_line_labels()
        if self.project_controller is not None:
            self.project_controller.preview_line(line_id)
            self.project_controller.refresh_artifacts(line_id)

    def _on_line_process_requested(self, line_id: str) -> None:
        """项目页双击测线 → 设为当前测线并加载数据，跳转处理页。"""
        line_id = str(line_id or '')
        if not line_id:
            return
        self._on_line_selected(line_id)
        self._goto_page('processingInterface')
        self._infobar('info', '数据预览', f'正在加载测线 {line_id} …')

    def _on_spatial_current_line(self, line_id: str) -> None:
        """空间信息页"设为当前测线" → 复用测线选择逻辑 + InfoBar 提示。"""
        line_id = str(line_id or '')
        if not line_id or line_id == self._current_line_id:
            return
        self._on_line_selected(line_id)
        self._infobar('success', '空间信息', f'已设为当前测线：{line_id}')

    def _on_artifacts_updated(self, line_id: str, artifacts: list) -> None:
        """artifacts_updated → 项目页成果表；处理完成后自动预览最新成果。"""
        if str(line_id) != self._current_line_id:
            return
        artifacts = list(artifacts or [])
        project = self._page('projectInterface')
        if hasattr(project, 'set_artifacts'):
            project.set_artifacts(artifacts)
        if self._preview_newest_artifact:
            self._preview_newest_artifact = False
            if artifacts and self.project_controller is not None:
                artifact_id = str(getattr(artifacts[-1], 'artifact_id', '') or '')
                if artifact_id:
                    self.project_controller.preview_artifact(
                        self._current_line_id, artifact_id)

    def _on_artifact_preview_requested(self, line_id: str, artifact_id: str) -> None:
        if self.project_controller is not None and self._require_project():
            self.project_controller.preview_artifact(str(line_id), str(artifact_id))

    def _on_dataset_preview(self, bundle) -> None:
        """原始数据预览 → 主页 / 项目页 / 处理页（原始）/ 解释页剖面。"""
        home = self._page('homeInterface')
        project = self._page('projectInterface')
        processing = self._page('processingInterface')
        interpretation = self._page('interpretationInterface')
        if hasattr(home, 'set_preview_bundle'):
            home.set_preview_bundle(bundle)
        if hasattr(project, 'set_preview_bundle'):
            project.set_preview_bundle(bundle)
        if hasattr(processing, 'set_original_bundle'):
            processing.set_original_bundle(bundle)
        if hasattr(interpretation, 'set_bundle'):
            interpretation.set_bundle(bundle)

    def _on_artifact_preview(self, artifact_id: str, bundle) -> None:
        """成果预览 → 项目页预览 + 处理页（处理结果）。"""
        project = self._page('projectInterface')
        processing = self._page('processingInterface')
        if hasattr(project, 'set_preview_bundle'):
            project.set_preview_bundle(bundle)
        if hasattr(processing, 'set_result_bundle'):
            processing.set_result_bundle(bundle)

    # ============================================================ 导入 / 预检 / 传感器同步
    def _on_import_requested(self, payload: dict) -> None:
        """import_requested：preflight=True→预检；False→提交导入任务。"""
        if self.project_controller is None or not self._require_project():
            return
        payload = dict(payload or {})
        if payload.get('preflight'):
            self.project_controller.preflight_import(
                str(payload.get('source', '')),
                str(payload.get('line_id', '') or 'L01'),
                float(payload.get('dielectric', constants.DEFAULT_DIELECTRIC)))
            return
        job_id = self.project_controller.import_line(
            str(payload.get('source', '')),
            str(payload.get('line_id', '') or 'L01'),
            str(payload.get('name', '') or ''),
            float(payload.get('dielectric', constants.DEFAULT_DIELECTRIC)))
        if job_id:
            self._import_job_ids.add(str(job_id))
            self._infobar('info', '导入测线', '导入任务已提交，可在任务页查看进度')

    def _on_preflight_ready(self, result) -> None:
        """preflight_ready → 项目页预检结果区（鸭子类型取字段）。"""
        project = self._page('projectInterface')
        if not hasattr(project, 'set_preflight_result'):
            return
        can_import = bool(getattr(result, 'can_import', False))
        message = str(getattr(result, 'message', '') or '')
        fmt = str(getattr(result, 'format_name', '') or '')
        samples = int(getattr(result, 'sample_count', 0) or 0)
        traces = int(getattr(result, 'trace_count', 0) or 0)
        parts = [message] if message else []
        if fmt:
            parts.append(f'格式: {fmt}')
        if samples and traces:
            parts.append(f'数据: {samples} 采样 × {traces} 道')
        suggestions = [str(s) for s in (getattr(result, 'suggestions', ()) or ())]
        parts.extend(suggestions)
        project.set_preflight_result('\n'.join(parts) or '预检完成', can_import)

    def _on_preflight_failed(self, message: str) -> None:
        project = self._page('projectInterface')
        if hasattr(project, 'set_preflight_result'):
            project.set_preflight_result(f'预检失败: {message}', False)

    def _on_sync_requested(self, payload: dict) -> None:
        if self.project_controller is None or not self._require_project():
            return
        payload = dict(payload or {})
        job_id = self.project_controller.sync_sensors(
            str(payload.get('line_id', '') or self._current_line_id),
            dict(payload.get('paths') or {}),
            dict(payload.get('settings') or {}))
        if job_id:
            self._import_job_ids.add(str(job_id))   # 同步完成后同样刷新测线
            self._infobar('info', '传感器同步', '同步任务已提交')

    # ============================================================ 处理页
    def _on_methods_loaded(self, methods: list) -> None:
        """方法库 → 处理页 MethodBrowser。"""
        processing = self._page('processingInterface')
        if hasattr(processing, 'set_methods'):
            processing.set_methods(methods)

    def _on_line_load_requested(self) -> None:
        """处理页"加载测线数据" → 预览当前测线原始数据。"""
        line_id = self._require_line()
        if line_id and self.project_controller is not None:
            self.project_controller.preview_line(line_id)
            self._infobar('info', '数据预览', f'正在加载测线 {line_id} …')

    def _on_run_requested(self, payload: dict) -> None:
        """run_requested(dict) → run_pipeline（含结果名回退）。"""
        if self.processing_controller is None:
            return
        line_id = self._require_line()
        if not line_id:
            return
        payload = dict(payload or {})
        steps = list(payload.get('steps') or [])
        if not steps:
            self._infobar('warning', '处理链', '处理链为空，请先添加处理步骤')
            return
        result_name = str(payload.get('result_name') or ''
                          ).strip() or f'处理结果_{line_id}'
        job_id = self.processing_controller.run_pipeline(
            self._current_project_id(), line_id, {'steps': steps}, result_name)
        if job_id:
            self._processing_job_id = str(job_id)
            processing = self._page('processingInterface')
            if hasattr(processing, 'set_running'):
                processing.set_running(True, str(job_id))

    def _on_processing_cancel(self) -> None:
        bridge = self._job_bridge()
        if bridge is not None and self._processing_job_id:
            bridge.cancel(self._processing_job_id)
            self.log_message(f'INFO 已请求取消处理任务 {self._processing_job_id}')

    def _on_run_finished(self, success: bool, message: str) -> None:
        """run_finished → 恢复运行态 + InfoBar + 刷新成果并自动预览。"""
        self._processing_job_id = ''
        processing = self._page('processingInterface')
        if hasattr(processing, 'set_running'):
            processing.set_running(False)
        if success:
            self._infobar('success', '处理链', message or '处理链运行完成')
            if self.project_controller is not None and self._current_line_id:
                self._preview_newest_artifact = True
                self.project_controller.refresh_artifacts(self._current_line_id)
                self.project_controller.refresh_lines()   # 处理状态列可能变化
        else:
            self._infobar('error', '处理链', message or '处理链运行失败')

    def _on_autotune_requested(self, method_id: str, params_hint: dict) -> None:
        if self.processing_controller is None:
            return
        line_id = self._require_line()
        if not line_id:
            return
        self.processing_controller.run_autotune(
            self._current_project_id(), line_id, str(method_id),
            dict(params_hint or {}))
        self._infobar('info', 'AutoTune 自动调参', f'已提交调参任务：{method_id}')

    def _on_autotune_finished(self, method_id: str, result: dict) -> None:
        processing = self._page('processingInterface')
        if hasattr(processing, 'set_autotune_result'):
            processing.set_autotune_result(method_id, result)
        self._infobar('success', 'AutoTune 自动调参', f'调参完成：{method_id}')

    def _on_autotune_failed(self, method_id: str, message: str) -> None:
        self._infobar('error', 'AutoTune 自动调参',
                      f'{method_id}: {message}')

    # ============================================================ 解释页
    @staticmethod
    def _snapshot_points(snapshot) -> list:
        """InterfaceEditSnapshot → [(trace, sample), ...]（鸭子类型）。"""
        annotation = getattr(snapshot, 'annotation', None)
        points = getattr(annotation, 'points', ()) or ()
        result = []
        for point in points:
            if isinstance(point, (tuple, list)) and len(point) >= 2:
                result.append((int(point[0]), int(point[1])))
            else:
                result.append((int(getattr(point, 'trace_index', 0)),
                               int(getattr(point, 'sample_index', 0))))
        return result

    def _on_open_session_requested(self) -> None:
        if self.interpretation_controller is None:
            return
        line_id = self._require_line()
        if line_id:
            self.interpretation_controller.open_session(
                self._current_project_id(), line_id)

    def _on_session_opened(self, snapshot) -> None:
        interpretation = self._page('interpretationInterface')
        if not hasattr(interpretation, 'set_points'):
            return
        interpretation.set_points(self._snapshot_points(snapshot))
        line_id = str(getattr(snapshot, 'line_id', '') or self._current_line_id)
        interpretation.set_session_info(f'会话已打开（{line_id}）')
        self._infobar('success', '界面解释标注', f'标注会话已打开：{line_id}')
        # 确保剖面数据就绪（测线选择已触发预览，此处兜底刷新）
        if self.project_controller is not None and self._current_line_id:
            self.project_controller.preview_line(self._current_line_id)

    def _on_session_updated(self, snapshot) -> None:
        interpretation = self._page('interpretationInterface')
        if hasattr(interpretation, 'set_points'):
            interpretation.set_points(self._snapshot_points(snapshot))

    def _on_session_failed(self, message: str) -> None:
        self._infobar('error', '界面解释标注', message)

    def _on_points_changed(self, points: list) -> None:
        if self.interpretation_controller is not None:
            self.interpretation_controller.replace_points(list(points or []))

    def _on_annotation_saved(self, message: str) -> None:
        interpretation = self._page('interpretationInterface')
        if hasattr(interpretation, 'set_session_info'):
            interpretation.set_session_info('已保存')
        self._infobar('success', '界面解释标注', message or '标注已保存')

    # ============================================================ 成果页
    def _on_spatial_requested(self, payload: dict) -> None:
        if self.delivery_controller is None or not self._require_project():
            return
        payload = dict(payload or {})
        job_id = self.delivery_controller.create_spatial(
            self._current_project_id(),
            str(payload.get('name', '') or '空间成果'),
            list(payload.get('line_ids') or []))
        if job_id:
            self._spatial_job_ids.add(str(job_id))
            self._infobar('info', '空间成果', '空间成果任务已提交')

    def _on_report_requested(self, payload: dict) -> None:
        if self.delivery_controller is None or not self._require_project():
            return
        job_id = self.delivery_controller.generate_report(
            self._current_project_id(),
            str(dict(payload or {}).get('package_name', '') or ''))
        if job_id:
            self._infobar('info', '项目报告', '报告生成任务已提交')

    def _on_report_generated(self, result) -> None:
        delivery = self._page('deliveryInterface')
        if hasattr(delivery, 'set_report_result'):
            delivery.set_report_result(result)

    def _on_backup_requested(self, dest_dir: str) -> None:
        if self.delivery_controller is None or not self._require_project():
            return
        job_id = self.delivery_controller.backup_project(
            self._current_project_id(), str(dest_dir))
        if job_id:
            self._infobar('info', '项目备份', f'备份任务已提交 → {dest_dir}')

    def _on_restore_requested(self, archive_path: str) -> None:
        if not self._backend_ready:
            self._infobar('warning', '恢复备份', '后端尚未就绪，请稍后再试')
            return
        if self.delivery_controller is None:
            return
        dest_root = str(self.settings.get(
            'project_root', constants.DEFAULT_PROJECT_ROOT))
        job_id = self.delivery_controller.restore_project(
            str(archive_path), dest_root)
        if job_id:
            self._infobar('info', '恢复备份', f'恢复任务已提交 → {dest_root}')

    # ============================================================ 任务中心（JobBridge → 任务控件）
    def _job_views(self) -> tuple:
        """(JobTable, LogPanel.mini_jobs, HomePage.mini_jobs)（None 已过滤）。"""
        views = []
        jobs = self._page('jobsInterface')
        if hasattr(jobs, 'job_table'):
            views.append(jobs.job_table())
        if hasattr(self.log_panel, 'mini_jobs'):
            views.append(self.log_panel.mini_jobs())
        home = self._page('homeInterface')
        if hasattr(home, 'mini_jobs'):
            views.append(home.mini_jobs())
        return tuple(v for v in views if v is not None)

    def _upsert_job(self, job_id: str) -> None:
        if job_id in self._known_job_ids:
            return
        self._known_job_ids.add(job_id)
        title = job_id
        bridge = self._job_bridge()
        if bridge is not None:
            title = bridge.titles().get(job_id) or job_id
        for view in self._job_views():
            view.upsert_job(job_id, title)

    def _on_job_status(self, job_id: str, status: str) -> None:
        job_id = str(job_id)
        self._upsert_job(job_id)
        for view in self._job_views():
            view.set_status(job_id, str(status))
        if str(status) in ('completed', 'failed', 'cancelled'):
            for view in self._job_views():
                if hasattr(view, 'remove_inactive'):
                    view.remove_inactive()

    def _on_job_progress(self, job_id: str, completed: int, total: int,
                         message: str) -> None:
        job_id = str(job_id)
        self._upsert_job(job_id)
        for view in self._job_views():
            view.update_progress(job_id, int(completed), int(total), str(message))
        if job_id == self._processing_job_id:
            processing = self._page('processingInterface')
            if hasattr(processing, 'set_progress'):
                processing.set_progress(int(completed), int(total), str(message))

    def _on_job_completed(self, job_id: str, success: bool, message: str,
                          result) -> None:
        job_id = str(job_id)
        level = 'SUCCESS' if success else 'WARNING'
        self.log_message(
            f'{level} 任务 {job_id[:8]}… {"完成" if success else "结束"}：{message}')
        for view in self._job_views():
            if hasattr(view, 'remove_inactive'):
                view.remove_inactive()
        # 导入/同步完成 → 刷新测线列表（触发项目页/成果页/自动预览）
        if job_id in self._import_job_ids:
            self._import_job_ids.discard(job_id)
            if success and self.project_controller is not None:
                self.project_controller.refresh_lines()
        # 空间成果完成 → 刷新空间成果表
        if job_id in self._spatial_job_ids:
            self._spatial_job_ids.discard(job_id)
            project_id = self._current_project_id()
            if success and project_id and self.delivery_controller is not None:
                self.delivery_controller.refresh_spatial(project_id)

    def _cancel_job(self, job_id: str) -> None:
        bridge = self._job_bridge()
        if bridge is not None:
            bridge.cancel(str(job_id))
            self.log_message(f'INFO 已请求取消任务 {str(job_id)[:8]}…')

    def _on_prune_jobs(self) -> None:
        """清理已完成任务：backend.jobs.prune（工作线程，不阻塞 UI）。"""
        backend = getattr(self.backend_controller, 'backend', None) \
            if self.backend_controller is not None else None
        if backend is None:
            self._infobar('warning', '任务中心', '后端尚未就绪')
            return

        def runner() -> None:
            try:
                removed = backend.jobs.prune()
            except Exception as exc:  # noqa: BLE001
                logger.warning('jobs prune 失败: %s', exc)
                self._log_signal.emit(f'WARNING 清理已完成任务失败: {exc}')
            else:
                self._log_signal.emit(f'INFO 已清理 {len(removed)} 个终态任务记录')

        threading.Thread(target=runner, name='mygpr-jobs-prune',
                         daemon=True).start()

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
