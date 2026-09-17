# -*- coding: utf-8 -*-
"""PageCoordinator — 跨页业务信号链的薄门面（阶段 3 按域拆分）。

职责边界：
- ``main_window.py``：窗口组装（页面创建 / 导航 / 主题 / 面板 / 快捷键 /
  后端就绪门控）与项目对话框；
- 本模块：窗口服务接口与控制器透传的唯一入口，构造三个域子接线器并
  把 ``connect_all()`` 分三段委派：
    - ``project``   :class:`ui.coordinator_project.ProjectChain`
      项目生命周期、测线/成果/预览扇出、导入/预检/传感器同步、成果删除链；
    - ``processing`` :class:`ui.coordinator_processing.ProcessingChain`
      处理执行、AutoTune、速度分析、深度切片、B-scan 预览、解释会话；
    - ``jobs``      :class:`ui.coordinator_jobs.JobHub`
      任务事件对 JobTable/MiniJobList×2 三视图的同构扇出、取消与清理。

运行态已显式分给职责所有者（测线→ProjectChain、处理/velocity→
ProcessingChain、任务集合→JobHub）；本子接线器只经门面公共属性互访，
不得跨类读私有属性。``_current_line_id`` 等兼容属性仅服务于既有
单元测试资产（tests/test_page_coordinator.py）。

单条信号链 = 子接线器一个公共方法。除 Qt 信号连接外不依赖窗口内部实现，
仅通过鸭子类型的窗口服务接口访问窗口，因此每条链可用假窗口独立测试；
``connect_all()`` 是唯一的信号注册入口，由主窗口组装完成后调用一次。

本模块不得创建 QWidget（与 controllers 同一纪律；对话框见 ui/dialogs）。
"""
from __future__ import annotations

from ui.coordinator_jobs import JobHub
from ui.coordinator_processing import ProcessingChain
from ui.coordinator_project import ProjectChain


class PageCoordinator:
    """跨页业务信号链门面：三个域子接线器的组装点与窗口服务接口。"""

    def __init__(self, window) -> None:
        self._win = window
        self.project = ProjectChain(self)
        self.processing = ProcessingChain(self)
        self.jobs = JobHub(self)

    # ------------------------------------------------------------ 窗口服务接口
    # （鸭子类型，便于假窗口测试；子接线器经门面调用，不直接摸 window 私有）
    def page(self, object_name: str):
        return self._win._page(object_name)

    def infobar(self, level: str, title: str, content: str,
                duration: int = None) -> None:
        self._win._infobar(level, title, content, duration)

    def log_message(self, msg: str) -> None:
        self._win.log_message(msg)

    def goto_page(self, object_name: str) -> None:
        self._win._goto_page(object_name)

    def file_tree(self):
        """左侧常驻文件树面板（窗口组装时创建；offscreen 冒烟可能无）。"""
        return getattr(self._win, '_file_tree_panel', None)

    def show_new_project_dialog(self) -> None:
        self._win._show_new_project_dialog()

    def open_project_dialog(self) -> None:
        self._win._open_project_dialog()

    @property
    def backend_ready(self) -> bool:
        return self._win._backend_ready

    def current_project_id(self):
        return self._win._current_project_id()

    def require_project(self) -> bool:
        return self._win._require_project()

    def require_line(self) -> str:
        return self._win._require_line()

    def job_bridge(self):
        return self._win._job_bridge()

    def connect_job_bridge(self, bridge) -> None:
        """把 JobBridge 三个任务信号接到本接线器的槽（唯一接线点）。

        显式属性访问：槽位被改名/移走时立刻 AttributeError 暴露，
        而不是 hasattr 探测静默跳过（那次事故：load_methods 不执行 → 方法库为空）。
        """
        bridge.progress_changed.connect(self._on_job_progress)
        bridge.status_changed.connect(self._on_job_status)
        bridge.job_completed.connect(self._on_job_completed)

    def current_line_id(self) -> str:
        """当前测线（窗口 _require_line 的委托目标，避免读私有属性）。"""
        return self.project.current_line_id

    # ---- 控制器 / 设置 / 日志面板透传（子接线器经门面只读访问）----
    @property
    def backend_controller(self):
        return self._win.backend_controller

    @property
    def project_controller(self):
        return self._win.project_controller

    @property
    def processing_controller(self):
        return self._win.processing_controller

    @property
    def interpretation_controller(self):
        return self._win.interpretation_controller

    @property
    def delivery_controller(self):
        return self._win.delivery_controller

    @property
    def settings(self):
        return self._win.settings

    @property
    def output_panel(self):
        return self._win.output_panel

    @property
    def log_signal(self):
        return self._win._log_signal

    # ============================================================ 信号注册（唯一入口）
    def connect_all(self) -> None:
        """全量业务接线：三段委派给域子接线器（页面全部交付，显式属性访问；
        AttributeError 即真 bug，应暴露修复而非静默跳过）。"""
        self.project.connect_all()
        self.processing.connect_all()
        self.jobs.connect_all()

    # ============================================================ JobBridge 槽（门面保留）
    # tests/test_ui_migration_regressions.py 钉住这三个槽位必须存在于门面。
    def _on_job_progress(self, job_id: str, completed: int, total: int,
                         message: str) -> None:
        self.jobs.on_progress(job_id, completed, total, message)

    def _on_job_status(self, job_id: str, status: str) -> None:
        self.jobs.on_status(job_id, status)

    def _on_job_completed(self, job_id: str, success: bool, message: str,
                          result) -> None:
        self.jobs.on_completed(job_id, success, message, result)

    # ============================================================ 测试资产兼容委托
    # tests/test_page_coordinator.py 经门面驱动单链；覆盖不得减少。
    def _on_line_selected(self, line_id: str) -> None:
        self.project.on_line_selected(line_id)

    def _on_run_requested(self, payload: dict) -> None:
        self.processing.on_run_requested(payload)

    def _on_run_finished(self, success: bool, message: str) -> None:
        self.processing.on_run_finished(success, message)

    # ------------------------------------------------------------ 运行态兼容属性
    # 状态本体在子接线器（职责所有者）；以下属性保持旧字段名，
    # 仅为既有测试资产（coordinator._current_line_id 等直读直写）服务。
    @property
    def _current_line_id(self) -> str:
        return self.project.current_line_id

    @_current_line_id.setter
    def _current_line_id(self, value: str) -> None:
        self.project.current_line_id = value

    @property
    def _pending_select_line_id(self) -> str:
        return self.project.pending_select_line_id

    @_pending_select_line_id.setter
    def _pending_select_line_id(self, value: str) -> None:
        self.project.pending_select_line_id = value

    @property
    def _processing_job_id(self) -> str:
        return self.processing.processing_job_id

    @_processing_job_id.setter
    def _processing_job_id(self, value: str) -> None:
        self.processing.processing_job_id = value

    @property
    def _processing_line_id(self) -> str:
        return self.processing.processing_line_id

    @_processing_line_id.setter
    def _processing_line_id(self, value: str) -> None:
        self.processing.processing_line_id = value

    @property
    def _processing_cancel_requested(self) -> bool:
        return self.processing.processing_cancel_requested

    @_processing_cancel_requested.setter
    def _processing_cancel_requested(self, value: bool) -> None:
        self.processing.processing_cancel_requested = value

    @property
    def _show_run_completion_notice(self) -> bool:
        return self.processing.show_run_completion_notice

    @_show_run_completion_notice.setter
    def _show_run_completion_notice(self, value: bool) -> None:
        self.processing.show_run_completion_notice = value

    @property
    def _preview_newest_artifact(self) -> bool:
        return self.processing.preview_newest_artifact

    @_preview_newest_artifact.setter
    def _preview_newest_artifact(self, value: bool) -> None:
        self.processing.preview_newest_artifact = value

    @property
    def _velocity_token(self):
        return self.processing.velocity_token

    @_velocity_token.setter
    def _velocity_token(self, value) -> None:
        self.processing.velocity_token = value

    @property
    def _known_job_ids(self) -> set:
        return self.jobs.known_job_ids

    @property
    def _import_job_ids(self) -> set:
        return self.jobs.import_job_ids

    @property
    def _spatial_job_ids(self) -> set:
        return self.jobs.spatial_job_ids
