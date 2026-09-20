# -*- coding: utf-8 -*-
"""JobHub — 任务事件统一分发 的子接线器。

域归属（PageCoordinator 拆分的三段之一）：JobBridge 三个任务信号
（progress/status/completed）对 JobTable / OutputPanel.mini_jobs /
HomePage.mini_jobs 三视图的同构扇出，以及任务取消、终态清理（prune）。

三个视图实现同一协议（upsert_job/set_status/update_progress/
remove_inactive），本类是唯一分发点；运行态（已知任务、导入/空间成果
任务集合）的本类是唯一所有者。

历史语义：任务中心页 JobTable 保留终态行（"清理已完成"手动清理）；
两个迷你视图只显示活动任务，终态即移除。

本模块不得创建 QWidget（与 controllers 同一纪律）。
"""
from __future__ import annotations

import time

# 任务取消确认阈值：刚提交的任务直接取消（误触代价低），已运行超过
# 该时长的任务先弹确认框（误触中断数小时计算，代价高）。
CANCEL_CONFIRM_AFTER_S = 30.0


def _ask_cancel_dialog(parent, title: str, elapsed_s: float) -> bool:
    """长任务取消确认框（对话框在 ui.dialogs；包装成模块级函数便于无 Qt 打桩）。

    返回 True = 用户确认取消。
    """
    from ui.dialogs import ask_cancel_job
    return ask_cancel_job(parent, title, elapsed_s)


class JobHub:
    """任务事件分发枢纽；纯 Python 接线类，构造时拿 PageCoordinator 门面。"""

    def __init__(self, coordinator) -> None:
        self._co = coordinator
        self.known_job_ids: set[str] = set()   # 已 upsert 到任务控件的任务
        self.import_job_ids: set[str] = set()  # 测线导入/传感器同步任务（完成后刷新测线）
        self.spatial_job_ids: set[str] = set()  # 空间成果任务（完成后刷新空间成果表）
        # job_id → 首次出现时刻（monotonic 秒）：取消确认框的"已运行时长"依据
        self._job_first_seen: dict[str, float] = {}

    # ============================================================ 信号注册
    def connect_all(self) -> None:
        """任务域接线：任务页 / 输出面板 / 主页迷你任务列表。"""
        co = self._co
        jobs = co.page('jobsInterface')
        jobs.cancel_requested.connect(self.on_cancel)
        jobs.prune_requested.connect(self.on_prune_jobs)

        output_panel = co.output_panel
        if output_panel is not None:
            output_panel.cancel_job_requested.connect(self.on_cancel)
            mini = output_panel.mini_jobs()
            if mini is not None:
                mini.job_clicked.connect(self.on_mini_job_clicked)

        home_jobs = co.page('homeInterface').mini_jobs()
        if home_jobs is not None:
            home_jobs.cancel_requested.connect(self.on_cancel)
            home_jobs.job_clicked.connect(self.on_mini_job_clicked)

    # ============================================================ 迷你列表点击定位
    def on_mini_job_clicked(self, job_id: str) -> None:
        """迷你任务列表行点击 → 跳任务中心页并选中定位该任务。"""
        co = self._co
        co.goto_page('jobsInterface')
        co.page('jobsInterface').job_table().focus_job(str(job_id))

    # ============================================================ 视图扇出
    def _views(self) -> tuple:
        """(JobTable, OutputPanel.mini_jobs, HomePage.mini_jobs)（None 已过滤）。"""
        co = self._co
        views = [co.page('jobsInterface').job_table()]
        output_panel = co.output_panel
        if output_panel is not None:
            views.append(output_panel.mini_jobs())
        views.append(co.page('homeInterface').mini_jobs())
        return tuple(v for v in views if v is not None)

    def _mini_views(self) -> tuple:
        """只显示活动任务的迷你视图（OutputPanel + 主页；None 已过滤）。

        终态自动移除只扇出到这里——任务中心页 JobTable 保留历史。
        """
        co = self._co
        views = []
        output_panel = co.output_panel
        if output_panel is not None:
            views.append(output_panel.mini_jobs())
        views.append(co.page('homeInterface').mini_jobs())
        return tuple(v for v in views if v is not None)

    def _upsert(self, job_id: str) -> None:
        if job_id in self.known_job_ids:
            return
        self.known_job_ids.add(job_id)
        self._job_first_seen.setdefault(job_id, time.monotonic())
        title = job_id
        bridge = self._co.job_bridge()
        if bridge is not None:
            title = bridge.titles().get(job_id) or job_id
        for view in self._views():
            view.upsert_job(job_id, title)

    # ============================================================ 任务事件
    def on_status(self, job_id: str, status: str) -> None:
        job_id = str(job_id)
        self._upsert(job_id)
        for view in self._views():
            view.set_status(job_id, str(status))
        if str(status) in ('completed', 'failed', 'cancelled'):
            for view in self._mini_views():
                view.remove_inactive()

    def on_progress(self, job_id: str, completed: int, total: int,
                    message: str) -> None:
        job_id = str(job_id)
        self._upsert(job_id)
        for view in self._views():
            view.update_progress(job_id, int(completed), int(total), str(message))
        if job_id == self._co.processing.processing_job_id:
            self._co.page('processingInterface').set_progress(
                int(completed), int(total), str(message))

    def on_completed(self, job_id: str, success: bool, message: str,
                     result) -> None:
        co = self._co
        job_id = str(job_id)
        self._job_first_seen.pop(job_id, None)   # 首现时刻只服务活动任务
        if success:
            co.log_message(
                f'SUCCESS 任务 {job_id[:8]}… 完成：{message}')
        else:
            co.log_message(
                f'WARNING 任务 {job_id[:8]}… 结束：{message}')
            # P0-1：失败必须显式反馈；处理任务由 run_finished 单独弹窗，避免重复
            if job_id != co.processing.processing_job_id:
                co.infobar('error', '任务失败',
                           message or '任务执行失败，详见任务页', duration=8000)
        status = 'completed' if success else 'failed'
        for view in self._views():
            view.set_status(job_id, status)
        # 导入/同步完成 → 刷新测线列表
        if job_id in self.import_job_ids:
            self.import_job_ids.discard(job_id)
            if success and co.project_controller is not None:
                co.project_controller.refresh_lines()
        # 空间成果完成 → 刷新空间成果表
        if job_id in self.spatial_job_ids:
            self.spatial_job_ids.discard(job_id)
            project_id = co.current_project_id()
            if success and project_id and co.delivery_controller is not None:
                co.delivery_controller.refresh_spatial(project_id)

    # ============================================================ 取消 / 清理
    def on_cancel(self, job_id: str) -> None:
        """取消任务：长任务先确认（dialogs），取消失败显式反馈（不再一律
        宣称"已请求取消"——backend 侧任务可能已结束/不存在）。"""
        bridge = self._co.job_bridge()
        if bridge is None:
            return
        job_id = str(job_id)
        if not job_id:
            return
        title = bridge.titles().get(job_id) or f'{job_id[:8]}…'
        seen = self._job_first_seen.get(job_id)
        elapsed = time.monotonic() - seen if seen is not None else 0.0
        if elapsed >= CANCEL_CONFIRM_AFTER_S:
            if not _ask_cancel_dialog(self._co.page('jobsInterface'),
                                      title, elapsed):
                return
        if job_id == self._co.processing.processing_job_id:
            # 从任务页/日志面板取消处理任务同样按"已取消"提示，而非失败
            self._co.processing.processing_cancel_requested = True
        if bridge.cancel(job_id) is False:
            # 显式 False = 后端拒绝（任务已结束/不存在）；None 视为旧实现成功
            self._co.log_message(f'INFO 任务 {title} 已结束，无需取消')
            self._co.infobar('warning', '任务中心',
                             f'任务「{title}」无法取消（可能已结束）')
            return
        self._co.log_message(f'INFO 已请求取消任务 {job_id[:8]}…')

    def on_prune_jobs(self) -> None:
        """清理已完成任务：BackendController.prune_jobs（工作线程，不阻塞 UI）。"""
        backend_controller = self._co.backend_controller
        if backend_controller is None or not backend_controller.prune_jobs():
            self._co.infobar('warning', '任务中心', '后端尚未就绪')
