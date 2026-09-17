#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project lifecycle / line import controller (QObject, no QWidget)."""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from ui.desktop_backend_facade import SensorSyncSettings, build_preview_bundle
from ui.controllers.backend_controller import friendly_error_message, run_command

_LOGGER = logging.getLogger(__name__)

_PREVIEW_MAX_SAMPLES = 900
_PREVIEW_MAX_TRACES = 1800


class ProjectController(QObject):
    """Mediates project open/close/import/preview between UI and backend."""

    log_message = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)
    project_opened = pyqtSignal(object)          # ProjectSummary
    project_closed = pyqtSignal()
    open_failed = pyqtSignal(str)
    lines_updated = pyqtSignal(list)             # list[ProjectLine]
    artifacts_updated = pyqtSignal(str, list)    # line_id, list[ProjectArtifact]
    dataset_preview_ready = pyqtSignal(object)   # PreviewBundle (raw data)
    artifact_preview_ready = pyqtSignal(str, object)  # artifact_id, PreviewBundle
    preflight_ready = pyqtSignal(object)         # ImportPreflight
    preflight_failed = pyqtSignal(str)           # 导入预检失败消息
    preview_invalidated = pyqtSignal()           # 预览中的成果已被删除
    spatial_tracks_ready = pyqtSignal(list)      # list[SpatialTrack]
    depth_preview_ready = pyqtSignal(object, list, float, int)  # payload, line_ids, cell_size_m, generation
    depth_layer_saved = pyqtSignal(str, list, float)       # job_id, line_ids, cell_size_m
    depth_save_failed = pyqtSignal(str)                    # message
    artifact_descendants_ready = pyqtSignal(str, list, dict)  # line_id, 后代闭包, {artifact_id: 名称}
    line_source_path_ready = pyqtSignal(str, object)          # line_id, 源文件路径|None

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._backend_controller = None
        self._current = None
        self._busy = False
        # 预览代数：每次发起新预览自增，worker 回包时若代数已过期则丢弃，
        # 防止快速切换测线时旧预览覆盖新预览。
        self._preview_generation = 0
        # 成果预览代数：与测线/深度预览独立计数（不同预览域互不失效）。
        # 共享计数会让处理链运行完成后的自动成果预览被并发的测线预览
        # （refresh_lines → 表格重选 → on_line_selected）作废，预览区
        # 停留在"暂无数据"。
        self._artifact_preview_generation = 0
        # 深度切片预览代数：与数据/成果预览独立计数（不同预览域互不失效）。
        self._depth_preview_generation = 0
        # 当前预览中的成果 id（删除该成果时需清空各页预览）
        self._current_preview_artifact_id = ''

    # ------------------------------------------------------------------
    def set_backend(self, backend_controller) -> None:
        self._backend_controller = backend_controller

    @property
    def current_project(self):
        return self._current

    @property
    def current_project_id(self) -> str | None:
        return self._current.project_id if self._current is not None else None

    # ------------------------------------------------------------------
    def _backend(self):
        controller = self._backend_controller
        backend = getattr(controller, "backend", None) if controller is not None else None
        if backend is None:
            self.log_message.emit("后端尚未就绪，请稍后再试")
        return backend

    def _job_bridge(self):
        return getattr(self._backend_controller, "job_bridge", None)

    def _set_busy(self, value: bool) -> None:
        value = bool(value)
        if value != self._busy:
            self._busy = value
            self.busy_changed.emit(value)

    def _project_id_or_warn(self) -> str | None:
        project_id = self.current_project_id
        if project_id is None:
            self.log_message.emit("请先在主页打开或新建项目")
        return project_id

    def create_project(self, root: str, name: str, meta: dict) -> None:
        if self._busy:
            self.log_message.emit("操作进行中，请稍后…")
            return
        # 新建项目同样切换项目上下文，使旧项目在途预览回包过期。
        self._depth_preview_generation += 1
        self._preview_generation += 1
        self._artifact_preview_generation += 1
        self._set_busy(True)
        run_command(
            _CreateProjectCommand(self, root, name, dict(meta or {})),
            name="mygpr-project-create",
        )

    def open_project(self, root: str) -> None:
        if self._busy:
            self.log_message.emit("操作进行中，请稍后…")
            return
        # 打开/切换项目前使旧项目在途预览回包过期（A→B 直切时 A 的
        # 回包代数与新项目相同，不推进则门卫会放行旧 payload）。
        self._depth_preview_generation += 1
        self._preview_generation += 1
        self._artifact_preview_generation += 1
        self._set_busy(True)
        run_command(
            _OpenProjectCommand(self, root),
            name="mygpr-project-open",
        )

    def close_current(self) -> None:
        project_id = self._project_id_or_warn()
        if project_id is None:
            return
        if self._busy:
            self.log_message.emit("操作进行中，请稍后…")
            return
        # 关闭前使在途预览回包过期，防止旧项目 payload 渲染进后续视图。
        self._depth_preview_generation += 1
        self._preview_generation += 1
        self._artifact_preview_generation += 1
        self._set_busy(True)
        run_command(
            _CloseProjectCommand(self, project_id),
            name="mygpr-project-close",
        )

    def invalidate_depth_previews(self) -> None:
        """使在途深度预览回包过期（项目关闭/测线变更等场景调用）。"""
        self._depth_preview_generation += 1

    def invalidate_artifact_previews(self) -> None:
        """使在途成果预览回包过期（切换测线/项目上下文等场景调用）。

        切换测线后在飞的旧测线成果预览不得渲染进新测线的"处理结果"
        分段；同测线重复选中（测线表重建重选）则不得作废在飞预览。
        """
        self._artifact_preview_generation += 1

    @property
    def depth_preview_generation(self) -> int:
        """深度预览当前代数（只读）：接线层交付门卫用，推进只在本类内部。"""
        return self._depth_preview_generation

    # ------------------------------------------------------------------
    def line_source_path(self, line_id: str) -> None:
        """异步查询当前项目某测线的源数据文件路径（右键菜单"复制路径/
        打开所在文件夹"素材）。

        读项目根 ``raw/<line_id>/import_manifest.json`` 的 ``source_path``
        （导入时由 field_line_store 持久化）；小文件 I/O 放到 worker 线程，
        结果经 ``line_source_path_ready(line_id, path|None)`` 回 GUI 线程。
        无项目/无清单/无字段时回 None。
        """
        line_id = str(line_id or '')
        if not line_id:
            return
        if self._current is None:
            self.line_source_path_ready.emit(line_id, None)
            return
        manifest = (Path(self._current.root_path) / 'raw' / line_id
                    / 'import_manifest.json')
        run_command(
            _LineSourcePathCommand(self, line_id, manifest),
            name="mygpr-line-source-path",
        )

    def refresh_lines(self) -> None:
        backend = self._backend()
        project_id = self.current_project_id
        if backend is None:
            return
        if project_id is None:
            self.lines_updated.emit([])
            return

        run_command(
            _RefreshLinesCommand(self, project_id),
            name="mygpr-lines-refresh",
        )

    def refresh_artifacts(self, line_id: str) -> None:
        backend = self._backend()
        project_id = self._project_id_or_warn()
        if backend is None or project_id is None:
            return
        line_id = str(line_id)

        run_command(
            _RefreshArtifactsCommand(self, project_id, line_id),
            name="mygpr-artifacts-refresh",
        )

    # ------------------------------------------------------------------
    def load_spatial_tracks(self) -> None:
        backend = self._backend()
        project_id = self.current_project_id
        if backend is None:
            return
        if project_id is None:
            self.spatial_tracks_ready.emit([])
            return

        run_command(
            _LoadSpatialTracksCommand(self, project_id),
            name="mygpr-spatial-tracks",
        )

    def preview_line(self, line_id: str) -> None:
        backend = self._backend()
        project_id = self._project_id_or_warn()
        if backend is None or project_id is None:
            return
        line_id = str(line_id)
        self._preview_generation += 1

        run_command(
            _PreviewLineCommand(self, project_id, line_id, self._preview_generation),
            name="mygpr-line-preview",
        )

    def preview_artifact(self, line_id: str, artifact_id: str) -> None:
        backend = self._backend()
        project_id = self._project_id_or_warn()
        if backend is None or project_id is None:
            return
        line_id = str(line_id)
        artifact_id = str(artifact_id)
        self._current_preview_artifact_id = artifact_id
        self._artifact_preview_generation += 1

        run_command(
            _PreviewArtifactCommand(self, project_id, line_id, artifact_id,
                                    self._artifact_preview_generation),
            name="mygpr-artifact-preview",
        )

    @staticmethod
    def _bundle_from_window(
        matrix: Any,
        line_id: str,
        *,
        title: str = "",
        time_window_ns: float = 250.0,
        length_m: float | None = None,
        sample_indices: Any = None,
        trace_indices: Any = None,
        total_samples: int = 0,
        total_traces: int = 0,
    ) -> Any:
        """从 read_window 结果构建预览 bundle。

        P1-5：透传真实 time_window_ns 与通过 sample/trace_indices 重建的物理轴，
        避免所有页面预览纵轴硬编码 250ns 导致刻度错误。
        """
        time_axis_ns = None
        distance_axis_m = None
        if sample_indices is not None and total_samples > 0:
            base = np.linspace(0.0, float(time_window_ns), int(total_samples), dtype=np.float32)
            time_axis_ns = base[np.asarray(sample_indices, dtype=np.int64)]
        if trace_indices is not None and total_traces > 0:
            base = np.linspace(0.0, float(length_m or max(int(total_traces) - 1, 1)), int(total_traces), dtype=np.float32)
            distance_axis_m = base[np.asarray(trace_indices, dtype=np.int64)]
        return build_preview_bundle(
            line_id=line_id,
            matrix=np.asarray(matrix, dtype=np.float32),
            title=title,
            time_window_ns=float(time_window_ns),
            time_axis_ns=time_axis_ns,
            distance_axis_m=distance_axis_m,
        )

    # ------------------------------------------------------------------
    def preflight_import(self, source: str, line_id: str, dielectric: float) -> None:
        run_command(
            _PreflightImportCommand(self, source, line_id, dielectric),
            name="mygpr-import-preflight",
        )

    def import_line(self, source: str, line_id: str, name: str, dielectric: float) -> str | None:
        backend = self._backend()
        bridge = self._job_bridge()
        project_id = self._project_id_or_warn()
        if backend is None or bridge is None or project_id is None:
            return None
        line_id = str(line_id or "L01")
        try:
            job_id = backend.submit_line_import(
                project_id,
                str(source),
                line_id=line_id,
                name=str(name or line_id),
                dielectric_constant=float(dielectric),
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("测线导入提交失败")
            self.log_message.emit(f"测线导入提交失败：{friendly_error_message(exc)}")
            return None
        self.log_message.emit(f"测线导入已提交：{line_id}")
        bridge.watch(job_id, title=f"导入测线 {line_id}")
        return job_id

    def sync_sensors(self, line_id: str, paths: dict, settings: dict) -> str | None:
        backend = self._backend()
        bridge = self._job_bridge()
        project_id = self._project_id_or_warn()
        if backend is None or bridge is None or project_id is None:
            return None
        line_id = str(line_id)
        paths = dict(paths or {})
        rtk_path = str(paths.get("rtk") or "").strip()
        if not rtk_path:
            self.log_message.emit("传感器同步需要 RTK 文件")
            return None
        settings_obj: SensorSyncSettings | None = None
        if settings:
            valid = {field.name for field in dataclasses.fields(SensorSyncSettings)}
            settings_obj = SensorSyncSettings(
                **{key: value for key, value in dict(settings).items() if key in valid}
            )

        def _optional(key: str) -> str | None:
            value = str(paths.get(key) or "").strip()
            return value or None

        try:
            job_id = backend.submit_sensor_sync(
                project_id,
                line_id,
                rtk_path=rtk_path,
                trace_timestamps_path=_optional("trace_timestamps"),
                imu_path=_optional("imu"),
                altimeter_path=_optional("altimeter"),
                settings=settings_obj,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("传感器同步提交失败")
            self.log_message.emit(f"传感器同步提交失败：{friendly_error_message(exc)}")
            return None
        self.log_message.emit(f"传感器同步已提交：{line_id}")
        bridge.watch(job_id, title=f"传感器同步 {line_id}")
        return job_id

    def delete_lines(self, line_ids: list[str], *, reason: str = "batch-delete") -> None:
        """批量删除当前项目中的多条测线（项目页 Delete 键入口）。"""
        backend = self._backend()
        project_id = self._project_id_or_warn()
        if backend is None or project_id is None:
            return
        line_ids = [str(lid) for lid in (line_ids or []) if lid]
        if not line_ids:
            return
        self._set_busy(True)
        run_command(
            _DeleteLinesCommand(self, project_id, line_ids, reason),
            name="mygpr-lines-delete",
        )

    def delete_artifacts(self, line_id: str, artifact_ids: list[str]) -> None:
        """删除成果（级联名单已由 coordinator 确认）→ 移入项目回收站。"""
        backend = self._backend()
        project_id = self._project_id_or_warn()
        if backend is None or project_id is None:
            return
        ids = [str(a) for a in (artifact_ids or []) if a]
        if not ids or not str(line_id):
            return
        self._set_busy(True)
        run_command(
            _DeleteArtifactsCommand(self, project_id, str(line_id), ids),
            name="mygpr-artifacts-delete",
        )

    def get_artifact_descendants(self, line_id: str,
                                 artifact_ids: list[str]) -> None:
        """异步查询某成果（或多个）的后代闭包（含自身）+ 名称表。

        供级联确认框：结果经 ``artifact_descendants_ready(line_id,
        descendants, names)`` 回 GUI 线程；查询失败只记日志（与原同步版
        返回 [] 一致，不弹确认框）。
        """
        backend = self._backend()
        project_id = self._project_id_or_warn()
        if backend is None or project_id is None:
            return
        ids = [str(a) for a in (artifact_ids or []) if a]
        if not ids or not str(line_id):
            return
        run_command(
            _ArtifactDescendantsCommand(self, project_id, str(line_id), ids),
            name="mygpr-artifact-descendants",
        )

    def request_depth_preview(self, line_ids: list[str], cell_size_m: float = 1.0) -> None:
        """离线程请求界面深度切片预览；带代数 token，旧回包按代数丢弃。"""
        backend = self._backend()
        project_id = self._project_id_or_warn()
        if backend is None or project_id is None:
            return
        line_ids = [str(lid) for lid in (line_ids or []) if lid]
        if not line_ids:
            return
        self._depth_preview_generation += 1
        run_command(
            _DepthPreviewCommand(self, project_id, line_ids,
                                 float(cell_size_m or 1.0),
                                 self._depth_preview_generation),
            name="mygpr-depth-preview",
        )

    def submit_depth_layer(self, line_ids: list[str], cell_size_m: float) -> str | None:
        """提交网格化界面深度图层 job；bridge.watch 后经 depth_layer_saved 回 UI。"""
        backend = self._backend()
        bridge = self._job_bridge()
        project_id = self._project_id_or_warn()
        if backend is None or bridge is None or project_id is None:
            return None
        line_ids = [str(lid) for lid in (line_ids or []) if lid]
        if not line_ids:
            return None
        try:
            job_id = backend.submit_grid_layer(
                project_id, line_ids, cell_size_m=float(cell_size_m or 1.0))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("深度图层提交失败")
            message = friendly_error_message(exc)
            self.log_message.emit(f"深度图层提交失败：{message}")
            self.depth_save_failed.emit(message)
            return None
        self.log_message.emit(f"深度图层任务已提交：{job_id[:8]}…")
        bridge.watch(job_id, title="网格化界面深度图层")

        def _on_done(job_id_: str, success: bool, message: str, _result: object) -> None:
            if job_id_ != job_id:
                return
            try:
                bridge.job_completed.disconnect(_on_done)
            except TypeError:
                pass
            if success:
                self.depth_layer_saved.emit(job_id_, line_ids, float(cell_size_m or 1.0))
            else:
                self.depth_save_failed.emit(message or "深度图层任务失败")

        bridge.job_completed.connect(_on_done)
        return job_id


# ------------------------------------------------------------------
# Worker commands (replaces run_worker closures)
# ------------------------------------------------------------------

class _CreateProjectCommand:
    __slots__ = ("_controller", "_root", "_name", "_meta")

    def __init__(self, controller: ProjectController, root: str, name: str, meta: dict) -> None:
        self._controller = controller
        self._root = root
        self._name = name
        self._meta = meta

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            c.open_failed.emit("后端尚未就绪")
            c._set_busy(False)
            return
        meta = self._meta
        try:
            summary = backend.projects.create_project(
                str(self._root),
                name=str(self._name),
                location=str(meta.get("location", "")),
                operator=str(meta.get("operator") or "操作员"),
                project_no=str(meta.get("project_no", "")),
                device_model=str(meta.get("device_model", "")),
                coordinate_system=str(meta.get("coordinate_system", "")),
                vertical_datum=str(meta.get("vertical_datum", "")),
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("新建项目失败")
            message = friendly_error_message(exc)
            c.log_message.emit(f"新建项目失败：{message}")
            c.open_failed.emit(message)
        else:
            # 与 _OpenProjectCommand 同理：项目上下文切换时刻推进代数。
            c._depth_preview_generation += 1
            c._preview_generation += 1
            c._artifact_preview_generation += 1
            c._current = summary
            c.log_message.emit(f"项目已创建：{summary.name}")
            c.project_opened.emit(summary)
            c.refresh_lines()
        finally:
            c._set_busy(False)


class _OpenProjectCommand:
    __slots__ = ("_controller", "_root")

    def __init__(self, controller: ProjectController, root: str) -> None:
        self._controller = controller
        self._root = root

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            c.open_failed.emit("后端尚未就绪")
            c._set_busy(False)
            return
        try:
            # 显式打开项目=人类操作；锁的持有进程已由 _pid_alive 判死时
            # 恢复陈旧锁（默认 False 会让 AUTO 静默降级为只读会话，
            # 处理链写 catalog 时才报 "Project catalog is read-only"）。
            summary = backend.projects.open_project(
                str(self._root), recover_stale_lock=True)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("打开项目失败")
            message = friendly_error_message(exc)
            c.log_message.emit(f"打开项目失败：{message}")
            c.open_failed.emit(message)
        else:
            # 项目上下文真正切换的时刻再次推进代数：busy 窗口内发起的
            # 预览请求仍指向旧项目，其回包代数可能与 open 前的推进撞车，
            # 不在此推进则门卫会放行旧项目 payload。
            c._depth_preview_generation += 1
            c._preview_generation += 1
            c._artifact_preview_generation += 1
            c._current = summary
            c.log_message.emit(f"项目已打开：{summary.name}")
            c.project_opened.emit(summary)
            c.refresh_lines()
        finally:
            c._set_busy(False)


class _CloseProjectCommand:
    __slots__ = ("_controller", "_project_id")

    def __init__(self, controller: ProjectController, project_id: str) -> None:
        self._controller = controller
        self._project_id = project_id

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            c._set_busy(False)
            return
        try:
            backend.projects.close_project(self._project_id, force=False)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("关闭项目失败")
            c.log_message.emit(f"关闭项目失败：{friendly_error_message(exc)}")
        else:
            # 与 open/create 成功路径同理：busy 窗口内发起的预览请求
            # 仍指向本项目，close_current 的前置推进可被其自增抵消；
            # 项目上下文清除时刻再次推进，确保 in-flight 回包全部过期。
            c._depth_preview_generation += 1
            c._preview_generation += 1
            c._artifact_preview_generation += 1
            c._current = None
            c.log_message.emit("项目已关闭")
            c.project_closed.emit()
        finally:
            c._set_busy(False)


class _RefreshLinesCommand:
    __slots__ = ("_controller", "_project_id")

    def __init__(self, controller: ProjectController, project_id: str) -> None:
        self._controller = controller
        self._project_id = project_id

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            return
        try:
            lines = list(backend.projects.list_lines(self._project_id))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("刷新测线列表失败")
            c.log_message.emit(f"刷新测线列表失败：{friendly_error_message(exc)}")
        else:
            c.lines_updated.emit(lines)


class _RefreshArtifactsCommand:
    __slots__ = ("_controller", "_project_id", "_line_id")

    def __init__(self, controller: ProjectController, project_id: str, line_id: str) -> None:
        self._controller = controller
        self._project_id = project_id
        self._line_id = line_id

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            return
        try:
            artifacts = list(backend.projects.list_artifacts(self._project_id, self._line_id))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("刷新成果列表失败")
            c.log_message.emit(f"刷新成果列表失败：{friendly_error_message(exc)}")
        else:
            c.artifacts_updated.emit(self._line_id, artifacts)


class _LoadSpatialTracksCommand:
    __slots__ = ("_controller", "_project_id")

    def __init__(self, controller: ProjectController, project_id: str) -> None:
        self._controller = controller
        self._project_id = project_id

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            return
        try:
            tracks = list(backend.spatial.load_tracks(self._project_id))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("加载空间轨迹失败")
            c.log_message.emit(f"加载空间轨迹失败：{friendly_error_message(exc)}")
        else:
            c.spatial_tracks_ready.emit(tracks)


class _PreviewLineCommand:
    __slots__ = ("_controller", "_project_id", "_line_id", "_generation")

    def __init__(
        self,
        controller: ProjectController,
        project_id: str,
        line_id: str,
        generation: int,
    ) -> None:
        self._controller = controller
        self._project_id = project_id
        self._line_id = line_id
        self._generation = generation

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            return
        try:
            info = backend.projects.get_dataset_info(self._project_id, self._line_id)
            matrix, sample_idx, trace_idx = backend.projects.read_window(
                self._project_id,
                self._line_id,
                max_samples=_PREVIEW_MAX_SAMPLES,
                max_traces=_PREVIEW_MAX_TRACES,
            )
            bundle = c._bundle_from_window(
                matrix, self._line_id, title=f"测线 {self._line_id}",
                time_window_ns=info.time_window_ns, length_m=info.length_m,
                sample_indices=sample_idx, trace_indices=trace_idx,
                total_samples=info.shape[0], total_traces=info.shape[1],
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("数据预览失败")
            c.log_message.emit(f"数据预览失败：{friendly_error_message(exc)}")
        else:
            if c._preview_generation != self._generation:
                return
            c.dataset_preview_ready.emit(bundle)


class _PreviewArtifactCommand:
    __slots__ = ("_controller", "_project_id", "_line_id", "_artifact_id", "_generation")

    def __init__(
        self,
        controller: ProjectController,
        project_id: str,
        line_id: str,
        artifact_id: str,
        generation: int,
    ) -> None:
        self._controller = controller
        self._project_id = project_id
        self._line_id = line_id
        self._artifact_id = artifact_id
        self._generation = generation

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            return
        try:
            info = backend.projects.get_artifact_dataset_info(
                self._project_id, self._line_id, self._artifact_id)
            matrix, sample_idx, trace_idx = backend.projects.read_artifact_window(
                self._project_id,
                self._line_id,
                self._artifact_id,
                max_samples=_PREVIEW_MAX_SAMPLES,
                max_traces=_PREVIEW_MAX_TRACES,
            )
            bundle = c._bundle_from_window(
                matrix, self._line_id, title=f"成果 {self._artifact_id}",
                time_window_ns=info.time_window_ns, length_m=info.length_m,
                sample_indices=sample_idx, trace_indices=trace_idx,
                total_samples=info.shape[0], total_traces=info.shape[1],
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("成果预览失败")
            c.log_message.emit(f"成果预览失败：{friendly_error_message(exc)}")
        else:
            if c._artifact_preview_generation != self._generation:
                return
            c.artifact_preview_ready.emit(self._artifact_id, bundle)


class _PreflightImportCommand:
    __slots__ = ("_controller", "_source", "_line_id", "_dielectric")

    def __init__(
        self,
        controller: ProjectController,
        source: str,
        line_id: str,
        dielectric: float,
    ) -> None:
        self._controller = controller
        self._source = source
        self._line_id = line_id
        self._dielectric = dielectric

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            c.preflight_failed.emit("后端尚未就绪")
            return
        try:
            result = backend.acquisition.preflight(
                str(self._source),
                line_id=str(self._line_id or "L01"),
                dielectric_constant=float(self._dielectric),
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("导入预检失败")
            message = friendly_error_message(exc)
            c.log_message.emit(f"导入预检失败：{message}")
            c.preflight_failed.emit(message)
        else:
            c.preflight_ready.emit(result)


class _DeleteLinesCommand:
    __slots__ = ("_controller", "_project_id", "_line_ids", "_reason")

    def __init__(
        self,
        controller: ProjectController,
        project_id: str,
        line_ids: list[str],
        reason: str = "batch-delete",
    ) -> None:
        self._controller = controller
        self._project_id = project_id
        self._line_ids = line_ids
        self._reason = reason

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            c._set_busy(False)
            return
        try:
            for line_id in self._line_ids:
                backend.maintenance.delete_line(self._project_id, line_id, reason=self._reason)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("删除测线失败")
            c.log_message.emit(f"删除测线失败：{friendly_error_message(exc)}")
        else:
            c.log_message.emit(f"已删除 {len(self._line_ids)} 条测线")
        finally:
            c._set_busy(False)
            c.refresh_lines()


class _DeleteArtifactsCommand:
    """worker 线程删除成果（移入回收站）；成功后刷新成果列表。"""

    __slots__ = ("_controller", "_project_id", "_line_id", "_artifact_ids")

    def __init__(
        self,
        controller: ProjectController,
        project_id: str,
        line_id: str,
        artifact_ids: list[str],
    ) -> None:
        self._controller = controller
        self._project_id = project_id
        self._line_id = line_id
        self._artifact_ids = artifact_ids

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            c._set_busy(False)
            return
        try:
            result = backend.delete_artifacts(
                self._project_id, self._line_id, list(self._artifact_ids))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("删除成果失败")
            c.log_message.emit(f"删除成果失败：{friendly_error_message(exc)}")
        else:
            deleted = list(result.deleted_artifact_ids)
            c.log_message.emit(
                f"已删除 {len(deleted)} 个成果（含级联），已移入项目回收站")
            if c._current_preview_artifact_id in deleted:
                c._current_preview_artifact_id = ''
                c.preview_invalidated.emit()
        finally:
            c._set_busy(False)
            try:
                artifacts = list(backend.projects.list_artifacts(
                    self._project_id, self._line_id))
            except Exception:  # noqa: BLE001
                _LOGGER.exception("刷新成果列表失败", exc_info=True)
                return
            c.artifacts_updated.emit(self._line_id, artifacts)


class _DepthPreviewCommand:
    """worker 线程执行 interface_depth_preview，结果经 depth_preview_ready 回 UI。"""

    __slots__ = ("_controller", "_project_id", "_line_ids", "_cell_size_m", "_generation")

    def __init__(
        self,
        controller: ProjectController,
        project_id: str,
        line_ids: list[str],
        cell_size_m: float,
        generation: int,
    ) -> None:
        self._controller = controller
        self._project_id = project_id
        self._line_ids = line_ids
        self._cell_size_m = cell_size_m
        self._generation = generation

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            return
        try:
            payload = backend.interface_depth_preview(
                self._project_id,
                self._line_ids,
                cell_size_m=self._cell_size_m,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("深度切片预览失败")
            c.log_message.emit(f"深度切片预览失败：{friendly_error_message(exc)}")
        else:
            # 双层防护 ①：worker 线程先按代数丢弃明显过期的回包。
            # 接收端还会按发射时快照的代数二次校验（close/open 推进
            # 代数可能发生在 emit 排队之后、slot 执行之前）。
            if c._depth_preview_generation != self._generation:
                return
            c.depth_preview_ready.emit(
                payload, list(self._line_ids), self._cell_size_m, self._generation)


class _LineSourcePathCommand:
    """worker 线程读 import_manifest.json 的 source_path（小文件 I/O 离开 UI 线程）。"""

    __slots__ = ("_controller", "_line_id", "_manifest")

    def __init__(self, controller: ProjectController, line_id: str,
                 manifest: Path) -> None:
        self._controller = controller
        self._line_id = line_id
        self._manifest = manifest

    def execute(self) -> None:
        path = None
        try:
            with open(self._manifest, 'r', encoding='utf-8') as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict):
            source = str(payload.get('source_path') or '')
            path = source or None
        self._controller.line_source_path_ready.emit(self._line_id, path)


class _ArtifactDescendantsCommand:
    """worker 线程查询后代闭包 + 成果名称（级联确认框素材）。"""

    __slots__ = ("_controller", "_project_id", "_line_id", "_artifact_ids")

    def __init__(self, controller: ProjectController, project_id: str,
                 line_id: str, artifact_ids: list[str]) -> None:
        self._controller = controller
        self._project_id = project_id
        self._line_id = line_id
        self._artifact_ids = artifact_ids

    def execute(self) -> None:
        c = self._controller
        backend = c._backend()
        if backend is None:
            return
        descendants: list[str] = []
        seen: set[str] = set()
        try:
            for aid in self._artifact_ids:
                closure = backend.list_artifact_descendants(
                    self._project_id, self._line_id, aid)
                for item in closure:
                    if item not in seen:
                        seen.add(item)
                        descendants.append(item)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.exception("查询成果后代失败")
            c.log_message.emit(f"查询成果后代失败：{friendly_error_message(exc)}")
            return
        names: dict[str, str] = {}
        try:
            artifacts = backend.projects.list_artifacts(
                self._project_id, self._line_id)
            for item in artifacts:
                aid = str(getattr(item, "artifact_id", "") or "")
                if aid:
                    names[aid] = (str(getattr(item, "name", "") or "")
                                  or aid[:8])
        except Exception:  # noqa: BLE001 - 名称仅供展示，失败退回 aid 前 8 位
            pass
        c.artifact_descendants_ready.emit(self._line_id, descendants, names)


__all__ = ["ProjectController"]
