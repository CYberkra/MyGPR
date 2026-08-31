#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composition root and stable backend façade."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from mygpr.application.autotune.ports import AutoTuneDependencies
from mygpr.application.acquisition.service import AcquisitionService
from mygpr.application.autotune.service import AutoTuneService
from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.jobs.runner import InMemoryJobRunner
from mygpr.application.processing.service import ProcessingService
from mygpr.application.processing.workbench_service import ProcessingWorkbenchService
from mygpr.application.project.processing_service import ProjectProcessingService
from mygpr.application.project.service import ProjectService
from mygpr.application.project.maintenance_service import ProjectMaintenanceService
from mygpr.application.reporting.service import ReportingService
from mygpr.application.interpretation.service import InterpretationService
from mygpr.application.interpretation.edit_service import InterpretationEditService
from mygpr.application.spatial.service import SpatialService
from mygpr.domain.processing.models import PipelineDefinition, ProcessingRequest
from mygpr.domain.common.errors import MyGPRError
from mygpr.domain.acquisition.models import SensorSyncSettings
from mygpr.infrastructure.processing.autotune_adapter import DomainAutoTuneConstraintPolicy
from mygpr.infrastructure.acquisition.legacy_adapter import (
    LegacyAcquisitionReader,
    LegacySensorSidecarParser,
    LegacySensorSynchronizer,
)
from mygpr.infrastructure.persistence.field_project_adapter import LegacyFieldProjectRepository
from mygpr.infrastructure.processing.block_executor import FileBackedBlockPipelineExecutor
from mygpr.infrastructure.system.resource_policy import LocalProcessingResourcePolicy
from mygpr.infrastructure.processing.native_adapter import (
    NativeProcessingCatalog,
    NativeProcessingExecutor,
)

BACKEND_API_VERSION = "1.0"
_WORKBENCH_SERVICES: dict[int, ProcessingWorkbenchService] = {}
_INTERPRETATION_EDIT_SERVICES: dict[int, InterpretationEditService] = {}


class BackendShutdownError(MyGPRError):
    """Raised when active jobs cannot stop before project sessions close."""

    error_code = "MYGPR_BACKEND_SHUTDOWN_TIMEOUT"
    category = "backend"
    default_hint = "继续等待任务结束，或检查未响应的处理算法和外部存储。"


@dataclass(slots=True)
class MyGPRBackend:
    """Backend services exposed to any presentation technology.

    No service returned here exposes Qt objects, database connections, h5py
    datasets or concrete algorithm callables.
    """

    processing: ProcessingService
    acquisition: AcquisitionService
    autotune: AutoTuneService
    projects: ProjectService
    project_processing: ProjectProcessingService
    reporting: ReportingService
    interpretation: InterpretationService
    spatial: SpatialService
    jobs: InMemoryJobRunner
    api_version: str = BACKEND_API_VERSION

    def __post_init__(self) -> None:
        if not hasattr(self.projects, "_maintenance_service"):
            setattr(self.projects, "_maintenance_service", ProjectMaintenanceService(self.projects))

    @property
    def maintenance(self) -> ProjectMaintenanceService:
        service = getattr(self.projects, "_maintenance_service", None)
        if service is None:
            service = ProjectMaintenanceService(self.projects)
            setattr(self.projects, "_maintenance_service", service)
        return service

    @property
    def processing_workbench(self) -> ProcessingWorkbenchService:
        key = id(self)
        service = _WORKBENCH_SERVICES.get(key)
        if service is None:
            service = ProcessingWorkbenchService(self.projects, self.processing)
            _WORKBENCH_SERVICES[key] = service
        return service

    @property
    def interpretation_edit(self) -> InterpretationEditService:
        key = id(self)
        service = _INTERPRETATION_EDIT_SERVICES.get(key)
        if service is None:
            service = InterpretationEditService(self.projects, self.interpretation)
            _INTERPRETATION_EDIT_SERVICES[key] = service
        return service

    @classmethod
    def create_default(cls, *, max_workers: int = 2) -> "MyGPRBackend":
        catalog = NativeProcessingCatalog()
        executor = NativeProcessingExecutor()
        block_executor = FileBackedBlockPipelineExecutor()
        resource_policy = LocalProcessingResourcePolicy()
        dependencies = AutoTuneDependencies(
            catalog=catalog,
            executor=executor,
            constraints=DomainAutoTuneConstraintPolicy(),
        )
        processing = ProcessingService(catalog, executor, block_executor, resource_policy)
        projects = ProjectService(LegacyFieldProjectRepository())
        acquisition = AcquisitionService(
            LegacyAcquisitionReader(),
            LegacySensorSidecarParser(),
            LegacySensorSynchronizer(),
            projects,
        )
        return cls(
            processing=processing,
            acquisition=acquisition,
            autotune=AutoTuneService(dependencies),
            projects=projects,
            project_processing=ProjectProcessingService(projects, processing),
            reporting=ReportingService(projects),
            interpretation=InterpretationService(projects),
            spatial=SpatialService(projects),
            jobs=InMemoryJobRunner(max_workers=max_workers),
        )

    def _submit_project_job(
        self,
        project_id: str,
        title: str,
        operation: Callable[[ExecutionContext], Any],
    ) -> str:
        """Acquire a project lease before queueing and release it at terminal state."""
        lease = self.projects.acquire_lease(project_id)
        try:
            return self.jobs.submit(
                title,
                operation,
                resource_keys=(f"project:{project_id}",),
                finalizer=lease.release,
            )
        except (RuntimeError, TypeError):
            lease.release()
            raise

    def submit_processing(self, request: ProcessingRequest, *, title: str | None = None) -> str:
        """Submit one processing method and return a stable job identifier."""
        method = self.processing.get_method(request.method_id)
        return self.jobs.submit(
            title or f"处理: {method.name}",
            lambda context: self.processing.execute_method(request, context),
        )

    def submit_pipeline(
        self,
        data: np.ndarray,
        pipeline: PipelineDefinition,
        *,
        header_info: dict[str, Any] | None = None,
        trace_metadata: dict[str, np.ndarray] | None = None,
        title: str | None = None,
    ) -> str:
        """Submit a complete processing pipeline."""
        return self.jobs.submit(
            title or pipeline.name,
            lambda context: self.processing.execute_pipeline(
                data,
                pipeline,
                header_info=header_info,
                trace_metadata=trace_metadata,
                context=context,
            ),
        )

    def submit_autotune(
        self,
        data: np.ndarray,
        method_key: str,
        *,
        title: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit AutoTune without requiring a GUI event loop."""

        def operation(context: ExecutionContext) -> dict[str, Any]:
            return self.autotune.tune_method(
                data,
                method_key,
                execution_context=context,
                **kwargs,
            )

        return self.jobs.submit(title or f"自动选参: {method_key}", operation)


    def submit_line_import(
        self,
        project_id: str,
        source: str,
        *,
        line_id: str,
        name: str = "",
        copy_into_project: bool = True,
        dielectric_constant: float = 9.0,
        title: str | None = None,
    ) -> str:
        """Import and normalize one measurement line through the shared job system."""
        return self._submit_project_job(
            project_id,
            title or f"导入测线: {line_id}",
            lambda context: self.acquisition.import_line(
                project_id,
                source,
                line_id=line_id,
                name=name,
                copy_into_project=copy_into_project,
                dielectric_constant=float(dielectric_constant),
                context=context,
            ),
        )


    def submit_batch_line_import(self, project_id: str, sources: list[str] | tuple[str, ...], *, title: str | None = None) -> str:
        selected = tuple(str(item) for item in sources)
        if not selected:
            raise ValueError("至少选择一个测线文件")
        return self._submit_project_job(project_id, title or f"批量导入测线: {len(selected)} 个文件", lambda context: self.maintenance.batch_import_lines(project_id, selected, context=context))

    def submit_project_quality_check(self, project_id: str, *, title: str | None = None) -> str:
        return self._submit_project_job(project_id, title or "项目数据质检", lambda context: self.maintenance.run_project_quality_check(project_id, context=context))


    def submit_line_quality_check(self, project_id: str, line_id: str, *, title: str | None = None) -> str:
        return self._submit_project_job(project_id, title or f"测线数据质检: {line_id}", lambda context: self.maintenance.run_line_quality_check(project_id, line_id))

    def submit_line_source_relink(self, project_id: str, line_id: str, new_source: str, *, allow_mismatch: bool = False, title: str | None = None) -> str:
        return self._submit_project_job(project_id, title or f"重新定位源文件: {line_id}", lambda context: self.maintenance.relink_line_source(project_id, line_id, new_source, allow_mismatch=allow_mismatch, context=context))

    def submit_source_file_check(self, project_id: str, *, title: str | None = None) -> str:
        return self._submit_project_job(project_id, title or "源文件完整性检查", lambda context: self.maintenance.check_source_files(project_id, context=context))

    def submit_line_transpose(self, project_id: str, line_id: str, *, title: str | None = None) -> str:
        return self._submit_project_job(project_id, title or f"B-scan 方向修正: {line_id}", lambda context: self.maintenance.transpose_line_dataset(project_id, line_id, context=context))

    def submit_sensor_sync(
        self,
        project_id: str,
        line_id: str,
        *,
        rtk_path: str,
        trace_timestamps_path: str | None = None,
        imu_path: str | None = None,
        altimeter_path: str | None = None,
        settings: SensorSyncSettings | None = None,
        title: str | None = None,
    ) -> str:
        """Synchronize RTK/IMU/altimeter streams and persist trace metadata."""
        return self._submit_project_job(
            project_id,
            title or f"传感器同步: {line_id}",
            lambda context: self.acquisition.synchronize_project_line(
                project_id,
                line_id,
                rtk_path=rtk_path,
                trace_timestamps_path=trace_timestamps_path,
                imu_path=imu_path,
                altimeter_path=altimeter_path,
                settings=settings,
                context=context,
            ),
        )

    def submit_project_pipeline(
        self,
        project_id: str,
        line_id: str,
        pipeline: PipelineDefinition,
        *,
        result_name: str = "",
        branch_id: str = "",
        input_artifact_id: str = "",
        title: str | None = None,
    ) -> str:
        """Process a persisted project line and commit the result."""
        return self._submit_project_job(
            project_id,
            title or f"项目处理: {line_id}",
            lambda context: self.project_processing.execute_pipeline(
                project_id,
                line_id,
                pipeline,
                result_name=result_name,
                branch_id=branch_id,
                input_artifact_id=input_artifact_id,
                context=context,
            ),
        )

    def submit_project_report(
        self,
        project_id: str,
        *,
        package_name: str | None = None,
        report_profile: dict[str, Any] | None = None,
        title: str | None = None,
    ) -> str:
        """Generate a report package through the shared job system."""
        return self._submit_project_job(
            project_id,
            title or "生成工程报告",
            lambda context: self.reporting.generate_package(
                project_id,
                package_name=package_name,
                report_profile=report_profile,
                context=context,
            ),
        )

    def submit_project_backup(
        self,
        project_id: str,
        destination_dir: str | None = None,
        *,
        require_external_device: bool = False,
        title: str | None = None,
    ) -> str:
        """Create and verify a portable project backup."""
        return self._submit_project_job(
            project_id,
            title or "项目备份",
            lambda context: self.projects.backup_project(
                project_id,
                destination_dir,
                require_external_device=require_external_device,
                context=context,
            ),
        )

    def submit_spatial_result(
        self,
        project_id: str,
        *,
        name: str,
        line_ids: tuple[str, ...] | list[str] | None = None,
        velocity_m_per_ns: float | None = None,
        generate_surface: bool = True,
        title: str | None = None,
    ) -> str:
        """Generate a versioned spatial result outside the GUI thread."""
        selected = None if line_ids is None else tuple(str(item) for item in line_ids)

        def operation(context: ExecutionContext) -> Any:
            context.report_progress(0, 2, "空间成果预检")
            preflight = self.spatial.preflight(
                project_id, line_ids=selected, generate_surface=generate_surface
            )
            if not bool(preflight.get("passed")):
                errors = "; ".join(str(item) for item in preflight.get("errors", ()))
                raise ValueError(errors or "spatial preflight failed")
            context.raise_if_cancelled()
            context.report_progress(1, 2, "生成空间成果")
            result = self.spatial.create_result(
                project_id,
                name=name,
                line_ids=selected,
                velocity_m_per_ns=velocity_m_per_ns,
                generate_surface=generate_surface,
            )
            context.report_progress(2, 2, "空间成果已提交")
            return result

        return self._submit_project_job(
            project_id, title or f"生成空间成果: {name}", operation
        )

    def build_georeference_3d(
        self,
        project_id: str,
        line_id: str,
        *,
        preview_lod: str = "auto",
        max_preview_traces: int = 240,
        max_preview_samples: int = 160,
        title: str | None = None,
    ) -> str:
        """Build a 3D georeference payload for one line as a shared-system job."""
        return self._submit_project_job(
            project_id,
            title or f"三维地理配准: {line_id}",
            lambda context: self.spatial.build_georeference_3d(
                project_id,
                line_id,
                preview_lod=preview_lod,
                max_preview_traces=max_preview_traces,
                max_preview_samples=max_preview_samples,
            ),
        )

    def submit_project_restore(
        self,
        archive_path: str,
        destination_root: str,
        *,
        project_dir_name: str | None = None,
        open_after_restore: bool = True,
        title: str | None = None,
    ) -> str:
        """Restore and verify a project outside the GUI thread."""

        def operation(context: ExecutionContext) -> Any:
            context.report_progress(0, 2, "校验备份包")
            context.check_cancelled()
            result = self.projects.restore_project(
                archive_path,
                destination_root,
                project_dir_name=project_dir_name,
                open_after_restore=open_after_restore,
            )
            context.report_progress(2, 2, "项目恢复完成")
            return result

        return self.jobs.submit(title or "恢复项目备份", operation)

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_futures: bool = True,
        timeout_s: float = 30.0,
    ) -> None:
        """Stop new work, cancel/drain active jobs, then close project sessions."""
        self.jobs.stop_accepting()
        self.jobs.cancel_all()
        pending = self.jobs.shutdown(
            wait=wait,
            cancel_futures=cancel_futures,
            cancel_running=True,
            timeout=max(0.0, float(timeout_s)) if wait else None,
        )
        if pending:
            raise BackendShutdownError(
                "backend shutdown timed out with active jobs: " + ", ".join(pending)
            )
        self.projects.close_all(force=False)
        _WORKBENCH_SERVICES.pop(id(self), None)
        _INTERPRETATION_EDIT_SERVICES.pop(id(self), None)


__all__ = ["BACKEND_API_VERSION", "BackendShutdownError", "MyGPRBackend"]
