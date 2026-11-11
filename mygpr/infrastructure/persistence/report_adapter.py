"""Report snapshot orchestration on the locked field-project session."""
from typing import Any, Mapping
from core.field_report_export import generate_project_report_package
from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.reporting.models import ReportPackage


class ReportPersistenceMixin:
    _lock: Any
    _store: Any
    read_markers: Any
    list_lines: Any

    def generate_report(
        self,
        *,
        package_name: str | None,
        report_profile: Mapping[str, Any],
        context: ExecutionContext,
    ) -> ReportPackage:
        with self._lock:
            result = generate_project_report_package(
                self._store,
                marker_documents=[self.read_markers(line.line_id).to_dict() for line in self.list_lines() if line.trace_count > 0],
                package_name=package_name,
                report_profile=dict(report_profile),
                cancel_checker=context.is_cancelled,
                progress_callback=context.report_progress,
            )
        metadata = result.to_dict()
        metadata["project_root"] = str(self._store.root.resolve())
        return ReportPackage(
            package_dir=result.package_dir,
            manifest_path=result.manifest_path,
            generated_at=result.generated_at,
            file_count=result.file_count,
            pdf_path=result.pdf_path,
            html_path=result.html_path,
            xlsx_path=result.xlsx_path,
            delivery_zip_path=result.delivery_zip_path,
            delivery_zip_sha256_path=result.delivery_zip_sha256_path,
            seal_path=result.seal_path,
            metadata=metadata,
        )

