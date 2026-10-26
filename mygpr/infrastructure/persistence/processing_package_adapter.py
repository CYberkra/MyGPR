"""Result export on one locked project session, without materializing its matrices."""
from mygpr.domain.project.result_package import select_saved_steps
from mygpr.infrastructure.reporting.processing_package_export import export_processing_package


class ProcessingPackageMixin:
    def export_processing_package(self, line_id, destination, *, artifact_id='', context):
        with self._lock:
            safe_line = self._validated_line(line_id)
            artifacts, warnings = select_saved_steps(self.list_artifacts(safe_line), artifact_id)
            result = export_processing_package(
                project_id=self.summary.project_id, line_id=safe_line, artifacts=artifacts,
                load_dataset=lambda aid: self._artifact_dataset(safe_line, aid)[2],
                load_raw_dataset=lambda: self._store.load_gpr_dataset(safe_line),
                destination=destination, target=artifact_id, warnings=warnings, context=context)
            for warning in warnings:
                context.emit_warning({'code': 'PROCESSING_PACKAGE_INCOMPLETE_LINEAGE', 'message': warning})
            context.report_progress(1, 1, f'已保存：{result.archive_path}')
            return result
