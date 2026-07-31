from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mygpr.application.jobs.models import JobStatus
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import BACKEND_API_VERSION, MyGPRBackend
from tests.qt_import_isolation import assert_qt_imports_unchanged, qt_module_snapshot

pytestmark = [
    pytest.mark.industrial,
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requirement("REQ-API-001", "REQ-PROJ-001", "REQ-REL-001"),
    pytest.mark.risk("RISK-DATA-LOSS", "RISK-SHUTDOWN-RACE"),
    pytest.mark.level("system"),
]

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "yingshan_real_v1"


def _line9_subset() -> tuple[np.ndarray, dict[str, object]]:
    manifest = json.loads((FIXTURE / "dataset_manifest.json").read_text(encoding="utf-8"))
    with np.load(FIXTURE / "yingshan_trace_subset_v1.npz", allow_pickle=False) as archive:
        matrix = np.asarray(archive["line_9_matrix"], dtype=np.float32)
    return matrix, dict(manifest["lines"]["9"])


def test_headless_real_data_project_workflow_is_reopenable_and_auditable(tmp_path: Path) -> None:
    qt_before = qt_module_snapshot()
    matrix, source = _line9_subset()
    backend = MyGPRBackend.create_default(max_workers=1)
    project_root = tmp_path / "yingshan-acceptance"
    try:
        assert backend.api_version == BACKEND_API_VERSION == "1.0"
        project = backend.projects.create_project(project_root, name="Yingshan acceptance")
        backend.projects.save_line_dataset(
            project.project_id,
            "L09",
            matrix,
            name="Yingshan line 9 deterministic subset",
            length_m=float(source["trace_interval_m"]) * (matrix.shape[1] - 1),
            time_window_ns=float(source["time_window_ns"]),
            metadata={"scientific_dataset_id": "yingshan_airborne_gpr_2025_v1", "source_sha256": source["sha256"]},
        )
        pipeline = PipelineDefinition(
            name="Acceptance preprocessing",
            steps=(
                PipelineStep("dewow", {"window": 23}),
                PipelineStep("subtracting_average_2D", {"ntraces": 16}),
                PipelineStep("sec_gain", {"gain_min": 1.0, "gain_max": 6.0, "power": 1.0}),
            ),
        )
        job_id = backend.submit_project_pipeline(project.project_id, "L09", pipeline, result_name="acceptance-v1")
        snapshot = backend.jobs.wait(job_id, timeout=60)
        assert snapshot.status is JobStatus.COMPLETED, snapshot.error_message
        artifact = snapshot.result
        assert artifact.status in {"success", "committed"}
        assert artifact.sha256
        assert artifact.shape == matrix.shape
        assert artifact.data_reference.startswith("h5://")
        assert any(event.event_type.value == "artifact_created" for event in backend.jobs.events(job_id))

        deep = backend.projects.audit_project(project.project_id, deep_hash=True, clean_staging=True, staging_min_age_s=0)
        assert deep.healthy, [issue.message for issue in deep.issues]
        package = backend.reporting.generate_package(project.project_id, package_name="acceptance-report")
        assert (project_root / package.manifest_path).is_file()
        backup = backend.projects.backup_project(project.project_id, tmp_path / "backups")
        assert backup.verified

        backend.projects.close_project(project.project_id)
        reopened = backend.projects.open_project(project_root)
        assert reopened.project_id == project.project_id
        assert backend.projects.get_dataset_info(reopened.project_id, "L09").shape == matrix.shape
        artifacts = backend.projects.list_artifacts(reopened.project_id, "L09")
        assert len(artifacts) == 1 and artifacts[0].sha256 == artifact.sha256
        assert_qt_imports_unchanged(qt_before)
    finally:
        backend.shutdown()
