from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.delivery_service import DeliveryService
from core.ingest_service import IngestService
from core.interpretation_service import InterpretationService
from core.processing_session import ProcessingSessionService
from core.qc_service import QcService


def _formal_project(tmp_path: Path):
    source = tmp_path / "line.csv"
    np.savetxt(source, np.arange(120, dtype=np.float32).reshape(20, 6), delimiter=",")
    temporary = IngestService.open_temporary(source)
    project = IngestService.formalize(temporary, tmp_path / "formal", name="Delivery")
    temporary.close()
    line_id = project.list_lines()[0].line_id
    qc = QcService(project)
    report = qc.run_line_qc(line_id)
    for item in report.items:
        if item.severity == "warning" and not item.acknowledged:
            qc.acknowledge_warning(line_id, item.code, "测试夹具确认该警告不阻断处理")
    return project


def test_delivery_service_checks_and_builds_auditable_evidence_package(tmp_path: Path) -> None:
    project = _formal_project(tmp_path)
    try:
        line_id = project.list_lines()[0].line_id
        session = ProcessingSessionService.open_line(project, line_id)
        session.apply_method("amplitude_scale", {"scale": 2.0})
        result = session.save_version("Scaled")
        InterpretationService(project).add_point(
            line_id,
            trace=2,
            sample=4,
            confidence=0.8,
            result_id=result.result_id,
            label="解释点",
        )

        service = DeliveryService(project)
        checks = service.run_checks()
        assert checks["summary"]["error_count"] == 0
        assert checks["summary"]["line_count"] == 1
        assert checks["summary"]["result_count"] == 1
        assert checks["summary"]["interpretation_count"] == 1

        package = service.build_package("field_delivery")
        assert package.exists()
        assert (package / "report.md").exists()
        assert (package / "manifest.json").exists()
        assert (package / "checksums.sha256").exists()
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema"] == "mygpr.delivery.v1"
        assert manifest["checks"]["summary"]["error_count"] == 0
        assert any(item["role"] == "processing_result" for item in manifest["evidence"])
        assert any(item["role"] == "interpretation" for item in manifest["evidence"])
    finally:
        project.close()


def test_delivery_service_blocks_package_when_integrity_error_exists(tmp_path: Path) -> None:
    project = _formal_project(tmp_path)
    try:
        line = project.list_lines()[0]
        raw = project.resolve_relative_path(line.raw_files[0].path)
        raw.chmod(0o666)
        raw.write_text("tampered", encoding="utf-8")
        checks = DeliveryService(project).run_checks()
        assert checks["summary"]["error_count"] >= 1
        try:
            DeliveryService(project).build_package("blocked")
        except RuntimeError:
            pass
        else:
            raise AssertionError("delivery package was built despite integrity error")
    finally:
        project.close()
