from __future__ import annotations

import json
from pathlib import Path

from core.interpretation_service import InterpretationService
from core.project_service import ProjectService


def test_interpretation_service_creates_updates_and_deletes_geojson_features(
    tmp_path: Path,
) -> None:
    project = ProjectService.create(tmp_path / "project", name="Interpretation")
    try:
        service = InterpretationService(project)
        point = service.add_point(
            "L001",
            trace=12,
            sample=34,
            confidence=0.8,
            result_id="R001",
            label="疑似界面点",
        )
        line = service.add_interface_line(
            "L001",
            points=[(0, 10), (10, 12), (20, 14)],
            confidence=0.9,
            result_id="R001",
            label="基覆界面",
        )
        interval = service.add_interval(
            "L001",
            trace_start=30,
            trace_end=45,
            sample_start=20,
            sample_end=60,
            confidence=0.6,
            label="异常区间",
        )

        features = service.list_features("L001")
        assert [feature.feature_type for feature in features] == [
            "point",
            "interface_line",
            "interval",
        ]
        assert point.geometry["type"] == "Point"
        assert line.geometry["type"] == "LineString"
        assert interval.geometry["type"] == "Polygon"

        updated = service.update_feature(
            "L001",
            point.feature_id,
            confidence=0.95,
            properties={"label": "确认界面点", "note": "复核"},
        )
        assert updated.confidence == 0.95
        assert updated.properties["label"] == "确认界面点"

        assert service.delete_feature("L001", interval.feature_id) is True
        assert len(service.list_features("L001")) == 2
        payload = json.loads((project.root / "interpretations" / "L001.geojson").read_text(encoding="utf-8"))
        assert payload["type"] == "FeatureCollection"
        assert payload["schema"] == "mygpr.interpretations.v1"
    finally:
        project.close()


def test_interpretation_service_rejects_invalid_confidence_and_geometry(tmp_path: Path) -> None:
    project = ProjectService.create(tmp_path / "project", name="Interpretation")
    try:
        service = InterpretationService(project)
        try:
            service.add_point("L001", trace=1, sample=2, confidence=1.2)
        except ValueError:
            pass
        else:
            raise AssertionError("confidence above 1 was accepted")

        try:
            service.add_interface_line("L001", points=[(1, 2)], confidence=0.5)
        except ValueError:
            pass
        else:
            raise AssertionError("one-point interface line was accepted")
    finally:
        project.close()
