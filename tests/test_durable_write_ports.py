"""DurableWritePort 收敛路径补测（diff-coverage 缺口，2026-10 审计批次引入）。

覆盖审计批次把四个手写落盘变体收敛到 ``DurableWritePort`` 后、既有测试
没打到的默认 writer 分支与协议桩。backend job 无 PyQt6，本文件保持零 Qt
依赖（不得 importorskip PyQt6——那样会被 backend job 跳过，缺口依旧）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from core.report_export_rows import _timestamp
from mygpr.application.grid.service import grid_attribute, write_grid_geojson
from mygpr.application.persistence_ports import DurableWritePort
from mygpr.domain.grid.models import AttributeGridRequest
from mygpr.infrastructure.persistence.durable_write import default_durable_write


def test_report_package_timestamp_is_compact_utc() -> None:
    """报告包目录名时间戳：14 位数字（%Y%m%d_%H%M%S），仅用于路径段。"""
    assert re.fullmatch(r"\d{8}_\d{6}", _timestamp())


def test_write_grid_geojson_default_writer(tmp_path: Path) -> None:
    """writer=None 兜底分支：契约注入 default_durable_write 落盘。"""
    grid = grid_attribute(AttributeGridRequest(
        x_m=(0.0, 0.5, 1.2, 1.2),
        y_m=(0.1, 0.2, 2.2, 2.25),
        values=(1.0, 3.0, 7.0, 9.0),
        attribute_name="cov",
        cell_size_m=1.0,
    ))
    destination = tmp_path / "grid.geojson"
    result = write_grid_geojson(grid, destination, crs_name="EPSG:4326")
    assert result == destination
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["crs"]["properties"]["name"] == "EPSG:4326"
    assert payload["type"] == "FeatureCollection"


def test_analysis_evidence_manifest_durable_write(tmp_path: Path) -> None:
    """成果包 manifest 经端口原子发布（include_data=False 纯 manifest 路径）。"""
    from mygpr.application.processing.analysis_service import (
        ProcessingAnalysisService,
    )

    manifest = {"schema": "probe", "files": []}
    manifest_path, sha, files = (
        ProcessingAnalysisService._write_evidence_files(
            None, tmp_path, manifest, include_data=False)
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert len(sha) == 64
    assert files and files[0].endswith("manifest.json")
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["manifest_sha256"] == sha


def test_core_durable_write_text_and_json(tmp_path: Path) -> None:
    """CoreDurableWrite 两个便捷委托：write_text / write_json。"""
    writer = default_durable_write()
    text_path = writer.write_text(tmp_path / "a.txt", "hello")
    assert text_path.read_text(encoding="utf-8") == "hello"
    json_path = writer.write_json(tmp_path / "a.json", {"k": 1})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"k": 1}


def test_protocol_default_bodies_are_stubs(tmp_path: Path) -> None:
    """协议桩体可被显式子类直接执行（结构契约的自我保护）。"""

    class _Bare(DurableWritePort):
        pass

    port = _Bare()
    assert port.write_bytes(tmp_path / "x.bin", b"") is None
    assert port.write_text(tmp_path / "x.txt", "") is None
    assert port.write_json(tmp_path / "x.json", {}) is None


def test_workbench_durable_writer_lazy_resolution() -> None:
    """_durable_writer 函数级延迟解析：import 期不触碰 infrastructure。"""
    from mygpr.application.processing.workbench_service import _durable_writer

    assert _durable_writer() is not None
