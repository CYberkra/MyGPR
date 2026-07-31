#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np

from core.field_import_preview import build_import_preflight
from core.field_project_operations import RecentProjectsStore, create_project, import_line_data, open_project, preview_import_source


def test_import_preflight_accepts_npy_matrix(tmp_path: Path) -> None:
    source = tmp_path / "line.npy"
    np.save(source, np.arange(64 * 32, dtype=np.float32).reshape(64, 32))

    preview = build_import_preflight(source, line_id="L01")

    assert preview.can_import is True
    assert preview.sample_count == 64
    assert preview.trace_count == 32
    assert preview.shape_text == "64 × 32"
    assert "可直接导入" in preview.message


def test_import_preflight_explains_recognized_vendor_format(tmp_path: Path) -> None:
    source = tmp_path / "profile.dzt"
    source.write_bytes(b"not a real dzt")

    preview = preview_import_source(source, line_id="L01")

    assert preview.can_import is False
    assert "GSSI DZT" in preview.format_name
    assert "尚未直接解码" in preview.message
    assert preview.suggestions


def test_create_open_recent_and_import_line_with_metadata(tmp_path: Path) -> None:
    recent_path = tmp_path / "recent.json"
    recent = RecentProjectsStore(recent_path)
    project = create_project(
        tmp_path,
        name="雨水管线普查",
        location="测试场地A",
        operator="测试员",
        recent_store=recent,
    )
    assert project.manifest.name == "雨水管线普查"
    assert project.manifest.location == "测试场地A"
    assert recent.load()[0].name == "雨水管线普查"

    source = tmp_path / "matrix.csv"
    matrix = np.random.default_rng(4).normal(size=(80, 40))
    np.savetxt(source, matrix, delimiter=",")
    line = import_line_data(project, source, line_id="L09", name="西侧道路测线")
    assert line.line_id == "L09"
    assert line.name == "西侧道路测线"
    assert line.gpr_dataset_path

    reopened = open_project(project.root, recent_store=recent)
    assert reopened.manifest.name == "雨水管线普查"
    assert reopened.get_line("L09").name == "西侧道路测线"
