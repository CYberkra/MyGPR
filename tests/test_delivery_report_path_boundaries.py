"""Report output paths and package-name boundaries; no scientific-data claims."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.field_report_export import _prepare_package_directories, generate_project_report_package


@pytest.mark.parametrize("name", ["../escaped", "..\\escaped", "/escaped", "C:\\escaped", "C:escaped", "nested/name", "nested\\name", ".", ".."])
def test_report_rejects_path_names_before_any_project_or_output_writes(tmp_path, name):
    store = SimpleNamespace(root=tmp_path / "project")
    store.root.mkdir()
    with patch("core.field_report_export.ensure_full_source_hashes") as hash_sources:
        with pytest.raises(ValueError):
            generate_project_report_package(store, package_name=name)
    hash_sources.assert_not_called()
    assert list(store.root.iterdir()) == []


@pytest.mark.parametrize("name", ["../escaped", "..\\escaped", "/escaped", "C:\\escaped", "C:escaped", "nested/name", "nested\\name", ".", ".."])
def test_report_directory_preparation_rejects_path_names(tmp_path, name):
    store = SimpleNamespace(root=tmp_path / "project")
    store.root.mkdir()
    with patch("core.field_report_export.Path.mkdir", side_effect=AssertionError("invalid name reached mkdir")) as mkdir:
        with pytest.raises(ValueError):
            _prepare_package_directories(store, package_name=name, spatial_result_id="")
    mkdir.assert_not_called()
    assert list(store.root.iterdir()) == []


def test_report_directory_names_allow_chinese_spaces_and_keep_duplicate_behavior(tmp_path):
    store = SimpleNamespace(root=tmp_path / "project")
    store.root.mkdir()
    first = _prepare_package_directories(store, package_name="测区 报告", spatial_result_id="")
    assert first["final"] == store.root / "reports" / "测区 报告"
    first["final"].mkdir()
    (first["final"] / "existing.txt").write_text("keep", encoding="utf-8")
    second = _prepare_package_directories(store, package_name="测区 报告", spatial_result_id="")
    assert second["final"].parent == first["final"].parent
    assert second["final"].name.startswith("测区 报告_")
    assert (first["final"] / "existing.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("name", [None, ""])
def test_empty_report_package_name_still_uses_default(tmp_path, name):
    store = SimpleNamespace(root=tmp_path / "project")
    store.root.mkdir()
    paths = _prepare_package_directories(store, package_name=name, spatial_result_id="")
    assert paths["final"].parent == store.root / "reports"
    assert paths["final"].name.startswith("report_")
