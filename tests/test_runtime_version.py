"""Source checkouts and installed wheels must report their actual version."""
import importlib
import importlib.metadata

import pytest


@pytest.fixture(params=[
    ("core.field_report_export", "_software_version", "core/field_report_export.py"),
    ("ui.pages.settings_page", "_read_version", "ui/pages/settings_page.py"),
])
def version_reader(request, tmp_path, monkeypatch):
    name, function, relative = request.param
    if name.startswith("ui."):
        pytest.importorskip("PyQt6")
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "__file__", str(tmp_path / relative))
    return getattr(module, function), tmp_path / "VERSION"


def test_runtime_version_reads_source_version_with_bom(version_reader):
    read, path = version_reader
    path.write_text("8.7.6\n", encoding="utf-8-sig")
    assert read() == "8.7.6"


@pytest.mark.parametrize("content", [None, "", "  \n"])
def test_runtime_version_uses_installed_metadata_without_source_version(version_reader, monkeypatch, content):
    read, path = version_reader
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "8.7.6" if name == "mygpr" else None)
    assert read() == "8.7.6"


def test_runtime_version_does_not_invent_a_version(version_reader, monkeypatch):
    read, _path = version_reader

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert read() == "unknown"
