"""Rename actions preserve selection; worker results return to the Qt thread."""

from types import SimpleNamespace as NS
from threading import Thread, get_ident

import pytest

pytest.importorskip("PyQt6")

from ui.pages.project_page import ProjectPage
from ui.controllers.project_controller import ProjectController
from ui.widgets.context_menus import make_menu


def chain_for(page, pc, processing):
    from importlib import util, import_module

    assert util.find_spec("ui.coordinator_identity"), "Project identity wiring missing"
    settings = {}

    class Settings:
        def get(self, key, default=None):
            return settings.get(key, default)

        def set(self, key, value):
            settings[key] = value

        def save(self):
            pass

        def add_recent_project(self, path):
            settings["recent_projects"] = [path]

    pages = {"projectInterface": page, "homeInterface": page.startup, "processingInterface": processing}
    co = NS(
        project_controller=pc,
        settings=Settings(),
        page=pages.get,
        current_project_id=lambda: pc.current_project_id,
        infobar=lambda *a: None,
        processing=NS(processing_job_id=""),
        jobs=NS(import_job_ids=set(), spatial_job_ids=set()),
    )
    events = []
    chain = NS(
        _co=co,
        current_line_id="L01",
        _active_job_ids=set(),
        _busy_sources=set(),
        can_change_project=lambda: True,
        on_project_closed=lambda: events.append("closed"),
        on_project_opened=lambda s: events.append(s.root_path),
    )
    identity = import_module("ui.coordinator_identity").ProjectIdentityChain(chain, page, None)
    return identity, chain, settings, events


@pytest.fixture
def page(qapp, tmp_path):
    value = ProjectPage()
    value.set_project_info(NS(project_id="P", name="校园", root_path=str(tmp_path), read_only=False, line_count=1))
    value.set_lines([NS(line_id="L01", name="原名称", trace_count=20, sample_count=18)])
    value.select_line("L01")
    yield value
    value.close()
    value.deleteLater()
    qapp.processEvents()


def helper(page):
    assert hasattr(page, "_identity"), "Project identity controls missing"
    return page._identity


def test_rename_request_has_frozen_project_and_line_and_waits_for_commit(page, monkeypatch):
    identity = helper(page)
    import ui.pages.project_identity_ui as module

    monkeypatch.setattr(module, "ask_display_name", lambda *a, **kw: "新名称")
    got = []
    identity.rename_line_requested.connect(lambda *args: got.append(args))
    identity.rename_line("L01")
    assert got == [("P", "L01", "新名称")]
    assert page._lines[0].name == "原名称" and page.current_line_id() == "L01"


def test_stale_rename_dialog_and_read_only_actions_are_rejected(page, monkeypatch):
    identity = helper(page)
    import ui.pages.project_identity_ui as module

    def answer(*args, **kw):
        page._summary = NS(project_id="Q", read_only=False)
        return "旧项目名称"

    monkeypatch.setattr(module, "ask_display_name", answer)
    got = []
    identity.rename_line_requested.connect(lambda *a: got.append(a))
    identity.rename_line("L01")
    assert not got
    page._summary = NS(project_id="P", read_only=True)
    menu = make_menu(page)
    identity.add_project_actions(menu)
    assert not next(a for a in menu.actions() if a.text() == "重命名项目…").isEnabled()
    assert not next(a for a in menu.actions() if a.text() == "迁移项目目录…").isEnabled()


def test_identity_controller_returns_on_gui_thread_and_updates_summary(qapp, monkeypatch):
    pc = ProjectController()
    assert hasattr(pc, "identity"), "Project identity controller missing"
    import ui.controllers.project_identity_controller as module

    pc._current = NS(project_id="P", name="旧项目", root_path="D:/old")
    changed = NS(project_id="P", name="新项目", root_path="D:/old")
    pc.set_backend(NS(backend=NS(projects=NS(rename_project=lambda *a: changed))))
    queued = []
    monkeypatch.setattr(module, "run_command", lambda command, **kw: queued.append(command))
    got = []
    pc.identity.finished.connect(lambda *args: got.append((get_ident(), args)))
    assert pc.identity.rename_project("P", "新项目")
    assert pc._busy
    worker = Thread(target=queued[0].execute)
    worker.start()
    worker.join()
    assert not got and pc._current.name == "旧项目"
    qapp.processEvents()
    assert got[0][0] == get_ident() and pc._current.name == "新项目" and not pc._busy


def test_controller_failure_preserves_project_and_releases_busy(qapp, monkeypatch):
    pc = ProjectController()
    assert hasattr(pc, "identity"), "Project identity controller missing"
    import ui.controllers.project_identity_controller as module

    pc._current = NS(project_id="P", name="旧项目", root_path="D:/old")

    def fail(*a):
        raise OSError("fixture failed")

    pc.set_backend(NS(backend=NS(projects=NS(rename_project=fail))))
    queued = []
    monkeypatch.setattr(module, "run_command", lambda command, **kw: queued.append(command))
    got = []
    pc.identity.finished.connect(lambda *a: got.append(a))
    assert pc.identity.rename_project("P", "新项目")
    queued[0].execute()
    qapp.processEvents()
    assert pc._current.name == "旧项目" and not pc._busy
    assert got and got[0][1] is False and "fixture failed" in got[0][2]


def test_committed_line_name_updates_dropdown_without_touching_draft(page, qapp, monkeypatch):
    from ui.pages.processing_page import ProcessingPage
    import numpy as np

    pc = ProjectController()
    pc._current = page._summary
    updated = NS(line_id="L01", name="新名称", trace_count=20, sample_count=18)
    pc.set_backend(NS(backend=NS(projects=NS(rename_line=lambda *a: updated))))
    processing = ProcessingPage()
    processing.set_project_context("P")
    processing.set_lines(page._lines)
    processing.set_line_label("L01")
    processing.set_original_bundle(NS(matrix=np.ones((18, 20)), vmin=-1, vmax=1))
    processing._pipeline_list.add_step("dewow", "Dewow", {"window": 41})
    before = processing.current_pipeline()
    matrix = processing._workspace.primary_view()._matrix
    identity, chain, settings, events = chain_for(page, pc, processing)
    import ui.controllers.project_identity_controller as module

    commands = []
    monkeypatch.setattr(module, "run_command", lambda command, **kw: commands.append(command))
    page._identity.rename_line_requested.emit("P", "L01", "新名称")
    commands[0].execute()
    qapp.processEvents()
    assert page._lines_table.item(0, 1).text() == "新名称"
    assert "新名称" in processing._line_combo.currentText()
    assert processing.current_pipeline() == before and processing._workspace.primary_view()._matrix is matrix
    assert chain.current_line_id == "L01" and not events
    processing.close()
    processing.deleteLater()


def test_project_rename_updates_recent_name_without_resetting_project(page, qapp, monkeypatch):
    pc = ProjectController()
    pc._current = page._summary
    changed = NS(**{**vars(page._summary), "name": "校园复测"})
    pc.set_backend(NS(backend=NS(projects=NS(rename_project=lambda *a: changed))))
    identity, chain, settings, events = chain_for(page, pc, None)
    import ui.controllers.project_identity_controller as module

    commands = []
    monkeypatch.setattr(module, "run_command", lambda command, **kw: commands.append(command))
    page._identity.rename_project_requested.emit("P", "校园复测")
    commands[0].execute()
    qapp.processEvents()
    assert page._summary.name == "校园复测" and chain.current_line_id == "L01" and not events
    assert settings["project_display_names"][changed.root_path] == "校园复测"


def test_relocation_persists_recent_and_pinned_paths_with_new_display_name(page, tmp_path):
    from ui.settings_manager import SettingsManager

    pc = ProjectController()
    pc._current = page._summary
    identity, chain, _, _ = chain_for(page, pc, None)
    path = str(tmp_path / "settings.json")
    settings = SettingsManager(path)
    identity.co.settings = settings
    old = page._summary.root_path
    target = str(tmp_path / "new-project")
    other = str(tmp_path / "another-project")
    settings.set("recent_projects", [old, other])
    settings.set("pinned_recent_projects", [old, other])
    settings.set("project_display_names", {old: "原名称", other: "另一个项目"})
    identity._remember(NS(root_path=target, name="新项目名称"), old)
    reloaded = SettingsManager(path)
    assert reloaded.get("recent_projects") == [target, other]
    assert reloaded.get("pinned_recent_projects") == [target, other]
    assert reloaded.get("project_display_names") == {target: "新项目名称", other: "另一个项目"}


def test_rename_does_not_discard_already_pending_line_list(qapp, monkeypatch):
    from ui.controllers import project_controller as projects_module
    import ui.controllers.project_identity_controller as module

    pc = ProjectController()
    pc._current = NS(project_id="P", root_path="D:/old", name="旧项目")
    changed = NS(project_id="P", root_path="D:/old", name="新项目")
    pc.set_backend(
        NS(
            backend=NS(
                projects=NS(rename_project=lambda *a: changed, list_lines=lambda *a: [NS(line_id="L01", name="原名称")])
            )
        )
    )
    commands = []
    monkeypatch.setattr(module, "run_command", lambda command, **kw: commands.append(command))
    monkeypatch.setattr(projects_module, "run_command", lambda command, **kw: commands.append(command))
    got = []
    pc.lines_updated.connect(got.append)
    pc.refresh_lines()
    pc.identity.rename_project("P", "新项目")
    commands[1].execute()
    commands[0].execute()
    qapp.processEvents()
    assert got and got[0][0].line_id == "L01"


def test_rename_refreshes_names_without_overriding_future_imports(qapp, monkeypatch):
    from ui.controllers import project_controller as projects_module
    from mygpr.domain.project.models import ProjectLine
    import ui.controllers.project_identity_controller as module

    pc = ProjectController()
    pc._current = NS(project_id="P", root_path="D:/old", name="项目")
    renamed = ProjectLine("L01", "用户改名")
    rows = [ProjectLine("L01", "旧名称")]

    def rename(*args):
        rows[0] = renamed
        return renamed

    projects = NS(rename_line=rename, list_lines=lambda *a: list(rows))
    pc.set_backend(NS(backend=NS(projects=projects)))
    commands = []
    monkeypatch.setattr(module, "run_command", lambda command, **kw: commands.append(command))
    monkeypatch.setattr(projects_module, "run_command", lambda command, **kw: commands.append(command))
    got = []
    pc.lines_updated.connect(got.append)
    pc.refresh_lines()
    pc.identity.rename_line("P", "L01", "用户改名")
    commands[1].execute()
    for command in commands[2:]:
        command.execute()
    commands[0].execute()
    qapp.processEvents()
    assert got and got[-1][0].name == "用户改名"
    projects.list_lines = lambda *a: [ProjectLine("L01", "后续重新导入的名称")]
    pc.refresh_lines()
    commands[-1].execute()
    qapp.processEvents()
    assert got[-1][0].name == "后续重新导入的名称"


def test_late_old_line_request_never_rolls_back_newer_import_name(qapp, monkeypatch):
    from mygpr.domain.project.models import ProjectLine
    from ui.controllers import project_controller as projects_module
    import ui.controllers.project_identity_controller as module

    pc = ProjectController()
    pc._current = NS(project_id="P", root_path="D:/old", name="项目")
    projects = NS(
        rename_line=lambda *a: ProjectLine("L01", "用户改名"), list_lines=lambda *a: [ProjectLine("L01", "用户改名")]
    )
    pc.set_backend(NS(backend=NS(projects=projects)))
    commands = []
    monkeypatch.setattr(module, "run_command", lambda command, **kw: commands.append(command))
    monkeypatch.setattr(projects_module, "run_command", lambda command, **kw: commands.append(command))
    got = []
    pc.lines_updated.connect(got.append)
    pc.refresh_lines()
    pc.identity.rename_line("P", "L01", "用户改名")
    commands[1].execute()
    projects.list_lines = lambda *a: [ProjectLine("L01", "导入新名称")]
    pc.refresh_lines()
    commands[-1].execute()
    qapp.processEvents()
    assert got[-1][0].name == "导入新名称"
    commands[0].execute()
    qapp.processEvents()
    assert got[-1][0].name == "导入新名称"
