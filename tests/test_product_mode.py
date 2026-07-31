from __future__ import annotations

from core.product_mode import build_workspaces, is_research_ui_enabled


def test_field_mode_hides_research_workspaces_by_default() -> None:
    assert is_research_ui_enabled({}) is False
    assert list(build_workspaces({}).items()) == [
        ("data_management", "项目管理"),
        ("processing_lab", "测线处理"),
        ("interpretation", "界面标注"),
        ("spatial", "空间成果"),
        ("delivery", "成果报告"),
    ]


def test_research_mode_can_reenable_simulation_workspace() -> None:
    env = {"MYGPR_ENABLE_RESEARCH_UI": "1"}
    assert is_research_ui_enabled(env) is True
    assert list(build_workspaces(env).items()) == [
        ("data_management", "项目管理"),
        ("processing_lab", "测线处理"),
        ("simulation_validation", "仿真验证"),
        ("interpretation", "界面标注"),
        ("spatial", "空间成果"),
        ("delivery", "成果报告"),
    ]
