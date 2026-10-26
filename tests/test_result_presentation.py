"""Result labels use persisted executed steps, including upstream runs."""

from importlib import import_module, util
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def presentation():
    def describe(*args):
        assert util.find_spec("ui.result_presentation"), "Result presentation missing"
        return import_module("ui.result_presentation").describe_artifact(*args)

    return describe


def artifact(aid, methods, *, parent="", previous="", name="处理结果_L01", line="L01"):
    steps = [dict(method_id=method, params={"window": 41}) for method in methods]
    return NS(
        artifact_id=aid,
        line_id=line,
        name=name,
        method_id=methods[-1] if methods else "",
        created_at="2026-10-26T12:34:56Z",
        shape=(18, 20),
        dtype="float32",
        manifest={
            "params": dict(
                lineage_version=2,
                lineage=steps,
                run_step_index=len(steps),
                run_input_artifact_id=parent,
                step_input_artifact_id=previous,
                run_group_id="0123456789abcdef",
            )
        },
    )


def test_run_prefix_is_not_double_counted(presentation):
    first = artifact("A", ["dewow"])
    final = artifact("B", ["dewow", "agcGain"], previous="A")
    result = presentation(final, [first, final])
    assert result.complete and len(result.steps) == 2
    assert "L01" in result.title and "漂移" in result.title and "AGC" in result.title
    assert "012345" not in result.title
    assert "41" in result.tooltip and "12:34:56" in result.tooltip


def test_followup_run_includes_upstream_saved_history(presentation):
    first = artifact("A", ["dewow", "agcGain"])
    last = artifact("B", ["time_cut"], parent="A", previous="A")
    result = presentation(last, [first, last])
    assert result.complete and [s["method_id"] for s in result.steps] == ["dewow", "agcGain", "time_cut"]
    assert "原始数据" in result.path and "裁剪" in result.path
    assert "本次新增" in result.tooltip


def test_custom_name_is_kept_alongside_processing_summary(presentation):
    item = artifact("A", ["dewow"], name="洞顶增强")
    result = presentation(item, [item])
    assert "洞顶增强" in result.title and "漂移" in result.title


def test_explicit_empty_result_label_does_not_expose_default_method_id(presentation):
    item = artifact("A", ["dewow"], name="dewow")
    item.manifest["params"]["result_label"] = ""
    result = presentation(item, [item])
    assert result.title == "L01 · 低频漂移校正（Dewow）"


def test_explicit_result_label_takes_precedence_over_internal_name(presentation):
    item = artifact("A", ["dewow"], name="dewow")
    item.manifest["params"]["result_label"] = "洞顶增强"
    assert "洞顶增强" in presentation(item, [item]).title


def test_long_chain_uses_last_step_and_total_count(presentation):
    item = artifact("A", ["dewow"] * 5 + ["time_cut"])
    result = presentation(item, [item])
    assert len(result.steps) == 6 and "6步" in result.title and "裁剪" in result.title
    assert len(result.title) < 65


@pytest.mark.parametrize("case", ["missing", "cycle", "foreign"])
def test_unknown_origin_is_explicit_and_never_guessed(presentation, case):
    last = artifact("B", ["agcGain"], parent="A")
    other = artifact("A", ["dewow"], parent="B" if case == "cycle" else "", line="L02" if case == "foreign" else "L01")
    result = presentation(last, [last] if case == "missing" else [other, last])
    assert not result.complete
    assert "不完整" in result.title and "不完整" in result.tooltip


def test_legacy_generic_title_does_not_invent_processing(presentation):
    item = artifact("A", ["dewow"])
    item.manifest = {}
    result = presentation(item, [item])
    assert not result.steps and not result.complete and "不完整" in result.title


def test_resource_tree_uses_same_executed_algorithm_summary(presentation):
    from ui.file_tree import build_lineage_model

    item = artifact("A", ["dewow", "agcGain"])
    expected = presentation(item, [item])
    node = build_lineage_model([item], "L01")[0].children[0]
    assert node.text == expected.title
    assert "window" in node.tooltip


def test_final_only_result_shows_all_executed_steps_without_inventing_saved_inputs(qapp):
    from ui.pages.processing_page import ProcessingPage
    import numpy as np

    page = ProcessingPage()
    page.set_line_label("L01")
    page.set_original_bundle(NS(matrix=np.ones((18, 20)), vmin=-1, vmax=1))
    final = artifact("B", ["dewow", "agcGain"])
    page.set_artifacts([final])
    page.select_artifact("B")
    buttons = page._saved_path.buttons()
    assert any("漂移" in b.text() for b in buttons)
    first = next(b for b in buttons if "漂移" in b.text())
    assert not first.isEnabled()  # executed, but its matrix was never saved
    assert any("AGC" in b.text() and b.isEnabled() for b in buttons)
    assert "window" in page._workspace.tab_bar().tabToolTip(page._workspace.tab_bar().currentIndex())
    page.close()
    page.deleteLater()
    qapp.processEvents()
