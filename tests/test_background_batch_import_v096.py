from __future__ import annotations

from pathlib import Path

from core.field_project_store import FieldProjectStore
from core.field_project_operations import batch_import_line_data


def test_batch_import_reports_file_progress_and_continues(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="progress-test")
    valid = Path("sample_data/gui_sidecar_all_data_main.csv")
    invalid = tmp_path / "broken.csv"
    invalid.write_text("not,a,gpr,file\n", encoding="utf-8")
    progress: list[tuple[int, int, bool, str]] = []

    def on_progress(current, total, result):
        progress.append((current, total, bool(result.success), result.line_id))

    summary = batch_import_line_data(store, [valid, invalid], progress_callback=on_progress)

    assert summary.total == 2
    assert summary.succeeded == 1
    assert summary.failed == 1
    assert len(progress) == 2
    assert progress[0][0:2] == (1, 2)
    assert progress[1][0:2] == (2, 2)
    assert store.list_lines()[0].gpr_dataset_path


def test_batch_import_cancel_marks_remaining_files(tmp_path: Path) -> None:
    store = FieldProjectStore.create_empty(tmp_path / "project", name="cancel-test")
    valid = Path("sample_data/gui_sidecar_all_data_main.csv")
    calls = {"n": 0}

    def cancel_requested() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    summary = batch_import_line_data(store, [valid, valid], cancel_requested=cancel_requested)

    assert summary.total == 2
    assert summary.succeeded == 1
    assert summary.failed == 1
    assert "取消" in summary.results[1].message


def test_field_workbench_uses_progress_dialog_for_batch_import() -> None:
    source = Path("ui/field_panels/project_page.py").read_text(encoding="utf-8")
    start = source.index("def _action_batch_import_lines_dialog")
    end = source.index("def _action_import_trajectory_dialog", start)
    body = source[start:end]
    assert "BatchImportProgressDialog" in body
    assert "batch_import_line_data(" not in body
