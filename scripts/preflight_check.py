#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight checks before packaging GPR GUI."""

from __future__ import annotations

import ast
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_syntax() -> None:
    targets = [
        ROOT / "app_qt.py",
        ROOT / "ui" / "gui_workbench.py",
        ROOT / "ui" / "gui_param_editor.py",
        ROOT / "core" / "workflow_executor.py",
        ROOT / "core" / "processing_engine.py",
        ROOT / "core" / "shared_data_state.py",
        ROOT / "core" / "app_paths.py",
    ]
    for path in targets:
        ast.parse(path.read_text(encoding="utf-8"))
    print("[OK] Syntax checks")


def check_runtime_flows() -> None:
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    from app_qt import GPRGuiQt

    app = QApplication.instance() or QApplication([])
    win = GPRGuiQt()

    data = np.tile(np.linspace(0, 10, 80, dtype=np.float32)[:, None], (1, 16))
    win.shared_data.load_data(data, path="demo.csv", source="preflight")

    # Workbench apply should compute and commit the latest method result.
    wb = win.page_workbench
    wb.select_method("dewow")
    wb.param_editor.param_widgets["window"].setValue(5)
    wb._run_current_method()
    deadline = time.time() + 5
    while (
        wb._preview_running or wb._pending_preview_request is not None
    ) and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert wb.preview_data is None, "Applied result should not leave stale preview data"
    assert not np.array_equal(win.data, win.original_data), (
        "Applied result should update shared current data"
    )

    win.undo_last()
    assert np.array_equal(win.data, win.original_data), (
        "Undo should restore original data"
    )

    # 临时对比快照不应覆盖当前正式结果。
    base = win.data.copy()
    win._set_compare_snapshots(
        [
            {"label": "dewow", "data": base * 0.1},
            {"label": "subtracting_average_2D", "data": base * 0.2},
        ]
    )
    assert np.array_equal(win.data, base), (
        "Transient compare snapshots must not overwrite current data"
    )
    assert [snap["label"] for snap in win.compare_snapshots] == [
        "原始",
        "当前",
        "dewow",
        "subtracting_average_2D",
    ]

    win.shared_data.apply_current_data(base + 1, push_history=True, label="dewow")
    assert [snap["label"] for snap in win.compare_snapshots] == ["原始", "当前"]

    # Report should capture last run summary.
    with tempfile.TemporaryDirectory() as tmpdir:
        win._default_output_dir = lambda: tmpdir  # type: ignore[method-assign]
        win._set_last_run_summary(
            "single",
            "预检单次处理",
            [
                {
                    "method_key": "dewow",
                    "method_name": "dewow",
                    "params": {"window": 5},
                    "elapsed_ms": 10.0,
                }
            ],
            notes=["preflight"],
        )
        win.generate_report()
        reports = list(Path(tmpdir).glob("report_*.md"))
        assert reports, "Report was not generated"
        content = reports[0].read_text(encoding="utf-8")
        assert "Last run: 预检单次处理" in content
        assert "preflight" in content

    print("[OK] Runtime smoke flows")


def main() -> int:
    check_syntax()
    check_runtime_flows()
    print("[OK] Preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
