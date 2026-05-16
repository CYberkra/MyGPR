#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight checks before packaging GPR GUI."""

from __future__ import annotations

import ast
import gc
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
        ROOT / "ui" / "gui_workflow_page.py",
        ROOT / "core" / "workflow_executor.py",
        ROOT / "core" / "workflow_data.py",
        ROOT / "core" / "workflow_registry.py",
        ROOT / "core" / "workflow_runtime_contracts.py",
        ROOT / "core" / "processing_engine.py",
        ROOT / "core" / "shared_data_state.py",
        ROOT / "core" / "app_paths.py",
    ]
    for path in targets:
        ast.parse(path.read_text(encoding="utf-8"))
    print("[OK] Syntax checks")


def check_workflow_registry() -> None:
    from core.workflow_registry import validate_workflow_registry

    errors = [
        issue
        for issue in validate_workflow_registry()
        if issue.severity == "error"
    ]
    assert not errors, "Workflow registry errors: " + "; ".join(
        f"{issue.code}: {issue.message}" for issue in errors
    )
    print("[OK] Workflow registry contract")


def check_runtime_flows() -> None:
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    from app_qt import GPRGuiQt
    from core.workflow_data import WorkflowMethod

    app = QApplication.instance() or QApplication([])
    win = GPRGuiQt()
    try:
        data = np.tile(np.linspace(0, 10, 80, dtype=np.float32)[:, None], (1, 16))
        win.shared_data.load_data(data, path="demo.csv", source="preflight")

        # Workflow apply should compute and commit the latest method result.
        method = WorkflowMethod(
            category="preprocessing",
            stage_id="trace_correction",
            method_id="dewow",
            params={"window": 5},
        )
        win.run_workflow_methods([method], realtime=False)
        deadline = time.time() + 5
        while (win._worker is not None or win._worker_thread is not None) and time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert win._worker is None, "Workflow worker should finish"
        assert not np.array_equal(win.data, win.original_data), (
            "Workflow result should update shared current data"
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
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()
        gc.collect()

    print("[OK] Runtime smoke flows")


def main() -> int:
    check_syntax()
    check_workflow_registry()
    check_runtime_flows()
    print("[OK] Preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
