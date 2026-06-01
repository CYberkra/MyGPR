#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight checks before packaging MyGPR."""

from __future__ import annotations

import ast
import gc
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GENERATED_ARTIFACT_SUFFIXES = {".out", ".h5", ".hdf5", ".vti", ".vtk", ".vtu"}
GENERATED_GPRMAX_OUTPUT_MARKERS = {
    "/converted/",
    "/paired_outputs/",
    "/gpu_smoke/",
    "/smoke_outputs/",
}
GENERATED_GPRMAX_OUTPUT_SUFFIXES = {".csv", ".npy", ".png", ".jpg", ".jpeg", ".json"}

LOCAL_ABSOLUTE_PATH_RE = re.compile(r"(?i)(?:(?<![A-Za-z])[A-Z]:[\\/]|/Users/|/home/[^/\s]+/|/mnt/[A-Za-z0-9_.-]+/)")
LOCAL_PATH_SCAN_SUFFIXES = {".py", ".bat", ".cmd", ".ps1", ".json", ".yaml", ".yml", ".md", ".txt"}
LOCAL_PATH_SCAN_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "dist", "build", ".venv", "venv"}
LOCAL_PATH_WARNING_LIMIT = 12
HISTORICAL_LOCAL_PATH_DOC_PREFIXES = {"docs", "doc", "artifacts"}


def _normalize_git_path(path: str) -> str:
    return str(path).replace("\\", "/").strip()


def _is_generated_artifact_path(path: str) -> bool:
    normalized = _normalize_git_path(path)
    suffix = Path(normalized).suffix.lower()
    if suffix in GENERATED_ARTIFACT_SUFFIXES:
        return True
    if normalized.startswith("experiments/gprmax/") and suffix in GENERATED_GPRMAX_OUTPUT_SUFFIXES:
        return any(marker in normalized for marker in GENERATED_GPRMAX_OUTPUT_MARKERS)
    return False


def _staged_files() -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _iter_repository_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        rel_parts = set(rel.parts)
        if rel_parts & LOCAL_PATH_SCAN_SKIP_DIRS:
            continue
        # Historical audit/report markdown intentionally preserves local paths.
        # Active config, scripts, launchers, and campaign YAML are still scanned.
        if rel.parts and rel.parts[0] in HISTORICAL_LOCAL_PATH_DOC_PREFIXES:
            continue
        if rel.parts and rel.parts[0] == "tests":
            continue
        if path.suffix.lower() == ".md":
            continue
        if rel.as_posix() == "scripts/preflight_check.py":
            continue
        if path.is_file() and path.suffix.lower() in LOCAL_PATH_SCAN_SUFFIXES:
            files.append(path)
    return files


def find_local_absolute_path_references(limit: int = LOCAL_PATH_WARNING_LIMIT) -> list[str]:
    """Return concise warnings for local absolute path literals kept in source/docs."""
    hits: list[str] = []
    for path in _iter_repository_text_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not LOCAL_ABSOLUTE_PATH_RE.search(line):
                continue
            rel = path.relative_to(ROOT).as_posix()
            excerpt = line.strip()
            if len(excerpt) > 140:
                excerpt = excerpt[:137] + "..."
            hits.append(f"{rel}:{lineno}: {excerpt}")
            if len(hits) >= limit:
                return hits
    return hits


def check_local_absolute_path_warnings() -> None:
    hits = find_local_absolute_path_references()
    if not hits:
        print("[OK] Local absolute path scan")
        return
    print("[WARN] Local absolute path references found; keep these local-only or move to config/env:")
    for item in hits:
        print(f"  - {item}")


def check_staged_generated_artifacts() -> None:
    blocked = [path for path in _staged_files() if _is_generated_artifact_path(path)]
    if blocked:
        details = "\n".join(f"  - {path}" for path in blocked)
        raise AssertionError(
            "Generated gprMax/native artifacts are staged and must stay out of "
            f"MyGPR source:\n{details}"
        )
    print("[OK] Staged generated artifact guard")


def check_syntax() -> None:
    targets = [
        ROOT / "app_qt.py",
        ROOT / "ui" / "gui_workbench.py",
        ROOT / "ui" / "gui_param_editor.py",
        ROOT / "core" / "workflow_executor.py",
        ROOT / "core" / "processing_engine.py",
        ROOT / "core" / "shared_data_model.py",
        ROOT / "core" / "shared_data_state.py",
        ROOT / "ui" / "shared_data_qt_adapter.py",
        ROOT / "ui" / "report_export_controller.py",
        ROOT / "ui" / "processing_lineage_controller.py",
        ROOT / "ui" / "bscan_interaction_controller.py",
        ROOT / "ui" / "autotune_sync_controller.py",
        ROOT / "core" / "app_paths.py",
        ROOT / "scripts" / "auto_tune_validation" / "background_window_policy.py",
        ROOT / "scripts" / "auto_tune_validation" / "run_background_ntraces_edge_check.py",
        ROOT / "scripts" / "auto_tune_validation" / "run_relative_background_window_policy.py",
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
    try:
        data = np.tile(np.linspace(0, 10, 80, dtype=np.float32)[:, None], (1, 16))
        win.shared_data.load_data(data, path="demo.csv", source="preflight")

        # Main workspace should accept loaded data and render without legacy Workbench.
        assert getattr(win, "page_workbench", None) is None, (
            "Legacy Workbench must not be part of the active main UI"
        )
        assert win.data is not None and win.original_data is not None
        win.plot_data(win.data)
        app.processEvents()

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
    check_staged_generated_artifacts()
    check_local_absolute_path_warnings()
    check_runtime_flows()
    print("[OK] Preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
