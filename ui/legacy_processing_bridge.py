#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit bridge between the new project workbench and the legacy processor."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, pyqtSignal

from core.command_contract import assert_command_allowed
from core.project_service import ProjectService

if TYPE_CHECKING:
    from app_qt import GPRGuiQt


class LegacyProcessingBridge(QObject):
    result_saved = pyqtSignal(object)

    def __init__(self, project: ProjectService, parent=None):
        super().__init__(parent)
        self.project = project
        self.window: GPRGuiQt | None = None
        self.line_id: str | None = None

    def open_line(self, line_id: str, *, state: str) -> "GPRGuiQt":
        assert_command_allowed(state, "processing")
        from app_qt import GPRGuiQt

        line = self.project.get_line(line_id)
        primary = line.raw_files[0]
        path = Path(primary.path)
        if not path.is_absolute():
            path = self.project.resolve_relative_path(path)
        win = GPRGuiQt(version_text=f"MyGPR 完整处理窗口 - {line.name}")
        if path.is_dir():
            payload = win._load_ascans_folder(str(path))
        elif path.suffix.lower() == ".csv":
            sidecars = self._resolved_sidecars(line.sidecars)
            kwargs = win._build_sidecar_loader_kwargs(str(path))
            kwargs.update(sidecars)
            payload = win._load_single_csv_with_progress(str(path), **kwargs)
        else:
            payload = win._load_common_gpr_file_with_progress(str(path))
        win.shared_data.load_data(
            payload["data"],
            path=str(path),
            header_info=payload.get("header_info"),
            trace_metadata=payload.get("trace_metadata"),
            source="project_bridge",
        )
        win.plot_data(win.data)
        win.statusBar().showMessage("结果不会自动写入项目；请回到主界面点击“保存处理结果”。")
        win.show()
        self.window = win
        self.line_id = line_id
        return win

    def save_current_result(self, *, name: str = "完整处理结果"):
        if self.window is None or self.line_id is None or self.window.data is None:
            raise RuntimeError("没有可保存的处理结果")
        replay = self.window.shared_data.get_replay_evidence_package() or {}
        chain = [
            {"label": item.get("label", "处理步骤"), "summary": item.get("summary", {})}
            for item in replay.get("snapshots", [])
        ]
        result = self.project.save_processing_result(
            self.line_id,
            self.window.data,
            name=name,
            processing_chain=chain,
            header_info=self.window.header_info or {},
            trace_metadata=self.window.trace_metadata or {},
        )
        self.result_saved.emit(result)
        return result

    def _resolved_sidecars(self, sidecars: dict[str, str]) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for kind in ("rtk", "imu", "altimeter"):
            value = sidecars.get(kind)
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = self.project.resolve_relative_path(path)
            resolved[f"{kind}_path"] = str(path)
        return resolved

    def close(self) -> None:
        if self.window is not None:
            self.window.close()
            self.window = None
            self.line_id = None


__all__ = ["LegacyProcessingBridge"]
