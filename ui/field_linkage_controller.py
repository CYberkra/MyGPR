#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UI-side coordinator for project events and module refreshes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.project_dependency_rules import resolve_event_impact
from core.project_events import ProjectEvent, ProjectEventType
from core.project_state_tracker import ProjectStateTracker


class ProjectLinkageController:
    """Coordinate event recording, stale-state updates and page refreshes.

    The controller keeps field pages from calling each other directly.  Pages
    emit semantic events; this controller persists project_state.json, updates
    report stale metadata and asks the workbench shell to refresh relevant UI.
    """

    def __init__(self, window: Any) -> None:
        self.window = window

    def emit(
        self,
        event_type: str,
        *,
        line_id: str | None = None,
        reason: str = "",
        changed_paths: list[str | Path] | None = None,
        payload: dict[str, Any] | None = None,
        refresh: bool = True,
        switch_to: str | None = None,
    ) -> dict[str, Any] | None:
        store = getattr(self.window, "project_store", None)
        project_root = getattr(self.window, "project_root", None)
        if store is None or project_root is None:
            return None
        event = ProjectEvent.create(
            event_type,
            project_root=project_root,
            line_id=line_id or getattr(self.window, "selected_line", None),
            reason=reason,
            changed_paths=changed_paths or [],
            payload=payload or {},
        )
        state = ProjectStateTracker(project_root).record_event(event)
        impact = resolve_event_impact(event)
        self._sync_report_manifest(event, state)
        try:
            store.append_log(self._event_log_message(event, impact))
        except Exception:
            pass
        if refresh:
            self.refresh_for_event(event, switch_to=switch_to)
        return state

    def refresh_for_event(self, event: ProjectEvent, *, switch_to: str | None = None) -> None:
        if hasattr(self.window, "_sync_project_lines_to_ui"):
            self.window._sync_project_lines_to_ui()
        if hasattr(self.window, "_refresh_project_selector_combo"):
            self.window._refresh_project_selector_combo()
        if hasattr(self.window, "_refresh_project_widgets"):
            self.window._refresh_project_widgets()
        if hasattr(self.window, "_refresh_processing_preview"):
            self.window._refresh_processing_preview()
        if hasattr(self.window, "_refresh_target_source_options"):
            self.window._refresh_target_source_options()
        if hasattr(self.window, "_refresh_target_widgets"):
            self.window._refresh_target_widgets()
        if switch_to and hasattr(self.window, "switch_workspace"):
            self.window.switch_workspace(switch_to)

    def _sync_report_manifest(self, event: ProjectEvent, state: dict[str, Any]) -> None:
        store = getattr(self.window, "project_store", None)
        if store is None:
            return
        dirty_report = bool((state.get("dirty") or {}).get("report"))
        if event.event_type == ProjectEventType.REPORT_GENERATED:
            if isinstance(store.manifest.reports, dict):
                store.manifest.reports.pop("stale_reason", None)
                store.manifest.reports.pop("stale_reasons", None)
            try:
                store.save_manifest()
            except Exception:
                pass
            return
        if not dirty_report:
            return
        reasons = list((state.get("stale_reasons") or {}).get("report") or [])
        if not isinstance(store.manifest.reports, dict):
            store.manifest.reports = {}
        store.manifest.reports["status"] = "需重新生成"
        store.manifest.reports["stale_reason"] = reasons[-1] if reasons else event.reason or "项目数据已变化"
        store.manifest.reports["stale_reasons"] = reasons[-8:]
        try:
            store.save_manifest()
        except Exception:
            pass

    @staticmethod
    def _event_log_message(event: ProjectEvent, impact: Any) -> str:
        label = event.event_type
        line = f" {event.line_id}" if event.line_id else ""
        reason = f"：{event.reason}" if event.reason else ""
        dirty = ",".join(sorted(impact.dirty_modules)) if getattr(impact, "dirty_modules", None) else ""
        suffix = f"；dirty={dirty}" if dirty else ""
        return f"项目事件 {label}{line}{reason}{suffix}"


__all__ = ["ProjectLinkageController"]
