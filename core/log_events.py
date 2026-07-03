"""Structured runtime log events for MyGPR."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

_EVENT_LEVELS = {
    "ERR": "error",
    "WARN": "warning",
    "DATA": "info",
    "METHOD": "info",
    "EXPORT": "info",
    "SYS": "info",
    "INFO": "info",
}


def classify_log_event(message: str) -> str:
    """Classify a message into MyGPR's compact UI event tags."""

    text = str(message or "")
    lower = text.lower()
    if any(token in text for token in ("失败", "错误", "异常")) or "error" in lower:
        return "ERR"
    if any(token in text for token in ("警告", "风险", "提示", "NaN", "Inf")):
        return "WARN"
    if any(token in text for token in ("已加载", "导入", "头信息", "shape=")):
        return "DATA"
    if any(token in text for token in ("正在应用", "应用", "处理", "预设")):
        return "METHOD"
    if any(token in text for token in ("导出", "保存", "Evidence", "报告")):
        return "EXPORT"
    if any(token in text for token in ("版本", "欢迎", "主题")):
        return "SYS"
    return "INFO"


@dataclass(slots=True)
class LogEvent:
    """A single structured UI/runtime event."""

    timestamp: str
    event_type: str
    level: str
    message: str
    source: str = "ui"
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        message: str,
        *,
        event_type: str | None = None,
        source: str = "ui",
        level: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "LogEvent":
        tag = (event_type or classify_log_event(message)).upper()
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            event_type=tag,
            level=level or _EVENT_LEVELS.get(tag, "info"),
            message=str(message),
            source=str(source or "ui"),
            context=dict(context or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mygpr.log_event.v1",
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "level": self.level,
            "source": self.source,
            "message": self.message,
            "context": dict(self.context or {}),
        }


class LogEventBuffer:
    """Small bounded in-memory structured log buffer."""

    def __init__(self, max_events: int = 1000):
        self.max_events = max(1, int(max_events or 1000))
        self._events: list[LogEvent] = []

    def append(self, event: LogEvent) -> None:
        self._events.append(event)
        overflow = len(self._events) - self.max_events
        if overflow > 0:
            del self._events[:overflow]

    def extend(self, events: Iterable[LogEvent]) -> None:
        for event in events:
            self.append(event)

    def to_list(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._events]

    def text_lines(self) -> list[str]:
        lines = []
        for event in self._events:
            local_time = event.timestamp.replace("+00:00", "Z")
            lines.append(f"[{local_time}] {event.event_type} {event.message}")
        return lines

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._events)
