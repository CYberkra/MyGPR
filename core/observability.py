#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured diagnostics, crash capture, and privacy-aware support bundles."""
from __future__ import annotations

import contextlib
import contextvars
import faulthandler
import json
import logging
import os
import platform
import sys
import threading
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator

from core.storage_primitives import atomic_output_path, atomic_write_json, utc_now

_context: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar("mygpr_log_context", default={})
_NATIVE_CRASH_STREAM = None
_HOOK_LOCK = threading.RLock()
_HOOK_LOG_DIR: Path | None = None
MAX_SUPPORT_FILE_BYTES = 8 * 1024 * 1024
MAX_SUPPORT_BUNDLE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class DiagnosticContext:
    project_id: str = ""
    line_id: str = ""
    job_id: str = ""
    artifact_id: str = ""
    event_id: str = ""

    def compact(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if value}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "schema": "mygpr.log_event.v1",
            "timestamp": utc_now(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "process_id": os.getpid(),
            "thread": record.threadName,
            **_context.get(),
        }
        diagnostic_id = getattr(record, "diagnostic_id", "")
        if diagnostic_id:
            payload["diagnostic_id"] = diagnostic_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@contextlib.contextmanager
def diagnostic_context(**values: str) -> Iterator[None]:
    merged = dict(_context.get())
    merged.update({key: str(value) for key, value in values.items() if value})
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


def configure_structured_logging(log_dir: str | Path, *, level: int = logging.INFO) -> Path:
    root = Path(log_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "mygpr-events.jsonl"
    handler = RotatingFileHandler(path, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    handler.setLevel(level)
    logger = logging.getLogger("mygpr")
    if not any(isinstance(item, RotatingFileHandler) and Path(item.baseFilename) == path for item in logger.handlers):
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = True
    return path


def new_diagnostic_id(prefix: str = "ERR") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12].upper()}"


def write_crash_report(
    log_dir: str | Path,
    exc_type: type[BaseException],
    exc_value: BaseException,
    tb: TracebackType | None,
    *,
    thread_name: str = "MainThread",
) -> Path:
    """Persist one uncaught Python exception without touching project data."""
    root = Path(log_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    diagnostic_id = new_diagnostic_id("CRASH")
    payload = {
        "schema": "mygpr.crash_report.v1",
        "diagnostic_id": diagnostic_id,
        "created_at": utc_now(),
        "process_id": os.getpid(),
        "thread": thread_name,
        "platform": platform.platform(),
        "python": sys.version,
        "exception_type": f"{exc_type.__module__}.{exc_type.__qualname__}",
        "exception_message": str(exc_value),
        "traceback": "".join(traceback.format_exception(exc_type, exc_value, tb)),
        "context": dict(_context.get()),
    }
    path = root / f"crash-{diagnostic_id}.json"
    atomic_write_json(path, payload)
    return path


def install_global_exception_hooks(log_dir: str | Path) -> Path:
    """Install process and worker-thread crash hooks once for the application."""
    global _HOOK_LOG_DIR, _NATIVE_CRASH_STREAM
    root = Path(log_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _HOOK_LOCK:
        if _HOOK_LOG_DIR == root:
            return root
        previous_sys_hook = sys.excepthook
        previous_thread_hook = threading.excepthook

        def process_hook(exc_type, exc_value, tb) -> None:
            if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
                previous_sys_hook(exc_type, exc_value, tb)
                return
            try:
                report = write_crash_report(root, exc_type, exc_value, tb)
                logging.getLogger("mygpr.crash").critical(
                    "Unhandled exception; crash report=%s", report, exc_info=(exc_type, exc_value, tb)
                )
            except OSError:
                logging.getLogger("mygpr.crash").critical(
                    "Unhandled exception and crash report write failed", exc_info=(exc_type, exc_value, tb)
                )
            previous_sys_hook(exc_type, exc_value, tb)

        def thread_hook(args: threading.ExceptHookArgs) -> None:
            if issubclass(args.exc_type, (KeyboardInterrupt, SystemExit)):
                previous_thread_hook(args)
                return
            exc_value = args.exc_value or RuntimeError(
                f"Unhandled {args.exc_type.__name__} without an exception value"
            )
            exc_info = (args.exc_type, exc_value, args.exc_traceback)
            try:
                report = write_crash_report(
                    root,
                    args.exc_type,
                    exc_value,
                    args.exc_traceback,
                    thread_name=args.thread.name if args.thread is not None else "unknown-thread",
                )
                logging.getLogger("mygpr.crash").critical(
                    "Unhandled worker exception; crash report=%s",
                    report,
                    exc_info=exc_info,
                )
            except OSError:
                logging.getLogger("mygpr.crash").critical(
                    "Unhandled worker exception and crash report write failed",
                    exc_info=exc_info,
                )
            previous_thread_hook(args)

        sys.excepthook = process_hook
        threading.excepthook = thread_hook
        if _NATIVE_CRASH_STREAM is None:
            native_path = root / "native-crash.log"
            try:
                _NATIVE_CRASH_STREAM = native_path.open("a", encoding="utf-8")
                faulthandler.enable(file=_NATIVE_CRASH_STREAM, all_threads=True)
            except OSError:
                _NATIVE_CRASH_STREAM = None
        _HOOK_LOG_DIR = root
    return root


def _redact_manifest(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {str(k): _redact_manifest(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_manifest(item, key=key) for item in value]
    if isinstance(value, str):
        lowered = key.lower()
        if lowered.endswith(("_path", "_root")) or lowered in {"path", "source", "source_file"}:
            return "<redacted-path>" if value else ""
    return value


def _support_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in (
        "logs/**/*.log",
        "logs/**/*.jsonl",
        "logs/crash-*.json",
        "logs/jobs/*.json",
        "metadata/quarantine/**/*.json",
    ):
        candidates.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(candidates))


def build_support_bundle(
    project_root: str | Path,
    destination: str | Path,
    *,
    include_project_manifest: bool = True,
) -> Path:
    """Create an atomic diagnostics ZIP; raw radar and source files are excluded."""
    root = Path(project_root).resolve(strict=True)
    destination_path = Path(destination).resolve()
    included: list[str] = []
    skipped: list[dict[str, Any]] = []
    total_bytes = 0

    with atomic_output_path(destination_path, suffix=".support.tmp") as temporary:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in _support_candidates(root):
                try:
                    resolved = path.resolve(strict=True)
                    relative = resolved.relative_to(root)
                except (OSError, ValueError):
                    skipped.append({"path": path.name, "reason": "outside-project-or-unreadable"})
                    continue
                if path.is_symlink() or resolved.is_symlink():
                    skipped.append({"path": relative.as_posix(), "reason": "symlink"})
                    continue
                size = resolved.stat().st_size
                if size > MAX_SUPPORT_FILE_BYTES or total_bytes + size > MAX_SUPPORT_BUNDLE_BYTES:
                    skipped.append({"path": relative.as_posix(), "reason": "size-limit", "size_bytes": size})
                    continue
                archive.write(resolved, relative.as_posix())
                included.append(relative.as_posix())
                total_bytes += size

            if include_project_manifest:
                for name in ("project.json", "project.mygpr.json"):
                    path = root / name
                    if not path.is_file() or path.is_symlink():
                        continue
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        skipped.append({"path": name, "reason": "manifest-unreadable"})
                        continue
                    archive.writestr(name, json.dumps(_redact_manifest(payload), ensure_ascii=False, indent=2))
                    included.append(name)
                    break

            metadata = {
                "schema": "mygpr.support_bundle.v2",
                "created_at": utc_now(),
                "platform": platform.platform(),
                "python": sys.version,
                "project_root_name": root.name,
                "privacy": "raw-data-and-source-paths-excluded",
                "included": included,
                "skipped": skipped,
                "uncompressed_bytes": total_bytes,
            }
            archive.writestr("support_bundle_manifest.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    return destination_path


__all__ = [
    "DiagnosticContext",
    "JsonFormatter",
    "MAX_SUPPORT_BUNDLE_BYTES",
    "MAX_SUPPORT_FILE_BYTES",
    "build_support_bundle",
    "configure_structured_logging",
    "diagnostic_context",
    "install_global_exception_hooks",
    "new_diagnostic_id",
    "write_crash_report",
]
