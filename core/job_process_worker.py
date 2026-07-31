#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spawn-isolated execution for serializable CPU/native jobs."""
from __future__ import annotations

import importlib
import multiprocessing as mp
import queue
import traceback
from dataclasses import dataclass, field
from typing import Any

from core.job_manager import JobCancelled


@dataclass(frozen=True)
class ProcessTaskSpec:
    module: str
    function: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.module or not self.function:
            raise ValueError("Process task requires module and function names")
        if self.function.startswith("_"):
            raise ValueError("Private callables cannot be launched as process jobs")


def _child_entry(spec: ProcessTaskSpec, output: mp.Queue) -> None:
    try:
        module = importlib.import_module(spec.module)
        function = getattr(module, spec.function)
        output.put(("ok", function(*spec.args, **spec.kwargs)))
    except BaseException as exc:  # child boundary must serialize all failures
        output.put(("error", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}))


class ProcessJobExecutor:
    """Execute a top-level callable in a fresh spawned process.

    Cancellation terminates the process tree boundary rather than waiting for a
    native library to return to Python.  Results must be pickle-serializable.
    """

    def __init__(self, *, poll_interval_s: float = 0.1) -> None:
        self.poll_interval_s = max(float(poll_interval_s), 0.02)

    def execute(self, spec: ProcessTaskSpec, *, cancel_requested=None, timeout_s: float | None = None) -> Any:
        spec.validate()
        context = mp.get_context("spawn")
        output: mp.Queue = context.Queue(maxsize=1)
        process = context.Process(target=_child_entry, args=(spec, output), daemon=False)
        process.start()
        started = __import__("time").monotonic()
        try:
            while process.is_alive():
                if cancel_requested is not None and cancel_requested():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                    raise JobCancelled("隔离进程任务已取消")
                if timeout_s is not None and __import__("time").monotonic() - started > timeout_s:
                    process.terminate()
                    process.join(timeout=5)
                    raise TimeoutError(f"Process job timed out after {timeout_s:.1f}s")
                process.join(timeout=self.poll_interval_s)
            try:
                state, payload = output.get_nowait()
            except queue.Empty as exc:
                raise RuntimeError(f"Process job exited without a result (exitcode={process.exitcode})") from exc
            if state == "ok":
                return payload
            raise RuntimeError(f"{payload['type']}: {payload['message']}\n{payload['traceback']}")
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            output.close()


__all__ = ["ProcessJobExecutor", "ProcessTaskSpec"]
