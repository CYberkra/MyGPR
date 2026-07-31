#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless project-level backend smoke flow."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from mygpr.application.jobs.models import JobStatus
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend


def run_project_smoke(root: str | Path | None = None) -> dict[str, object]:
    temporary = None
    if root is None:
        temporary = tempfile.TemporaryDirectory(prefix="mygpr-project-smoke-")
        project_root = Path(temporary.name) / "project"
    else:
        project_root = Path(root).expanduser().resolve()
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        summary = backend.projects.create_project(project_root, name="Project smoke")
        rows, cols = 64, 40
        y = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, cols, dtype=np.float32)[None, :]
        raw = (0.05 * y + np.exp(-((y - 0.6 - 0.02 * np.sin(6.28 * x)) ** 2) / 0.004)).astype(np.float32)
        backend.projects.save_line_dataset(
            summary.project_id,
            "L01",
            raw,
            name="Smoke line",
            length_m=20.0,
            time_window_ns=450.0,
        )
        pipeline = PipelineDefinition(
            name="Project smoke pipeline",
            steps=(PipelineStep("dewow", {"window": 11}),),
        )
        job_id = backend.submit_project_pipeline(summary.project_id, "L01", pipeline)
        snapshot = backend.jobs.wait(job_id, timeout=30)
        audit = backend.projects.audit_project(summary.project_id)
        artifact = snapshot.result
        return {
            "backend_api_version": backend.api_version,
            "project_id": summary.project_id,
            "line_shape": list(backend.projects.get_dataset_info(summary.project_id, "L01").shape),
            "job_status": snapshot.status.value,
            "artifact_id": getattr(artifact, "artifact_id", ""),
            "artifact_reference": getattr(artifact, "data_reference", ""),
            "integrity_healthy": audit.healthy,
            "qt_loaded": any(name.startswith("PyQt") for name in __import__("sys").modules),
        }
    finally:
        backend.shutdown()
        if temporary is not None:
            temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="")
    args = parser.parse_args()
    payload = run_project_smoke(args.root or None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["job_status"] == JobStatus.COMPLETED.value and payload["integrity_healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
