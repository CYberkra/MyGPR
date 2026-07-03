#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processed-result artifact persistence mixin for field projects."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.field_project_models import local_now, validate_line_id

PROCESSING_SAVE_SCHEMA = "mygpr.processing_save.v2"


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(int(v) for v in arr.shape)).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


class FieldArtifactStoreMixin:
    """Manage saved processing artifacts under ``processed/``."""

    def save_processed_line(self, line_id: str, data: np.ndarray, params: dict[str, Any]) -> tuple[Path, Path]:
        """Persist one processed B-scan and its traceability sidecars.

        The saved result is an immutable timestamped artifact.  A timestamped
        parameter JSON and processing manifest are written alongside the matrix
        so artifact indexing and target-source binding can resolve the exact
        algorithm, parameters and input/output fingerprints later.  A
        ``<line_id>_params.json`` file is also maintained as a latest pointer for
        older callers, but indexes must prefer the timestamped sidecar.
        """
        safe_line_id = validate_line_id(line_id)
        line = self.get_line(safe_line_id)
        matrix = np.asarray(data, dtype=np.float32)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError(f"处理结果必须是非空二维矩阵，当前 shape={matrix.shape!r}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        line_dir = self.root / "processed" / safe_line_id
        line_dir.mkdir(parents=True, exist_ok=True)
        data_path = line_dir / f"{safe_line_id}_processed_{timestamp}.npy"
        params_path = line_dir / f"{safe_line_id}_params_{timestamp}.json"
        latest_params_path = line_dir / f"{safe_line_id}_params.json"
        manifest_path = line_dir / f"{safe_line_id}_processing_manifest_{timestamp}.json"

        np.save(data_path, matrix)
        artifact_id = data_path.stem
        saved_at = local_now()
        raw_manifest = dict(params.get("manifest") or {})
        method_id = str(raw_manifest.get("method_id") or params.get("method") or "")
        method_name = str(raw_manifest.get("method_name") or params.get("method_name") or method_id)
        input_dataset = params.get("input_dataset") if isinstance(params.get("input_dataset"), dict) else {}
        collected_params = params.get("params") if isinstance(params.get("params"), dict) else {}

        manifest = {
            **raw_manifest,
            "schema": raw_manifest.get("schema") or "mygpr.processing_manifest.v2",
            "save_schema": PROCESSING_SAVE_SCHEMA,
            "line_id": safe_line_id,
            "source_line_id": str(raw_manifest.get("source_line_id") or raw_manifest.get("line_id") or safe_line_id),
            "artifact_id": artifact_id,
            "artifact_role": raw_manifest.get("artifact_role") or "processing_result",
            "method_id": method_id,
            "method_name": method_name,
            "params": collected_params,
            "saved_at": saved_at,
            "data_path": data_path.relative_to(self.root).as_posix(),
            "params_path": params_path.relative_to(self.root).as_posix(),
            "latest_params_path": latest_params_path.relative_to(self.root).as_posix(),
            "manifest_path": manifest_path.relative_to(self.root).as_posix(),
            "output_data_sha256": raw_manifest.get("output_data_sha256") or _array_sha256(matrix),
            "output_shape": list(matrix.shape),
            "input_dataset": input_dataset or raw_manifest.get("input_dataset", {}),
        }
        manifest_bytes = _json_bytes(manifest)
        manifest["manifest_sha256"] = _sha256_bytes(manifest_bytes)

        payload = {
            "schema": PROCESSING_SAVE_SCHEMA,
            "line_id": safe_line_id,
            "artifact_id": artifact_id,
            "updated_at": saved_at,
            "method": method_id,
            "method_name": method_name,
            "params": collected_params,
            "input_dataset": input_dataset,
            "data_path": data_path.relative_to(self.root).as_posix(),
            "params_path": params_path.relative_to(self.root).as_posix(),
            "manifest_path": manifest_path.relative_to(self.root).as_posix(),
            "output_shape": list(matrix.shape),
            "output_data_sha256": manifest["output_data_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        }
        payload["params_sha256"] = _sha256_bytes(_json_bytes(payload))
        manifest["params_sha256"] = payload["params_sha256"]

        self.write_json(manifest_path, manifest)
        self.write_json(params_path, payload)
        self.write_json(latest_params_path, {**payload, "latest": True})

        line.processing_status = "已完成"
        line.data_quality = "★★★★★" if line.data_quality not in {"--", ""} else "★★★★☆"
        line.processed_result = data_path.relative_to(self.root).as_posix()
        line.params_path = params_path.relative_to(self.root).as_posix()
        line.updated_at = saved_at
        self.upsert_line(line)
        self.append_log(
            f"保存处理结果 {safe_line_id}: {data_path.name}, params={params_path.name}, artifact={artifact_id}"
        )
        return data_path, params_path


__all__ = ["FieldArtifactStoreMixin", "PROCESSING_SAVE_SCHEMA"]
