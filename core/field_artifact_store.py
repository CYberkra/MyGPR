#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processed-result artifact persistence mixin for field projects."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.field_project_models import local_now, validate_line_id

PROCESSING_SAVE_SCHEMA = "mygpr.processing_save.v3"
CATALOG_COMMIT_ERRORS = (
    sqlite3.Error,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    KeyError,
)


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


def _artifact_paths(root: Path, line_id: str, timestamp: str, artifact_id: str) -> dict[str, Path]:
    line_dir = root / "processed" / line_id
    line_dir.mkdir(parents=True, exist_ok=True)
    return {
        "line_dir": line_dir,
        "params": line_dir / f"{line_id}_params_{timestamp}.json",
        "latest_params": line_dir / f"{line_id}_params.json",
        "manifest": line_dir / f"{line_id}_processing_manifest_{timestamp}.json",
        "descriptor": line_dir / f"{artifact_id}.artifact",
    }


def _base_manifest(
    *,
    line_id: str,
    artifact_id: str,
    shape: tuple[int, int],
    params: dict[str, Any],
    saved_at: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str, str, str]:
    raw_manifest = dict(params.get("manifest") or {})
    method_id = str(raw_manifest.get("method_id") or params.get("method") or "")
    method_name = str(raw_manifest.get("method_name") or params.get("method_name") or method_id)
    input_dataset = params.get("input_dataset") if isinstance(params.get("input_dataset"), dict) else {}
    collected_params = params.get("params") if isinstance(params.get("params"), dict) else {}
    branch_id = str(params.get("branch_id") or raw_manifest.get("branch_id") or f"{line_id}:main")
    parent_artifact_id = str(params.get("parent_artifact_id") or raw_manifest.get("parent_artifact_id") or "")
    manifest = {
        **raw_manifest,
        "schema": raw_manifest.get("schema") or "mygpr.processing_manifest.v3",
        "save_schema": PROCESSING_SAVE_SCHEMA,
        "line_id": line_id,
        "source_line_id": str(raw_manifest.get("source_line_id") or raw_manifest.get("line_id") or line_id),
        "artifact_id": artifact_id,
        "artifact_role": raw_manifest.get("artifact_role") or "processing_result",
        "branch_id": branch_id,
        "parent_artifact_id": parent_artifact_id,
        "method_id": method_id,
        "method_name": method_name,
        "params": collected_params,
        "saved_at": saved_at,
        "input_dataset": input_dataset or raw_manifest.get("input_dataset", {}),
        "output_shape": list(shape),
    }
    return manifest, collected_params, method_id, method_name, branch_id, parent_artifact_id


class FieldArtifactStoreMixin:
    """Manage immutable processing artifacts for hybrid and legacy projects."""

    def save_processed_line(
        self,
        line_id: str,
        data: np.ndarray,
        params: dict[str, Any],
        *,
        cancel_requested=None,
        progress_callback=None,
    ) -> tuple[Path, Path]:
        safe_line_id = validate_line_id(line_id)
        line = self.get_line(safe_line_id)
        shape = tuple(int(v) for v in getattr(data, "shape", ()))
        if len(shape) != 2 or not all(v > 0 for v in shape):
            raise ValueError(f"处理结果必须是非空二维矩阵，当前 shape={shape!r}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        artifact_id = f"{safe_line_id}_processed_{timestamp}"
        paths = _artifact_paths(self.root, safe_line_id, timestamp, artifact_id)
        saved_at = local_now()
        manifest, collected_params, method_id, method_name, branch_id, parent_id = _base_manifest(
            line_id=safe_line_id,
            artifact_id=artifact_id,
            shape=shape,
            params=params,
            saved_at=saved_at,
        )
        storage_result = self._persist_processed_matrix(
            line_id=safe_line_id,
            artifact_id=artifact_id,
            matrix=data,
            manifest=manifest,
            collected_params=collected_params,
            branch_id=branch_id,
            parent_artifact_id=parent_id,
            paths=paths,
            cancel_requested=cancel_requested,
            progress_callback=progress_callback,
        )
        return self._finalize_processed_artifact(
            line=line,
            line_id=safe_line_id,
            artifact_id=artifact_id,
            shape=shape,
            saved_at=saved_at,
            method_id=method_id,
            method_name=method_name,
            collected_params=collected_params,
            input_dataset=manifest.get("input_dataset") or {},
            paths=paths,
            storage_result=storage_result,
        )

    def _persist_processed_matrix(
        self,
        *,
        line_id: str,
        artifact_id: str,
        matrix: Any,
        manifest: dict[str, Any],
        collected_params: dict[str, Any],
        branch_id: str,
        parent_artifact_id: str,
        paths: dict[str, Path],
        cancel_requested=None,
        progress_callback=None,
    ) -> dict[str, Any]:
        if getattr(self.storage, "is_hybrid", False):
            result = self.storage.save_processing_artifact(
                line_id=line_id,
                artifact_id=artifact_id,
                matrix=matrix,
                manifest=manifest,
                params=collected_params,
                branch_id=branch_id,
                parent_artifact_id=parent_artifact_id,
                cancel_requested=cancel_requested,
                progress_callback=progress_callback,
            )
            full_manifest = dict(result["manifest"])
            full_manifest.update(
                data_path=result["data_uri"],
                h5_path=result["h5_path"],
                dataset_path=result["dataset_path"],
                output_data_sha256=result["sha256"],
                output_shape=list(result["shape"]),
                output_dtype=result["dtype"],
                storage_mode="hdf5_line_container",
            )
            return {
                "hybrid": True,
                "data_path": self.storage.line_container_path(line_id),
                "data_reference": result["data_uri"],
                "output_hash": str(result["sha256"]),
                "branch_id": str(result.get("branch_id") or branch_id),
                "parent_artifact_id": str(result.get("parent_artifact_id") or parent_artifact_id),
                "manifest": full_manifest,
            }

        if cancel_requested is not None and cancel_requested():
            from core.job_manager import JobCancelled
            raise JobCancelled("处理结果保存已取消")
        matrix_array = np.asarray(matrix, dtype=np.float32)
        data_path = paths["line_dir"] / f"{artifact_id}.npy"
        np.save(data_path, matrix_array)
        if progress_callback is not None:
            progress_callback(1, 1, "处理结果矩阵已保存")
        if cancel_requested is not None and cancel_requested():
            data_path.unlink(missing_ok=True)
            from core.job_manager import JobCancelled
            raise JobCancelled("处理结果保存已取消")
        data_reference = data_path.relative_to(self.root).as_posix()
        full_manifest = {
            **manifest,
            "data_path": data_reference,
            "output_data_sha256": manifest.get("output_data_sha256") or _array_sha256(matrix_array),
            "output_shape": list(matrix_array.shape),
            "output_dtype": str(matrix_array.dtype),
            "storage_mode": "npy",
        }
        return {
            "hybrid": False,
            "data_path": data_path,
            "data_reference": data_reference,
            "output_hash": full_manifest["output_data_sha256"],
            "branch_id": branch_id,
            "parent_artifact_id": parent_artifact_id,
            "manifest": full_manifest,
        }

    def _finalize_processed_artifact(
        self,
        *,
        line: Any,
        line_id: str,
        artifact_id: str,
        shape: tuple[int, int],
        saved_at: str,
        method_id: str,
        method_name: str,
        collected_params: dict[str, Any],
        input_dataset: dict[str, Any],
        paths: dict[str, Path],
        storage_result: dict[str, Any],
    ) -> tuple[Path, Path]:
        manifest = dict(storage_result["manifest"])
        branch_id = str(storage_result["branch_id"])
        parent_id = str(storage_result["parent_artifact_id"])
        for key, path in (
            ("params_path", paths["params"]),
            ("latest_params_path", paths["latest_params"]),
            ("manifest_path", paths["manifest"]),
        ):
            manifest[key] = path.relative_to(self.root).as_posix()
        manifest["branch_id"] = branch_id
        manifest["parent_artifact_id"] = parent_id
        manifest["manifest_sha256"] = _sha256_bytes(
            _json_bytes({key: value for key, value in manifest.items() if key != "manifest_sha256"})
        )
        payload = self._processing_payload(
            line_id=line_id,
            artifact_id=artifact_id,
            shape=shape,
            saved_at=saved_at,
            method_id=method_id,
            method_name=method_name,
            collected_params=collected_params,
            input_dataset=input_dataset,
            paths=paths,
            storage_result=storage_result,
            manifest=manifest,
        )
        manifest["params_sha256"] = payload["params_sha256"]
        if storage_result["hybrid"]:
            self._commit_catalog_manifest(
                line_id=line_id,
                artifact_id=artifact_id,
                branch_id=branch_id,
                parent_artifact_id=parent_id,
                collected_params=collected_params,
                saved_at=saved_at,
                manifest=manifest,
            )
        self._write_processing_sidecars(line_id, artifact_id, saved_at, paths, manifest, payload, storage_result)
        line.processing_status = "已完成"
        line.data_quality = "★★★★★" if line.data_quality not in {"--", ""} else "★★★★☆"
        line.processed_result = (
            paths["descriptor"].relative_to(self.root).as_posix()
            if storage_result["hybrid"] else storage_result["data_reference"]
        )
        line.params_path = paths["params"].relative_to(self.root).as_posix()
        line.updated_at = saved_at
        self.upsert_line(line)
        self.append_log(
            f"保存处理结果 {line_id}: artifact={artifact_id}, "
            f"storage={manifest.get('storage_mode')}, branch={branch_id}"
        )
        return (
            paths["descriptor"] if storage_result["hybrid"] else storage_result["data_path"],
            paths["params"],
        )

    def _processing_payload(self, **values: Any) -> dict[str, Any]:
        paths = values["paths"]
        storage_result = values["storage_result"]
        manifest = values["manifest"]
        payload = {
            "schema": PROCESSING_SAVE_SCHEMA,
            "line_id": values["line_id"],
            "artifact_id": values["artifact_id"],
            "branch_id": storage_result["branch_id"],
            "parent_artifact_id": storage_result["parent_artifact_id"],
            "updated_at": values["saved_at"],
            "method": values["method_id"],
            "method_name": values["method_name"],
            "params": values["collected_params"],
            "input_dataset": values["input_dataset"],
            "data_path": storage_result["data_reference"],
            "params_path": paths["params"].relative_to(self.root).as_posix(),
            "manifest_path": paths["manifest"].relative_to(self.root).as_posix(),
            "output_shape": list(values["shape"]),
            "output_data_sha256": storage_result["output_hash"],
            "manifest_sha256": manifest["manifest_sha256"],
            "storage_mode": manifest.get("storage_mode", ""),
        }
        payload["params_sha256"] = _sha256_bytes(_json_bytes(payload))
        return payload

    def _commit_catalog_manifest(self, **values: Any) -> None:
        manifest = values["manifest"]
        try:
            self.storage.catalog.register_artifact(
                {
                    "artifact_id": values["artifact_id"],
                    "line_id": values["line_id"],
                    "artifact_kind": "processing",
                    "artifact_role": manifest.get("artifact_role") or "processing_result",
                    "branch_id": values["branch_id"],
                    "parent_artifact_id": values["parent_artifact_id"],
                    "h5_path": manifest.get("h5_path", ""),
                    "dataset_path": manifest.get("dataset_path", ""),
                    "status": manifest.get("status") or "success",
                    "dtype": manifest.get("output_dtype", ""),
                    "shape": manifest.get("output_shape", []),
                    "sha256": manifest.get("output_data_sha256", ""),
                    "params": values["collected_params"],
                    "manifest": manifest,
                    "created_at": values["saved_at"],
                }
            )
        except CATALOG_COMMIT_ERRORS:
            self.storage.rollback_processing_artifact(values["line_id"], values["artifact_id"])
            raise

    def _write_processing_sidecars(
        self,
        line_id: str,
        artifact_id: str,
        saved_at: str,
        paths: dict[str, Path],
        manifest: dict[str, Any],
        payload: dict[str, Any],
        storage_result: dict[str, Any],
    ) -> None:
        self.write_json(paths["manifest"], manifest)
        self.write_json(paths["params"], payload)
        self.write_json(paths["latest_params"], {**payload, "latest": True})
        if not storage_result["hybrid"]:
            return
        self.write_json(
            paths["descriptor"],
            {
                "schema": "mygpr.artifact_pointer.v1",
                "artifact_id": artifact_id,
                "line_id": line_id,
                "data_uri": storage_result["data_reference"],
                "h5_path": manifest.get("h5_path", ""),
                "dataset_path": manifest.get("dataset_path", ""),
                "manifest_path": paths["manifest"].relative_to(self.root).as_posix(),
                "params_path": paths["params"].relative_to(self.root).as_posix(),
                "created_at": saved_at,
            },
        )


__all__ = ["FieldArtifactStoreMixin", "PROCESSING_SAVE_SCHEMA"]
