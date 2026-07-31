#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite project catalog for MyGPR hybrid projects.

HDF5 owns large numerical arrays.  SQLite owns project relationships, branches,
artifact lineage, exports and audit records.  Connections are intentionally
short-lived so GUI and worker threads do not share sqlite connection objects.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

from core.storage_primitives import utc_now

CATALOG_SCHEMA_VERSION = 1


def _json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ProjectCatalog:
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path).resolve()
        self.read_only = bool(read_only)

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            uri = f"file:{self.path.as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=30.0)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        if not self.read_only:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise PermissionError("Project catalog is read-only")
        connection = self._connect()
        committed = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
            committed = True
        finally:
            if not committed:
                connection.rollback()
            connection.close()

    def initialize(self, *, project_id: str, project_name: str) -> None:
        if self.read_only:
            return
        with self.transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS catalog_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lines (
                    line_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    h5_path TEXT NOT NULL DEFAULT '',
                    raw_dataset_path TEXT NOT NULL DEFAULT '/raw/bscan',
                    status TEXT NOT NULL DEFAULT '',
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    trace_count INTEGER NOT NULL DEFAULT 0,
                    length_m REAL NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS processing_branches (
                    branch_id TEXT PRIMARY KEY,
                    line_id TEXT NOT NULL REFERENCES lines(line_id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    parent_branch_id TEXT REFERENCES processing_branches(branch_id),
                    head_artifact_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    line_id TEXT NOT NULL REFERENCES lines(line_id) ON DELETE CASCADE,
                    artifact_kind TEXT NOT NULL,
                    artifact_role TEXT NOT NULL,
                    branch_id TEXT REFERENCES processing_branches(branch_id),
                    parent_artifact_id TEXT REFERENCES artifacts(artifact_id),
                    h5_path TEXT NOT NULL DEFAULT '',
                    dataset_path TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'committed',
                    dtype TEXT NOT NULL DEFAULT '',
                    shape_json TEXT NOT NULL DEFAULT '[]',
                    sha256 TEXT NOT NULL DEFAULT '',
                    params_json TEXT NOT NULL DEFAULT '{}',
                    manifest_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_line_kind
                    ON artifacts(line_id, artifact_kind, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_artifacts_parent
                    ON artifacts(parent_artifact_id);
                CREATE TABLE IF NOT EXISTS exports (
                    export_id TEXT PRIMARY KEY,
                    export_kind TEXT NOT NULL,
                    source_artifact_id TEXT REFERENCES artifacts(artifact_id),
                    path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sha256 TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    object_type TEXT NOT NULL DEFAULT '',
                    object_id TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS migration_journal (
                    migration_id TEXT PRIMARY KEY,
                    source_schema TEXT NOT NULL,
                    target_schema TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL DEFAULT ''
                );
                """
            )
            meta = {
                "schema_version": str(CATALOG_SCHEMA_VERSION),
                "project_id": str(project_id),
                "project_name": str(project_name),
                "created_at": utc_now(),
            }
            db.executemany(
                "INSERT INTO catalog_meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                list(meta.items()),
            )

    def set_meta(self, key: str, value: Any) -> None:
        with self.transaction() as db:
            db.execute(
                "INSERT INTO catalog_meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(key), _json(value) if not isinstance(value, str) else value),
            )

    def upsert_line(self, line: Any, *, h5_path: str = "", raw_dataset_path: str = "/raw/bscan") -> None:
        payload = asdict(line) if is_dataclass(line) else dict(line)
        now = str(payload.get("updated_at") or utc_now())
        with self.transaction() as db:
            db.execute(
                """
                INSERT INTO lines(line_id,name,h5_path,raw_dataset_path,status,sample_count,trace_count,length_m,payload_json,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(line_id) DO UPDATE SET
                    name=excluded.name,
                    h5_path=CASE WHEN excluded.h5_path<>'' THEN excluded.h5_path ELSE lines.h5_path END,
                    raw_dataset_path=excluded.raw_dataset_path,
                    status=excluded.status,
                    sample_count=excluded.sample_count,
                    trace_count=excluded.trace_count,
                    length_m=excluded.length_m,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(payload.get("line_id") or ""),
                    str(payload.get("name") or payload.get("line_id") or ""),
                    str(h5_path),
                    str(raw_dataset_path),
                    str(payload.get("processing_status") or ""),
                    int(payload.get("raw_rows") or 0),
                    int(payload.get("trace_count") or 0),
                    float(payload.get("length_m") or 0.0),
                    _json(payload),
                    now,
                ),
            )


    def list_lines(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM lines ORDER BY line_id").fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.get("payload_json") or "{}")
            except (json.JSONDecodeError, TypeError, ValueError):
                item["payload"] = {}
            result.append(item)
        return result

    def delete_line(self, line_id: str) -> None:
        """Delete one line and cascade branches/artifacts from the catalog.

        Export records are retained as historical evidence but their artifact
        foreign key is cleared before the processing artifacts are removed.
        """
        with self.transaction() as db:
            db.execute(
                "UPDATE exports SET source_artifact_id=NULL WHERE source_artifact_id IN "
                "(SELECT artifact_id FROM artifacts WHERE line_id=?)",
                (str(line_id),),
            )
            db.execute("DELETE FROM lines WHERE line_id=?", (str(line_id),))

    def ensure_branch(
        self,
        *,
        line_id: str,
        branch_id: str,
        name: str,
        parent_branch_id: str | None = None,
        head_artifact_id: str = "",
    ) -> None:
        now = utc_now()
        with self.transaction() as db:
            db.execute(
                """
                INSERT INTO processing_branches(
                    branch_id,line_id,name,parent_branch_id,head_artifact_id,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(branch_id) DO UPDATE SET
                    name=excluded.name,
                    parent_branch_id=COALESCE(excluded.parent_branch_id,processing_branches.parent_branch_id),
                    head_artifact_id=CASE
                        WHEN excluded.head_artifact_id<>'' THEN excluded.head_artifact_id
                        ELSE processing_branches.head_artifact_id
                    END,
                    updated_at=excluded.updated_at
                """,
                (branch_id, line_id, name, parent_branch_id, str(head_artifact_id or ""), now, now),
            )

    def list_branches(self, *, line_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            if line_id:
                rows = db.execute(
                    "SELECT * FROM processing_branches WHERE line_id=? ORDER BY created_at",
                    (line_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM processing_branches ORDER BY line_id,created_at"
                ).fetchall()
        return [dict(row) for row in rows]

    def branch_head(self, branch_id: str) -> str:
        with self._connect() as db:
            row = db.execute(
                "SELECT head_artifact_id FROM processing_branches WHERE branch_id=?",
                (str(branch_id),),
            ).fetchone()
        return str(row[0] or "") if row else ""

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM artifacts WHERE artifact_id=?", (str(artifact_id),)).fetchone()
        if row is None:
            return None
        item = dict(row)
        for key in ("shape_json", "params_json", "manifest_json"):
            try:
                item[key[:-5] if key.endswith("_json") else key] = json.loads(item[key])
            except (json.JSONDecodeError, TypeError, ValueError):
                item[key[:-5] if key.endswith("_json") else key] = [] if key == "shape_json" else {}
        return item

    def delete_artifact(self, artifact_id: str) -> None:
        """Remove one artifact and restore its branch head to the parent."""
        with self.transaction() as db:
            row = db.execute(
                "SELECT branch_id,parent_artifact_id FROM artifacts WHERE artifact_id=?",
                (str(artifact_id),),
            ).fetchone()
            if row is None:
                return
            branch_id = str(row["branch_id"] or "")
            parent_id = str(row["parent_artifact_id"] or "")
            db.execute(
                "UPDATE exports SET source_artifact_id=NULL WHERE source_artifact_id=?",
                (str(artifact_id),),
            )
            if branch_id:
                db.execute(
                    "UPDATE processing_branches SET head_artifact_id=?,updated_at=? "
                    "WHERE branch_id=? AND head_artifact_id=?",
                    (parent_id, utc_now(), branch_id, str(artifact_id)),
                )
            db.execute("DELETE FROM artifacts WHERE artifact_id=?", (str(artifact_id),))

    def register_artifact(self, payload: dict[str, Any]) -> None:
        now = str(payload.get("created_at") or utc_now())
        with self.transaction() as db:
            db.execute(
                """
                INSERT INTO artifacts(
                    artifact_id,line_id,artifact_kind,artifact_role,branch_id,parent_artifact_id,
                    h5_path,dataset_path,status,dtype,shape_json,sha256,params_json,manifest_json,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    status=excluded.status,dtype=excluded.dtype,shape_json=excluded.shape_json,
                    sha256=excluded.sha256,params_json=excluded.params_json,
                    manifest_json=excluded.manifest_json,updated_at=excluded.updated_at
                """,
                (
                    str(payload["artifact_id"]), str(payload["line_id"]),
                    str(payload.get("artifact_kind") or "processing"),
                    str(payload.get("artifact_role") or "processing_result"),
                    str(payload.get("branch_id") or "") or None,
                    str(payload.get("parent_artifact_id") or "") or None,
                    str(payload.get("h5_path") or ""), str(payload.get("dataset_path") or ""),
                    str(payload.get("status") or "committed"), str(payload.get("dtype") or ""),
                    _json(payload.get("shape") or []), str(payload.get("sha256") or ""),
                    _json(payload.get("params") or {}), _json(payload.get("manifest") or {}), now, utc_now(),
                ),
            )
            branch_id = str(payload.get("branch_id") or "")
            if branch_id:
                db.execute(
                    "UPDATE processing_branches SET head_artifact_id=?,updated_at=? WHERE branch_id=?",
                    (str(payload["artifact_id"]), utc_now(), branch_id),
                )

    def list_artifacts(self, *, line_id: str | None = None, artifact_kind: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if line_id:
            clauses.append("line_id=?")
            values.append(line_id)
        if artifact_kind:
            clauses.append("artifact_kind=?")
            values.append(artifact_kind)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as db:
            rows = db.execute(f"SELECT * FROM artifacts{where} ORDER BY created_at DESC", values).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in ("shape_json", "params_json", "manifest_json"):
                try:
                    item[key[:-5] if key.endswith("_json") else key] = json.loads(item[key])
                except (json.JSONDecodeError, TypeError, ValueError):
                    item[key[:-5] if key.endswith("_json") else key] = [] if key == "shape_json" else {}
            result.append(item)
        return result

    def register_export(self, payload: dict[str, Any]) -> None:
        with self.transaction() as db:
            db.execute(
                """
                INSERT INTO exports(export_id,export_kind,source_artifact_id,path,status,sha256,metadata_json,created_at)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(export_id) DO UPDATE SET
                    path=excluded.path,status=excluded.status,sha256=excluded.sha256,metadata_json=excluded.metadata_json
                """,
                (
                    str(payload["export_id"]), str(payload.get("export_kind") or "file"),
                    str(payload.get("source_artifact_id") or "") or None,
                    str(payload.get("path") or ""), str(payload.get("status") or "generated"),
                    str(payload.get("sha256") or ""), _json(payload.get("metadata") or {}),
                    str(payload.get("created_at") or utc_now()),
                ),
            )

    def list_exports(self, *, export_kind: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            if export_kind:
                rows = db.execute(
                    "SELECT * FROM exports WHERE export_kind=? ORDER BY created_at DESC",
                    (export_kind,),
                ).fetchall()
            else:
                rows = db.execute("SELECT * FROM exports ORDER BY created_at DESC").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.get("metadata_json") or "{}")
            except (json.JSONDecodeError, TypeError, ValueError):
                item["metadata"] = {}
            result.append(item)
        return result

    def append_audit(self, event_type: str, *, object_type: str = "", object_id: str = "", payload: Any = None) -> None:
        if self.read_only:
            return
        with self.transaction() as db:
            db.execute(
                "INSERT INTO audit_log(event_type,object_type,object_id,payload_json,created_at) VALUES(?,?,?,?,?)",
                (str(event_type), str(object_type), str(object_id), _json(payload or {}), utc_now()),
            )

    def checkpoint(self, *, truncate: bool = True) -> None:
        if self.read_only or not self.path.exists():
            return
        with self._connect() as db:
            mode = "TRUNCATE" if truncate else "PASSIVE"
            db.execute(f"PRAGMA wal_checkpoint({mode})").fetchall()

    def integrity_check(self) -> tuple[bool, str]:
        with self._connect() as db:
            row = db.execute("PRAGMA integrity_check").fetchone()
        message = str(row[0] if row else "unknown")
        return message.lower() == "ok", message


__all__ = ["CATALOG_SCHEMA_VERSION", "ProjectCatalog"]
