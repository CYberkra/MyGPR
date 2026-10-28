from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Callable

import pytest

from core.project_catalog import ProjectCatalog
from mygpr.interfaces.cli.project_smoke import run_project_smoke


@pytest.fixture
def catalog_connections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    catalog = ProjectCatalog(tmp_path / "catalog.sqlite")
    catalog.initialize(project_id="lifecycle", project_name="Lifecycle")
    connections: list[sqlite3.Connection] = []
    connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        # Retain references so correctness cannot depend on garbage collection.
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    yield catalog, connections
    for connection in connections:
        connection.close()


READ_OPERATIONS: tuple[tuple[str, Callable[[ProjectCatalog], object]], ...] = (
    ("lines", lambda catalog: catalog.list_lines()),
    ("branches", lambda catalog: catalog.list_branches()),
    ("branch_head", lambda catalog: catalog.branch_head("missing")),
    ("artifact", lambda catalog: catalog.get_artifact("missing")),
    ("artifacts", lambda catalog: catalog.list_artifacts()),
    ("exports", lambda catalog: catalog.list_exports()),
    ("integrity", lambda catalog: catalog.integrity_check()),
)


def _assert_closed(connections: list[sqlite3.Connection]) -> None:
    assert connections, "operation did not open the expected catalog connection"
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")


@pytest.mark.parametrize("read_only", [False, True], ids=["writable", "read_only"])
@pytest.mark.parametrize("name,operation", READ_OPERATIONS, ids=[item[0] for item in READ_OPERATIONS])
def test_catalog_read_closes_connection(catalog_connections, read_only, name, operation) -> None:
    catalog, connections = catalog_connections
    selected = ProjectCatalog(catalog.path, read_only=read_only)
    operation(selected)
    _assert_closed(connections)


def test_catalog_checkpoint_closes_connection(catalog_connections) -> None:
    catalog, connections = catalog_connections
    catalog.checkpoint()
    _assert_closed(connections)


def test_catalog_failed_query_closes_connection(catalog_connections) -> None:
    catalog, connections = catalog_connections
    with catalog.transaction() as connection:
        connection.execute("DROP TABLE lines")
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        catalog.list_lines()
    _assert_closed(connections)


@pytest.mark.parametrize("read_only", [False, True], ids=["writable", "read_only"])
def test_catalog_failed_connection_setup_closes_connection(catalog_connections, read_only) -> None:
    catalog, connections = catalog_connections
    catalog.path.write_bytes(b"invalid sqlite database")
    selected = ProjectCatalog(catalog.path, read_only=read_only)
    healthy, message = selected.integrity_check()
    assert not healthy
    assert message
    _assert_closed(connections)


def test_catalog_transaction_commits_and_rolls_back_and_closes(catalog_connections) -> None:
    catalog, connections = catalog_connections
    with catalog.transaction() as connection:
        connection.execute("INSERT INTO catalog_meta VALUES('committed', 'yes')")
    with pytest.raises(RuntimeError, match="cancel write"):
        with catalog.transaction() as connection:
            connection.execute("INSERT INTO catalog_meta VALUES('cancelled', 'no')")
            raise RuntimeError("cancel write")
    with catalog.transaction() as connection:
        rows = connection.execute(
            "SELECT key FROM catalog_meta WHERE key IN ('committed', 'cancelled')"
        ).fetchall()
    assert [row[0] for row in rows] == ["committed"]
    _assert_closed(connections)


def test_project_smoke_shutdown_releases_project_directory(tmp_path: Path) -> None:
    root = tmp_path / "project"
    payload = run_project_smoke(root)
    assert payload["job_status"] == "completed"
    assert payload["integrity_healthy"]
    # On Windows this fails if a SQLite, HDF5 or mmap handle remains open.
    shutil.rmtree(root)
    assert not root.exists()
