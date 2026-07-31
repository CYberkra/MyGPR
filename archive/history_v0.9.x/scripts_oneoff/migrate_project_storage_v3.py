#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line entry for explicit non-destructive Hybrid Store migration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.field_project_store import FieldProjectStore
from core.project_storage_migration import migrate_project_to_hybrid


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate a MyGPR field project to HDF5 + SQLite Hybrid Store v1")
    parser.add_argument("project", type=Path, help="Project root containing project.json")
    args = parser.parse_args()
    with FieldProjectStore.open(args.project, access_mode="write") as store:
        result = migrate_project_to_hybrid(
            store,
            progress_callback=lambda current, total, message: print(f"[{current}/{total}] {message}"),
        )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
