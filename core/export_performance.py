# -*- coding: utf-8 -*-
"""Low-risk export performance helpers.

These helpers reduce boilerplate around multi-file report sidecar writes and
make export operations easier to instrument.  They do not change artifact
schemas or numerical processing results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_json_sidecars(items: Iterable[tuple[str | Path, Any]], *, json_safe=None) -> int:
    """Write a sequence of JSON sidecar files and return the file count."""

    count = 0
    for path, payload in items:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        value = json_safe(payload) if json_safe is not None else payload
        with target.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        count += 1
    return count


def write_text_sidecars(items: Iterable[tuple[str | Path, str]]) -> int:
    """Write a sequence of text sidecar files and return the file count."""

    count = 0
    for path, text in items:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text or "", encoding="utf-8")
        count += 1
    return count
