#!/usr/bin/env python3
"""Backend-only pytest configuration and isolated runtime roots."""
from __future__ import annotations

import enum
import os
from pathlib import Path
import shutil
import sys
import tempfile

# Python 3.10 compatibility: provide a StrEnum that behaves like the 3.11+ builtin.
# core/domain modules import `from enum import StrEnum`; without this patch the
# fallback str() representation is the enum name, breaking severity comparisons.
try:
    from enum import StrEnum  # noqa: F401
except ImportError:
    class StrEnum(str, enum.Enum):
        def __str__(self) -> str:
            return self.value

    enum.StrEnum = StrEnum

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RUNTIME_ROOT: Path | None = None


def pytest_configure(config) -> None:
    marker_docs = {
        "industrial": "industrial acceptance/regression evidence",
        "acceptance": "end-to-end user workflow acceptance",
        "reliability": "fault recovery and long-run reliability",
        "performance": "resource and throughput verification",
        "scientific_validation": "field-data and scientific-result validation",
        "property": "property/fuzz-oriented input contract tests",
        "static_contract": "source/configuration architecture contract",
        "requirement(ids)": "requirement identifiers covered by a test",
        "risk(ids)": "risk identifiers mitigated by a test",
        "level(name)": "test level",
        "external_data": "requires external immutable acceptance data",
        "hardware": "requires physical hardware",
        "windows": "requires Windows target environment",
        "large_data": "large-data acceptance test",
        "release_only": "release evidence lane only",
    }
    for marker, description in marker_docs.items():
        config.addinivalue_line("markers", f"{marker}: {description}")
    global _RUNTIME_ROOT
    if not os.environ.get("MYGPR_RUNTIME_ROOT"):
        _RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="mygpr-backend-test-runtime-"))
        os.environ["MYGPR_RUNTIME_ROOT"] = str(_RUNTIME_ROOT)
    os.environ.setdefault("MYGPR_LOG_DIR", str(Path(os.environ["MYGPR_RUNTIME_ROOT"]) / "logs"))


def pytest_unconfigure(config) -> None:
    global _RUNTIME_ROOT
    if _RUNTIME_ROOT is not None:
        runtime_text = str(_RUNTIME_ROOT)
        shutil.rmtree(_RUNTIME_ROOT, ignore_errors=True)
        if os.environ.get("MYGPR_RUNTIME_ROOT") == runtime_text:
            os.environ.pop("MYGPR_RUNTIME_ROOT", None)
        log_root = str(_RUNTIME_ROOT / "logs")
        if os.environ.get("MYGPR_LOG_DIR") == log_root:
            os.environ.pop("MYGPR_LOG_DIR", None)
        _RUNTIME_ROOT = None
