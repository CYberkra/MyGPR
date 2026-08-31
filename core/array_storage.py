#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable NumPy artifact writes without loading complete arrays into bytes."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.storage_primitives import fsync_directory


def _atomic_numpy_write(path: str | Path, writer) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
        fsync_directory(target.parent)
    finally:
        temp_path.unlink(missing_ok=True)
    return target


def atomic_save_npy(path: str | Path, array: Any, *, allow_pickle: bool = False) -> Path:
    return _atomic_numpy_write(
        path,
        lambda stream: np.save(stream, np.asarray(array), allow_pickle=allow_pickle),
    )


def atomic_save_npz_compressed(path: str | Path, arrays: Mapping[str, Any]) -> Path:
    normalized = {str(key): np.asarray(value) for key, value in arrays.items()}
    return _atomic_numpy_write(
        path,
        lambda stream: np.savez_compressed(stream, **normalized),
    )


__all__ = ["atomic_save_npy", "atomic_save_npz_compressed"]
