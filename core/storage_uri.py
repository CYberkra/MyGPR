#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable URIs for datasets stored inside project-local HDF5 containers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

H5_URI_PREFIX = "h5://"
H5_DATASET_SEPARATOR = "::"


@dataclass(frozen=True)
class H5DatasetURI:
    file_path: str
    dataset_path: str

    def __post_init__(self) -> None:
        if not self.dataset_path.startswith("/"):
            object.__setattr__(self, "dataset_path", "/" + self.dataset_path)

    def to_string(self) -> str:
        return f"{H5_URI_PREFIX}{self.file_path}{H5_DATASET_SEPARATOR}{self.dataset_path}"


def make_h5_uri(file_path: str | Path, dataset_path: str) -> str:
    value = Path(file_path).as_posix()
    return H5DatasetURI(value, dataset_path).to_string()


def is_h5_uri(value: str | Path | None) -> bool:
    return str(value or "").startswith(H5_URI_PREFIX)


def parse_h5_uri(value: str | Path) -> H5DatasetURI:
    text = str(value)
    if not is_h5_uri(text) or H5_DATASET_SEPARATOR not in text:
        raise ValueError(f"Invalid HDF5 dataset URI: {value!r}")
    body = text[len(H5_URI_PREFIX):]
    file_path, dataset_path = body.split(H5_DATASET_SEPARATOR, 1)
    if not file_path or not dataset_path:
        raise ValueError(f"Invalid HDF5 dataset URI: {value!r}")
    return H5DatasetURI(file_path=file_path, dataset_path=dataset_path)


def resolve_h5_uri(project_root: str | Path, value: str | Path) -> tuple[Path, str]:
    uri = parse_h5_uri(value)
    path = Path(uri.file_path)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve(), uri.dataset_path


__all__ = [
    "H5DatasetURI",
    "H5_DATASET_SEPARATOR",
    "H5_URI_PREFIX",
    "is_h5_uri",
    "make_h5_uri",
    "parse_h5_uri",
    "resolve_h5_uri",
]
