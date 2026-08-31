#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Slice-on-demand proxy for a dataset inside an HDF5 file.

The proxy deliberately opens the file for each read.  Interactive viewport reads
therefore never keep a long-lived HDF5 handle and remain safe when project files
are copied, backed up, or reopened by another process in read-only mode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


class HDF5ArrayProxy(NDArrayOperatorsMixin):
    def __init__(self, file_path: str | Path, dataset_path: str) -> None:
        self.file_path = Path(file_path).resolve()
        self.dataset_path = str(dataset_path)
        with h5py.File(self.file_path, "r", libver="latest", swmr=True) as handle:
            dataset = handle[self.dataset_path]
            self.shape = tuple(int(v) for v in dataset.shape)
            self.dtype = np.dtype(dataset.dtype)
            self.ndim = int(dataset.ndim)
            self.chunks = tuple(int(v) for v in dataset.chunks) if dataset.chunks else None
        self.size = int(np.prod(self.shape, dtype=np.int64))

    def __getitem__(self, selection: Any) -> np.ndarray:
        with h5py.File(self.file_path, "r", libver="latest", swmr=True) as handle:
            return np.asarray(handle[self.dataset_path][selection])

    def iter_blocks(
        self,
        *,
        block_rows: int | None = None,
        sample_start: int = 0,
        sample_end: int | None = None,
        trace_start: int = 0,
        trace_end: int | None = None,
    ):
        """Yield bounded row blocks without materialising the full dataset."""
        row_start = max(0, int(sample_start))
        row_end = self.shape[0] if sample_end is None else min(self.shape[0], int(sample_end))
        col_start = max(0, int(trace_start))
        col_end = self.shape[1] if trace_end is None else min(self.shape[1], int(trace_end))
        if row_end < row_start or col_end < col_start:
            raise ValueError("invalid HDF5 block selection")
        rows_per_block = int(block_rows or (self.chunks[0] if self.chunks else 1024))
        rows_per_block = max(1, rows_per_block)
        with h5py.File(self.file_path, "r", libver="latest", swmr=True) as handle:
            dataset = handle[self.dataset_path]
            for start in range(row_start, row_end, rows_per_block):
                end = min(start + rows_per_block, row_end)
                yield start, end, np.asarray(dataset[start:end, col_start:col_end])

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # NumPy 2 compatible
        with h5py.File(self.file_path, "r", libver="latest", swmr=True) as handle:
            array = np.asarray(handle[self.dataset_path][...], dtype=dtype)
        if copy:
            return array.copy()
        return array

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Compatibility fallback for legacy whole-array NumPy operations.

        Viewport rendering and the new storage layer remain slice-based.  Older
        processing code that directly performs ``proxy * scalar`` is still
        supported by explicitly materialising that operation until the
        algorithm is migrated to the chunk executor.
        """
        converted = [np.asarray(value) if isinstance(value, (HDF5ArrayProxy, HDF5TransposeProxy)) else value for value in inputs]
        out = kwargs.get("out")
        if out is not None:
            kwargs["out"] = tuple(
                np.asarray(value) if isinstance(value, (HDF5ArrayProxy, HDF5TransposeProxy)) else value
                for value in out
            )
        return getattr(ufunc, method)(*converted, **kwargs)

    @property
    def T(self):
        return HDF5TransposeProxy(self)

    def __repr__(self) -> str:
        return (
            f"HDF5ArrayProxy(file_path={str(self.file_path)!r}, "
            f"dataset_path={self.dataset_path!r}, shape={self.shape}, dtype={self.dtype})"
        )


class HDF5TransposeProxy(NDArrayOperatorsMixin):
    """Lazy transposed view used by orientation repair without full loading."""

    def __init__(self, source: HDF5ArrayProxy) -> None:
        self.source = source
        self.shape = (source.shape[1], source.shape[0])
        self.dtype = source.dtype
        self.ndim = 2
        self.size = source.size
        self.chunks = tuple(reversed(source.chunks)) if source.chunks else None

    def __getitem__(self, selection: Any) -> np.ndarray:
        if not isinstance(selection, tuple):
            selection = (selection, slice(None))
        if len(selection) != 2:
            return np.asarray(self)[selection]
        rows, cols = selection
        return np.asarray(self.source[cols, rows]).T

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        array = np.asarray(self.source, dtype=dtype).T
        if copy:
            return array.copy()
        return array

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        converted = [np.asarray(value) if isinstance(value, (HDF5ArrayProxy, HDF5TransposeProxy)) else value for value in inputs]
        out = kwargs.get("out")
        if out is not None:
            kwargs["out"] = tuple(
                np.asarray(value) if isinstance(value, (HDF5ArrayProxy, HDF5TransposeProxy)) else value
                for value in out
            )
        return getattr(ufunc, method)(*converted, **kwargs)

    @property
    def T(self):
        return self.source


__all__ = ["HDF5ArrayProxy", "HDF5TransposeProxy"]
