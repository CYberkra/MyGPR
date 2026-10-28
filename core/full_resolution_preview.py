"""Lossless, tiled reads into a temporary file-backed B-scan display matrix."""
from pathlib import Path
import os
import shutil
import tempfile
import weakref

import numpy as np


def _dispose(mapping, path):
    mapping.close()
    Path(path).unlink(missing_ok=True)


def read_full_resolution_window(shape, reader, *, cancel_requested=None,
                                max_block_bytes=8 * 1024 * 1024):
    rows, columns = map(int, shape)
    if min(rows, columns) <= 0 or max_block_bytes < 4:
        raise ValueError('无有效 B-scan 数据或读取块大小')
    fd, filename = tempfile.mkstemp(prefix='mygpr-full-bscan-', suffix='.npy')
    os.close(fd)
    path = Path(filename)
    finalizer = None
    completed = False
    try:
        required = rows * columns * 4 + 1024 * 1024
        if shutil.disk_usage(path.parent).free < required:
            raise OSError('全分辨率 B-scan 缓存磁盘空间不足；未执行降采样')
        matrix = np.lib.format.open_memmap(path, mode='w+', dtype='float32',
                                          shape=(rows, columns))
        # ndarray views retain this memmap. Close its OS handle before unlinking on Windows.
        finalizer = weakref.finalize(matrix, _dispose, matrix._mmap, path)
        block_columns = min(columns, 2048, max_block_bytes // 4)
        block_rows = max(1, min(rows, max_block_bytes // (4 * block_columns)))
        for row in range(0, rows, block_rows):
            end_row = min(rows, row + block_rows)
            for column in range(0, columns, block_columns):
                if cancel_requested is not None and cancel_requested():
                    raise RuntimeError('全分辨率 B-scan 读取已取消')
                end_column = min(columns, column + block_columns)
                data, sample_index, trace_index = reader(
                    sample_start=row, sample_end=end_row,
                    trace_start=column, trace_end=end_column,
                    max_samples=end_row-row, max_traces=end_column-column)
                if (data.shape != (end_row-row, end_column-column)
                        or not np.array_equal(sample_index, np.arange(row, end_row))
                        or not np.array_equal(trace_index, np.arange(column, end_column))):
                    raise ValueError('B-scan 读取未返回完整分辨率数据；拒绝显示抽样结果')
                matrix[row:end_row, column:end_column] = data
        matrix.flush()
        completed = True
        return matrix, np.arange(rows, dtype=np.int64), np.arange(columns, dtype=np.int64)
    finally:
        if not completed:
            if finalizer is not None:
                finalizer()
            else:
                path.unlink(missing_ok=True)
