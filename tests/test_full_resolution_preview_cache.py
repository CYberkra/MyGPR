"""Full-resolution display reads are lossless, bounded and own temporary cache lifetime."""
import gc
from pathlib import Path

import numpy as np
import pytest

from core.full_resolution_preview import read_full_resolution_window


def reader_for(data, calls):
    def read(**kw):
        calls.append(kw)
        rs = slice(kw['sample_start'], kw['sample_end'])
        cs = slice(kw['trace_start'], kw['trace_end'])
        return data[rs, cs], np.arange(rs.start, rs.stop), np.arange(cs.start, cs.stop)
    return read


def test_tiled_reads_keep_all_values_and_cache_survives_array_views():
    data = np.arange(7*11, dtype='float32').reshape(7, 11)
    calls = []
    matrix, rows, columns = read_full_resolution_window(
        data.shape, reader_for(data, calls), max_block_bytes=64)
    path = Path(matrix.filename)
    array = np.asarray(matrix)
    np.testing.assert_array_equal(array, data)
    np.testing.assert_array_equal(rows, np.arange(7))
    np.testing.assert_array_equal(columns, np.arange(11))
    assert len(calls) > 1 and all(k['max_samples']*k['max_traces']*4 <= 64 for k in calls)
    del matrix
    gc.collect()
    assert path.is_file()
    np.testing.assert_array_equal(array, data)
    del array
    gc.collect()
    assert not path.exists()


@pytest.mark.parametrize('cancelled', [False, True])
def test_failed_or_cancelled_read_removes_temporary_cache(tmp_path, monkeypatch, cancelled):
    import core.full_resolution_preview as module

    monkeypatch.setattr(module.tempfile, 'tempdir', str(tmp_path))

    def broken(**kw):
        raise OSError('controlled read failure')

    with pytest.raises((RuntimeError, OSError), match='取消|controlled'):
        read_full_resolution_window((5, 5), broken, cancel_requested=lambda: cancelled)
    assert list(tmp_path.iterdir()) == []


def test_refuses_decimated_blocks_instead_of_silently_degrading(tmp_path, monkeypatch):
    import core.full_resolution_preview as module

    monkeypatch.setattr(module.tempfile, 'tempdir', str(tmp_path))
    with pytest.raises(ValueError, match='完整分辨率'):
        read_full_resolution_window((5, 5), lambda **kw: (
            np.zeros((5, 3), dtype='float32'), np.arange(5), np.arange(0, 5, 2)))
    assert list(tmp_path.iterdir()) == []


def test_insufficient_disk_space_never_starts_a_read(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import core.full_resolution_preview as module

    monkeypatch.setattr(module.tempfile, 'tempdir', str(tmp_path))
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1))
    calls = []
    with pytest.raises(OSError, match='磁盘空间不足'):
        read_full_resolution_window((5, 5), lambda **kw: calls.append(kw))
    assert calls == [] and list(tmp_path.iterdir()) == []


def test_windows_memory_probe_works_without_psutil(monkeypatch):
    import os
    import sys
    from mygpr.infrastructure.system.resource_policy import available_memory_bytes

    if os.name != 'nt':
        pytest.skip('Windows host memory probe')
    monkeypatch.setitem(sys.modules, 'psutil', None)
    assert available_memory_bytes() > 0
