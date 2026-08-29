# -*- coding: utf-8 -*-
"""本地 DEM XYZ 格网解析（ui.widgets.local_dem）测试。"""
import numpy as np
import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py qapp 设计

from ui.widgets.local_dem import dem_covers_bbox, load_xyz_grid  # noqa: E402


def _write(tmp_path, text, name='dem.xyz'):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return str(path)


def test_comma_grid_with_header(tmp_path):
    path = _write(tmp_path, (
        'Global Mapper export\n'
        'lon,lat,elev\n'
        '104.0, 31.0, 450.5\n'
        '104.1, 31.0, 451.0\n'
        '104.0, 31.1, 452.0\n'
        '104.1, 31.1, 452.5\n'))
    dem = load_xyz_grid(path)
    assert dem['lons'].tolist() == [104.0, 104.1]
    assert dem['lats'].tolist() == [31.0, 31.1]
    assert dem['elev'].shape == (2, 2)
    assert dem['elev'][0, 0] == pytest.approx(450.5)
    assert dem['elev'][1, 1] == pytest.approx(452.5)


def test_whitespace_and_semicolon(tmp_path):
    path = _write(tmp_path, (
        '104.0 31.0 100\n'
        '104.1\t31.0\t101\n'
        '104.0;31.1;102\n'
        '104.1 31.1 103\n'))
    dem = load_xyz_grid(path)
    assert dem['elev'].tolist() == [[100.0, 101.0], [102.0, 103.0]]


def test_unordered_rows_sorted_axes(tmp_path):
    path = _write(tmp_path, (
        '104.1 31.1 11\n'
        '104.0 31.1 10\n'
        '104.1 31.0 1\n'
        '104.0 31.0 0\n'))
    dem = load_xyz_grid(path)
    assert dem['lons'].tolist() == [104.0, 104.1]
    assert dem['lats'].tolist() == [31.0, 31.1]
    assert dem['elev'].tolist() == [[0.0, 1.0], [10.0, 11.0]]


def test_scattered_points_rejected(tmp_path):
    # 100 个随机散点不构成规则格网 → 拒绝
    rng = np.random.default_rng(0)
    lines = [f'{x} {y} 500' for x, y in rng.random((100, 2))]
    path = _write(tmp_path, '\n'.join(lines))
    with pytest.raises(ValueError, match='格网'):
        load_xyz_grid(path)


def test_too_small_grid_rejected(tmp_path):
    path = _write(tmp_path, '104.0 31.0 100\n')
    with pytest.raises(ValueError):
        load_xyz_grid(path)


def test_no_valid_data_rejected(tmp_path):
    path = _write(tmp_path, 'header only\nnothing numeric\n')
    with pytest.raises(ValueError, match='数据行'):
        load_xyz_grid(path)


def test_dem_covers_bbox():
    dem = {'lons': np.array([104.0, 104.1]), 'lats': np.array([31.0, 31.1])}
    # 完全在内部
    assert dem_covers_bbox(dem, (104.01, 31.01, 104.09, 31.09))
    # 边界贴齐也算覆盖
    assert dem_covers_bbox(dem, (104.0, 31.0, 104.1, 31.1))
    # 各方向超出
    assert not dem_covers_bbox(dem, (103.9, 31.01, 104.09, 31.09))
    assert not dem_covers_bbox(dem, (104.01, 31.01, 104.2, 31.09))
    assert not dem_covers_bbox(dem, (104.01, 30.9, 104.09, 31.09))
    assert not dem_covers_bbox(dem, (104.01, 31.01, 104.09, 31.2))
