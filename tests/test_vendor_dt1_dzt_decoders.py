#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DT1/HD 与 DZT 解码器验收测试（任务 F / P0-3）。

分层与营山 Golden 相同：
1. 仓库内裁剪文件（1024B 头 + 32 道真实数据）必须与 npz 参考子集逐位一致；
2. 头字段与 GPRPy 样例的 .HD 文本头 / DZT 二进制头逐项断言；
3. ``MYGPR_VENDOR_SAMPLE_DATA`` 指向外部全文件时，加跑全文件 SHA256 与
   全矩阵统计回归（CI 无外部资产时自动跳过）。

二进制布局事实与 SHA256 见 ``tests/fixtures/vendor_formats_v1/vendor_formats_manifest.json``。
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from core.gpr_format_registry import get_format_spec
from core.gpr_io import auto_load_data
from core.gpr_vendor_readers import (
    GPRFormatReadError,
    read_gssi_dzt,
    read_sensors_software_dt1,
)

FIXTURES = Path(__file__).resolve().parents[0] / "fixtures" / "vendor_formats_v1"
MANIFEST = json.loads((FIXTURES / "vendor_formats_manifest.json").read_text(encoding="utf-8"))
SUBSET_NPZ = FIXTURES / "vendor_trace_subset_v1.npz"


def _subset(key: str) -> np.ndarray:
    with np.load(SUBSET_NPZ, allow_pickle=False) as archive:
        return np.asarray(archive[key], dtype=np.float32)


# ---------------------------------------------------------------------------
# 1) 仓库内裁剪文件：真实二进制逐位回归（无外部依赖）
# ---------------------------------------------------------------------------


def test_dzt_clip_decodes_bitwise_to_reference_subset() -> None:
    result = read_gssi_dzt(FIXTURES / "gssi_FILE____032_head32.DZT")
    expected = _subset("gssi_dzt_matrix")
    assert result["format"].value == "gssi_dzt"
    assert result["data"].shape == expected.shape
    assert result["data"].dtype == np.float32
    assert np.array_equal(result["data"], expected)


def test_dt1_clip_decodes_bitwise_to_reference_subset() -> None:
    result = read_sensors_software_dt1(FIXTURES / "sns_XLINE00_traces32.DT1")
    expected = _subset("sns_dt1_matrix")
    assert result["format"].value == "sensors_software_dt1"
    assert result["data"].shape == expected.shape
    assert np.array_equal(result["data"], expected)


def test_dzt_header_fields_match_manifest() -> None:
    header_spec = MANIFEST["formats"]["gssi_dzt"]["header"]
    info = read_gssi_dzt(FIXTURES / "gssi_FILE____032_head32.DZT")["header_info"]
    assert info["a_scan_length"] == header_spec["rh_nsamp"]
    assert info["bits_per_sample"] == header_spec["rh_bits"]
    assert info["total_time_ns"] == pytest.approx(header_spec["rhf_range_ns"])
    assert info["trace_interval_m"] == pytest.approx(header_spec["rhf_spm"])
    assert info["channels"] == header_spec["rh_nchan"]


def test_dt1_header_fields_match_hd_text_and_unit_conversion() -> None:
    fields = MANIFEST["formats"]["sensors_software_dt1"]["header_text_fields"]
    info = read_sensors_software_dt1(FIXTURES / "sns_XLINE00_traces32.DT1")["header_info"]
    assert info["total_time_ns"] == pytest.approx(float(fields["TOTAL TIME WINDOW"]))
    assert info["nominal_frequency_mhz"] == pytest.approx(float(fields["NOMINAL FREQUENCY"]))
    assert info["stacks"] == int(float(fields["NUMBER OF STACKS"]))
    # HD 声明 ft 单位：2.0 ft 步距必须换算为 0.6096 m
    assert fields["POSITION UNITS"] == "ft"
    assert info["trace_interval_m"] == pytest.approx(0.6096)
    assert info["a_scan_length"] == int(float(fields["NUMBER OF PTS/TRC"]))
    assert info["num_traces"] == 32  # 裁剪文件只含 32 道


# ---------------------------------------------------------------------------
# 2) auto_load_data 分发 + 注册表 support 升级
# ---------------------------------------------------------------------------


def test_auto_load_dispatches_dzt_and_dt1() -> None:
    dzt = auto_load_data(FIXTURES / "gssi_FILE____032_head32.DZT")
    assert dzt["header_info"]["source"].value == "gssi_dzt"
    dt1 = auto_load_data(FIXTURES / "sns_XLINE00_traces32.DT1")
    assert dt1["header_info"]["source"].value == "sensors_software_dt1"


def test_registry_marks_dt1_and_dzt_as_native_subset() -> None:
    assert get_format_spec("line.dt1").support == "native-subset"
    assert get_format_spec("line.hd").support == "native-subset"
    assert get_format_spec("line.dzt").support == "native-subset"


# ---------------------------------------------------------------------------
# 3) 错误路径（合成最小文件）
# ---------------------------------------------------------------------------


def test_dt1_requires_hd_sidecar(tmp_path: Path) -> None:
    data = tmp_path / "line.DT1"
    data.write_bytes(struct.pack("<32f", *([1.0, 0.0, 4.0] + [0.0] * 29)))
    with pytest.raises(GPRFormatReadError, match="需要同名 .hd"):
        read_sensors_software_dt1(data)


def test_dt1_rejects_size_mismatch(tmp_path: Path) -> None:
    # 声明每道 4 样本（128B 道头 + 8B 数据 = 136B），文件多出 1 字节 → 不整除必须报错
    head = [1.0, 0.0, 4.0] + [0.0] * 29
    data = tmp_path / "line.DT1"
    data.write_bytes(struct.pack("<32f", *head) + b"\x00" * 8 + b"\x00")
    (tmp_path / "line.HD").write_text("NUMBER OF TRACES   = 2\n", encoding="utf-8")
    with pytest.raises(GPRFormatReadError, match="不符"):
        read_sensors_software_dt1(data)


def test_dzt_rejects_unknown_bits(tmp_path: Path) -> None:
    head = bytearray(1024)
    struct.pack_into("<3h", head, 2, 1024, 8, 24)  # rh_data=1块, samples=8, bits=24
    file = tmp_path / "line.DZT"
    file.write_bytes(bytes(head) + b"\x00" * 64)
    with pytest.raises(GPRFormatReadError, match="无效"):
        read_gssi_dzt(file)


def test_dzt_rejects_truncated_file(tmp_path: Path) -> None:
    file = tmp_path / "line.DZT"
    file.write_bytes(b"\x00" * 512)
    with pytest.raises(GPRFormatReadError, match="过短"):
        read_gssi_dzt(file)


# ---------------------------------------------------------------------------
# 4) 外部全文件回归（CI 无资产时跳过）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("format_key", ["gssi_dzt", "sensors_software_dt1"])
def test_full_vendor_files_match_frozen_hashes(format_key: str) -> None:
    root = os.environ.get("MYGPR_VENDOR_SAMPLE_DATA")
    if not root:
        pytest.skip("set MYGPR_VENDOR_SAMPLE_DATA to the vendor sample asset root")
    spec = MANIFEST["formats"][format_key]
    if format_key == "gssi_dzt":
        path = Path(root) / "gssi_dzt" / spec["file"]
    else:
        path = Path(root) / "sns_dt1" / spec["files"]["data"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = spec["sha256"]
    if isinstance(expected, dict):  # DT1 按 data/header/gps 分键
        expected = expected["data"]
    assert digest == expected, f"{format_key}: 全文件哈希与 manifest 不符"


def test_full_dt1_matrix_statistics_match_manifest() -> None:
    root = os.environ.get("MYGPR_VENDOR_SAMPLE_DATA")
    if not root:
        pytest.skip("set MYGPR_VENDOR_SAMPLE_DATA to the vendor sample asset root")
    spec = MANIFEST["formats"]["sensors_software_dt1"]
    result = read_sensors_software_dt1(Path(root) / "sns_dt1" / spec["files"]["data"])
    shape = spec["matrix"]["shape"]
    assert list(result["data"].shape) == shape
    subset = spec["subset"]
    actual = result["data"][:, : subset["trace_count"]]
    x = np.asarray(actual, dtype=np.float64)
    stats = subset["statistics"]
    assert float(x.min()) == pytest.approx(stats["min"], abs=1e-3)
    assert float(x.max()) == pytest.approx(stats["max"], abs=1e-3)
    assert float(np.sqrt((x**2).mean())) == pytest.approx(stats["rms"], rel=1e-6)
