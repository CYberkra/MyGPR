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
    read_geotech_oko_gpr2,
    read_gssi_dzt,
    read_mala_rd,
    read_envi_bsq,
    read_segy_fixed,
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


def test_dt1_sidecar_lookup_is_case_insensitive(tmp_path: Path) -> None:
    """扩展名大小写与约定相反时也能配对（Linux 文件系统区分大小写）。"""
    samples = 4
    head = [1.0, 0.0, float(samples)] + [0.0] * 29
    payload = b"".join(struct.pack("<h", tr * 10 + k) for tr in range(3) for k in range(samples))
    data = tmp_path / "line.dt1"
    data.write_bytes(b"".join(struct.pack("<32f", *head) + payload[tr * samples * 2 : (tr + 1) * samples * 2] for tr in range(3)))
    (tmp_path / "line.HD").write_text(
        "NUMBER OF TRACES   = 3\nNUMBER OF PTS/TRC  = 4\nPOSITION UNITS     = m\n",
        encoding="utf-8",
    )
    result = read_sensors_software_dt1(data)
    assert result["data"].shape == (samples, 3)


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
# 4) RD3 / ENVI / SEG-Y 样例（真实数据派生，仓库内）
# ---------------------------------------------------------------------------


def test_mala_rd3_derived_sample_roundtrip() -> None:
    spec = MANIFEST["formats"]["mala_rd3_derived"]
    result = read_mala_rd(FIXTURES / spec["files"]["data"])
    expected = _subset("gssi_dzt_matrix")
    assert result["data"].shape == tuple(spec["matrix"]["shape"])
    assert np.array_equal(result["data"], expected.astype(np.float32))
    assert result["header_info"]["total_time_ns"] == pytest.approx(48.0)


def test_envi_derived_sample_roundtrip() -> None:
    spec = MANIFEST["formats"]["envi_bsq_derived"]
    result = read_envi_bsq(FIXTURES / spec["files"]["data"])
    expected = _subset("gssi_dzt_matrix")  # reader returns samples x traces
    assert result["data"].shape == tuple(spec["matrix"]["shape"])
    assert np.array_equal(result["data"], expected)


def test_segy_clip_decodes_real_f3_subset() -> None:
    spec = MANIFEST["formats"]["segy_real"]
    result = read_segy_fixed(FIXTURES / spec["file"])
    assert result["data"].shape == tuple(spec["matrix"]["shape"])
    assert result["header_info"]["sample_format_code"] == spec["header"]["sample_format_code"]
    assert np.isfinite(result["data"]).all()


# ---------------------------------------------------------------------------
# 5) OKO GPR2（合成样例，RGPR readGPR2 布局）
# ---------------------------------------------------------------------------


def test_oko_synthetic_roundtrip_and_header_fields() -> None:
    spec = MANIFEST["formats"]["oko_gpr2_synthetic"]
    result = read_geotech_oko_gpr2(FIXTURES / spec["file"])
    expected = _subset("oko_gpr2_matrix")
    assert result["format"].value == "geotech_oko_gpr2"
    assert result["data"].shape == tuple(spec["matrix"]["shape"])
    assert np.array_equal(result["data"], expected)
    info = result["header_info"]
    assert info["total_time_ns"] == pytest.approx(120.0)
    assert info["trace_interval_m"] == pytest.approx(0.025)
    assert info["antenna_name"] == "АБ-400"
    assert info["trace_positions"][-1] == pytest.approx(23 * 25)


def test_oko_rejects_bad_magic(tmp_path: Path) -> None:
    file = tmp_path / "line.GPR2"
    file.write_bytes(b"\x00" * 512)
    with pytest.raises(GPRFormatReadError, match="魔数"):
        read_geotech_oko_gpr2(file)


def test_registry_marks_oko_as_native_subset() -> None:
    assert get_format_spec("line.gpr2").support == "native-subset"
    assert get_format_spec("line.gpr").support == "native-subset"


# ---------------------------------------------------------------------------
# 6) 外部全文件回归（CI 无资产时跳过）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("format_key", ["gssi_dzt", "sensors_software_dt1", "segy_real"])
def test_full_vendor_files_match_frozen_hashes(format_key: str) -> None:
    root = os.environ.get("MYGPR_VENDOR_SAMPLE_DATA")
    if not root:
        pytest.skip("set MYGPR_VENDOR_SAMPLE_DATA to the vendor sample asset root")
    spec = MANIFEST["formats"][format_key]
    if format_key == "gssi_dzt":
        path = Path(root) / "gssi_dzt" / spec["file"]
    elif format_key == "sensors_software_dt1":
        path = Path(root) / "sns_dt1" / spec["files"]["data"]
    else:
        path = Path(root) / "segy" / spec["external_full_file"]["name"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == spec["external_full_file"]["sha256"]
        return
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
