from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from core.gpr_format_registry import get_format_spec, supported_file_dialog_filter
from core.gpr_io import auto_load_data
from core.gpr_vendor_readers import GprReaderFormat


def test_format_registry_covers_common_gpr_inputs():
    assert get_format_spec("line.DZT").key == "gssi_dzt"
    assert get_format_spec("line.dt1").key == "sensors_software_dt1"
    assert get_format_spec("line.rd3").key == "mala_rd"
    assert get_format_spec("line.iprb").key == "impulseradar_iprb"
    assert get_format_spec("line.segy").key == "segy"
    assert "*.dzt" in supported_file_dialog_filter().lower()
    assert "*.rd3" in supported_file_dialog_filter().lower()


def test_auto_load_numpy_npy(tmp_path: Path):
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "profile.npy"
    np.save(path, arr)
    result = auto_load_data(str(path))
    np.testing.assert_allclose(result["data"], arr)
    assert result["header_info"]["a_scan_length"] == 3
    assert result["header_info"]["num_traces"] == 4


def test_auto_load_mala_rd3_with_rad(tmp_path: Path):
    samples, traces = 4, 3
    (tmp_path / "line.rad").write_text(
        "SAMPLES: 4\nTIME WINDOW: 80\nDISTANCE INTERVAL: 0.05\nLAST TRACE: 3\n",
        encoding="utf-8",
    )
    raw = np.arange(samples * traces, dtype=np.int16)
    raw.tofile(tmp_path / "line.rd3")
    result = auto_load_data(str(tmp_path / "line.rd3"))
    assert result["data"].shape == (samples, traces)
    assert result["header_info"]["source"] == GprReaderFormat.MALA_RD
    assert result["header_info"]["total_time_ns"] == 80


def test_auto_load_impulseradar_iprb_with_iprh(tmp_path: Path):
    samples, traces = 5, 2
    (tmp_path / "profile.iprh").write_text(
        "HEADER VERSION: 20\nDATA VERSION: 16\nSAMPLES: 5\nFREQUENCY: 10000\n",
        encoding="utf-8",
    )
    np.arange(samples * traces, dtype=np.int16).tofile(tmp_path / "profile.iprb")
    result = auto_load_data(str(tmp_path / "profile.iprb"))
    assert result["data"].shape == (samples, traces)
    assert result["header_info"]["source"] == GprReaderFormat.IMPULSERADAR_IPRB
    assert result["header_info"]["data_version"] == 16


def test_auto_load_fixed_segy_int16(tmp_path: Path):
    samples, traces = 4, 2
    text_header = b"C" * 3200
    bin_header = bytearray(400)
    bin_header[16:18] = struct.pack(">H", 1000)  # sample interval us
    bin_header[20:22] = struct.pack(">H", samples)
    bin_header[24:26] = struct.pack(">H", 3)  # int16
    body = bytearray()
    values = np.arange(samples * traces, dtype=">i2")
    for tr in range(traces):
        trace_header = bytearray(240)
        trace_header[114:116] = struct.pack(">H", samples)
        body += trace_header
        body += values[tr * samples : (tr + 1) * samples].tobytes()
    path = tmp_path / "line.sgy"
    path.write_bytes(text_header + bytes(bin_header) + bytes(body))
    result = auto_load_data(str(path))
    assert result["data"].shape == (samples, traces)
    assert result["header_info"]["source"] == GprReaderFormat.SEGY_FIXED


def test_recognized_but_not_native_format_fails_clearly(tmp_path: Path):
    # 当前注册表所有格式均可解码；该消息契约保留给未来新登记的 recognized-only 格式
    from core.gpr_vendor_readers import unsupported_known_format_message

    message = unsupported_known_format_message(str(tmp_path / "line.xxx"), "示例格式", "补充说明")
    assert "已被识别为常见 GPR 数据格式" in message
    assert "补充说明" in message
