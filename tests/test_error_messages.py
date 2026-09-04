# -*- coding: utf-8 -*-
"""错误映射测试：friendly_error_message 对 MyGPRError 子类输出可操作建议。"""
from __future__ import annotations


import pytest

pytest.importorskip("PyQt6")  # 后端 CI（无 Qt）自动跳过，见 tests/conftest.py qapp 设计

from ui.controllers.backend_controller import (  # noqa: E402
    PROJECT_BUSY_MESSAGE,
    friendly_error_message,
)


def _mgpr_error(code: str, message: str, hint: str = "") -> BaseException:
    """模拟 MyGPRError 子类（含 error_code/hint/default_hint 属性，不 import domain）。"""
    hint_value = hint

    class _FakeMyGPRError(Exception):
        error_code = code
        hint = hint_value
        default_hint = "查看日志中的技术详情"

    return _FakeMyGPRError(message)


class TestFriendlyErrorMessage:
    def test_none(self):
        assert friendly_error_message(None) == "未知错误"

    def test_plain_exception_falls_back(self):
        assert friendly_error_message(ValueError("bad value")) == "bad value"

    def test_project_busy_uses_constant(self):
        exc = _mgpr_error("MYGPR_PROJECT_BUSY", "project busy")
        assert friendly_error_message(exc) == PROJECT_BUSY_MESSAGE

    def test_mgpr_error_with_hint(self):
        exc = _mgpr_error("MYGPR_PARAMETER_VALIDATION_ERROR",
                          "参数校验失败", hint="请检查处理参数范围")
        text = friendly_error_message(exc)
        assert "[MYGPR_PARAMETER_VALIDATION_ERROR]" in text
        assert "参数校验失败" in text
        assert "请检查处理参数范围" in text

    def test_mgpr_error_without_hint_uses_default(self):
        exc = _mgpr_error("MYGPR_INPUT_DATA_ERROR", "数据读取失败")
        text = friendly_error_message(exc)
        assert "[MYGPR_INPUT_DATA_ERROR]" in text
        assert "查看日志中的技术详情" in text

    def test_hdf5_corrupt_bad_layout_message(self):
        exc = KeyError("Unable to synchronously open object "
                       "(bad version number for layout message)")
        text = friendly_error_message(exc)
        assert "数据文件损坏" in text
        assert "原始 CSV 重新导入" in text

    def test_hdf5_corrupt_file_signature(self):
        text = friendly_error_message(OSError("file signature not found"))
        assert "数据文件损坏" in text

    def test_hdf5_corrupt_truncated(self):
        text = friendly_error_message(OSError("truncated file: eof = 0, sblock = 1"))
        assert "数据文件损坏" in text

    def test_unrelated_english_message_untouched(self):
        # 非 HDF5 签名的异常保持原样（不误伤普通 OSError）
        assert friendly_error_message(OSError("permission denied")) == "permission denied"
