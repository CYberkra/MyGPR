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

    def test_real_mgpr_error_no_duplicate_code_or_hint(self):
        # R4: MyGPRError.__str__ 是 compact_message（含 [code] 与 建议后缀），
        # 重组时不得出现双 [code] 或双 建议后缀
        from mygpr.domain.common.errors import ProcessingMethodError
        exc = ProcessingMethodError("方法执行失败")
        text = friendly_error_message(exc)
        assert text.count("[MYGPR_PROCESSING_METHOD_ERROR]") == 1
        assert "建议" not in text
        assert text.count("先用单步方法和默认参数复现") == 1
        assert text.startswith("[MYGPR_PROCESSING_METHOD_ERROR] 方法执行失败 — ")

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
        # 非 HDF5 签名、非占用类异常保持原样
        assert friendly_error_message(ValueError("boom")) == "boom"

    def test_permission_error_maps_to_file_in_use(self):
        # R2: Windows os.replace 发布窗口被预览句柄占用 → PermissionError
        exc = PermissionError("[WinError 5] 拒绝访问。: 'a.h5' -> 'b.h5'")
        text = friendly_error_message(exc)
        assert "文件被占用" in text
        assert "重试" in text

    def test_file_in_use_english_oserror_maps(self):
        text = friendly_error_message(OSError(
            "Unable to open file (file being used by another process)"))
        assert "文件被占用" in text
