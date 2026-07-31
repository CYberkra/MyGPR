"""Structured error taxonomy for MyGPR.

The classes in this module are intentionally GUI-free.  They let GUI, CLI,
Evidence export, AutoTune and gprMax helpers report failures with stable error
codes and audit-friendly metadata while still behaving like normal exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import traceback
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass

class ErrorSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ErrorCategory(StrEnum):
    INPUT = "input"
    PROCESSING = "processing"
    EXPORT = "export"
    CONVERSION = "conversion"
    SCORING = "scoring"
    GENERAL = "general"

__all__ = [
    "MyGPRError",
    "ErrorInfo",
    "ErrorSeverity",
    "ErrorCategory",
    "InputDataError",
    "ProcessingMethodError",
    "ParameterValidationError",
    "ProjectBusyContractError",
    "EvidenceExportError",
    "GprMaxConversionError",
    "AutoTuneScoringError",
    "error_info_from_exception",
    "format_error_for_user",
]


@dataclass(slots=True)
class ErrorInfo:
    """Serializable structured representation of an exception."""

    error_type: str
    error_code: str
    category: str
    user_message: str
    technical_detail: str = ""
    hint: str = ""
    recoverable: bool = True
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mygpr.error_info.v1",
            "error_type": self.error_type,
            "error_code": self.error_code,
            "category": self.category,
            "user_message": self.user_message,
            "technical_detail": self.technical_detail,
            "hint": self.hint,
            "recoverable": bool(self.recoverable),
            "context": dict(self.context or {}),
        }

    def compact_message(self) -> str:
        parts = [f"[{self.error_code}] {self.user_message}"]
        if self.hint:
            parts.append(f"建议：{self.hint}")
        if self.technical_detail:
            parts.append(f"技术详情：{self.technical_detail}")
        return "\n".join(parts)


class MyGPRError(RuntimeError):
    """Base class for structured MyGPR runtime errors."""

    error_code = "MYGPR_ERROR"
    category = "runtime"
    default_hint = "查看全局日志和交付文件中的错误信息。"
    recoverable = True

    def __init__(
        self,
        message: str,
        *,
        technical_detail: str | None = None,
        hint: str | None = None,
        context: Mapping[str, Any] | None = None,
        recoverable: bool | None = None,
    ):
        super().__init__(message)
        self.user_message = str(message)
        self.technical_detail = str(technical_detail or "")
        self.hint = str(hint or self.default_hint or "")
        self.context = dict(context or {})
        self.recoverable = bool(self.recoverable if recoverable is None else recoverable)

    def to_error_info(self) -> ErrorInfo:
        return ErrorInfo(
            error_type=self.__class__.__name__,
            error_code=self.error_code,
            category=self.category,
            user_message=self.user_message,
            technical_detail=self.technical_detail,
            hint=self.hint,
            recoverable=self.recoverable,
            context=self.context,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_error_info().to_dict()

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.to_error_info().compact_message()


class InputDataError(MyGPRError):
    error_code = "MYGPR_INPUT_DATA_ERROR"
    category = "input_data"
    default_hint = "确认输入文件存在、可读，并且可解析为二维 GPR 矩阵。"


class ProcessingMethodError(MyGPRError):
    error_code = "MYGPR_PROCESSING_METHOD_ERROR"
    category = "processing"
    default_hint = "先用单步方法和默认参数复现，再逐项调整参数。"




class ParameterValidationError(MyGPRError):
    error_code = "MYGPR_PARAMETER_VALIDATION_ERROR"
    category = "validation"
    default_hint = "检查处理方法参数名称、类型、范围和互斥关系。"


class ProjectBusyContractError(MyGPRError):
    error_code = "MYGPR_PROJECT_BUSY"
    category = "project"
    default_hint = "等待关联任务结束或先取消任务，再关闭项目。"


class EvidenceExportError(MyGPRError):
    error_code = "MYGPR_EVIDENCE_EXPORT_ERROR"
    category = "evidence_export"
    default_hint = "确认输出目录可写、磁盘空间充足，并避免路径被其他程序占用。"


class GprMaxConversionError(MyGPRError):
    error_code = "MYGPR_GPRMAX_CONVERSION_ERROR"
    category = "gprmax_conversion"
    default_hint = "确认 raw/background 输出完整、shape 一致，且选择了正确接收分量。"


class AutoTuneScoringError(MyGPRError):
    error_code = "MYGPR_AUTOTUNE_SCORING_ERROR"
    category = "autotune_scoring"
    default_hint = "确认关注范围、候选参数和评分模式有效；参考数据模式需要目标响应。"


def error_info_from_exception(
    exc: BaseException,
    *,
    category: str | None = None,
    context: Mapping[str, Any] | None = None,
    include_traceback: bool = False,
) -> ErrorInfo:
    """Convert arbitrary exception to :class:`ErrorInfo`."""

    if isinstance(exc, MyGPRError):
        info = exc.to_error_info()
        if context:
            merged = dict(info.context or {})
            merged.update(dict(context))
            info.context = merged
        return info

    detail = str(exc)
    if include_traceback:
        detail = "\n".join(traceback.format_exception_only(type(exc), exc)).strip()
    err_category = category or "runtime"
    return ErrorInfo(
        error_type=type(exc).__name__,
        error_code=f"MYGPR_{err_category.upper()}_ERROR",
        category=err_category,
        user_message=detail or type(exc).__name__,
        technical_detail=detail,
        hint="查看日志中的技术详情，并确认输入、参数和依赖环境。",
        recoverable=True,
        context=dict(context or {}),
    )


def format_error_for_user(exc: BaseException, *, category: str | None = None, context: Mapping[str, Any] | None = None) -> str:
    """Return compact user-facing message for any exception."""

    return error_info_from_exception(exc, category=category, context=context).compact_message()
