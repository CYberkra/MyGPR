#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-facing labels shared by the field product UI.

The project files keep stable English keys for compatibility.  This module maps
those keys to labels that make sense to field operators.
"""

from __future__ import annotations

SEVERITY_LABELS = {
    "error": "阻断",
    "warning": "待复核",
    "info": "提示",
}

QC_CODE_LABELS = {
    "missing_primary": "缺少主雷达数据",
    "missing_raw": "原始数据缺失",
    "raw_integrity_mismatch": "原始数据被修改",
    "raw_integrity_pending": "原始数据待校验",
    "invalid_matrix": "B-scan 数据无效",
    "non_finite_samples": "存在异常采样值",
    "matrix_shape": "数据结构可读取",
    "airborne_metadata_missing": "缺少逐道空间元数据",
    "rtk_missing": "缺少 RTK 辅助文件",
    "imu_missing": "缺少 IMU 辅助文件",
    "altimeter_missing": "缺少高度计辅助文件",
    "rtk_valid": "RTK 辅助文件有效",
    "imu_valid": "IMU 辅助文件有效",
    "altimeter_valid": "高度计辅助文件有效",
    "rtk_invalid": "RTK 辅助文件异常",
    "imu_invalid": "IMU 辅助文件异常",
    "altimeter_invalid": "高度计辅助文件异常",
    "no_processing_result": "尚未保存处理结果",
    "no_interpretation": "尚未添加目标标注",
}

DELIVERY_ROLE_LABELS = {
    "delivery_manifest": "成果清单",
    "delivery_report": "项目报告",
    "delivery_checksums": "文件校验清单",
    "spatial_synthesis": "空间成果汇总",
    "line_record": "测线记录",
    "qc_report": "质控记录",
    "interpretation": "目标标注",
    "processing_result": "处理结果",
    "evidence": "交付文件",
}

SIDECAR_LABELS = {
    "rtk": "RTK",
    "imu": "IMU",
    "altimeter": "高度计",
    "trace_timestamps": "逐道时间戳",
}

LINE_STATUS_LABELS = {
    "imported": "已导入",
    "ready": "可处理",
    "processing": "处理中",
    "processed": "已处理",
    "error": "异常",
}


def _fallback_from_key(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "--"
    return text.replace("_", " ")


def severity_label(value: str) -> str:
    return SEVERITY_LABELS.get(str(value or "").lower(), str(value or "").upper() or "--")


def qc_code_label(code: str) -> str:
    return QC_CODE_LABELS.get(str(code or ""), _fallback_from_key(code))


def delivery_role_label(role: str) -> str:
    return DELIVERY_ROLE_LABELS.get(str(role or ""), _fallback_from_key(role))


def sidecar_label(kind: str) -> str:
    return SIDECAR_LABELS.get(str(kind or ""), _fallback_from_key(kind))


def line_status_label(status: str) -> str:
    return LINE_STATUS_LABELS.get(str(status or ""), _fallback_from_key(status))


__all__ = [
    "delivery_role_label",
    "line_status_label",
    "qc_code_label",
    "severity_label",
    "sidecar_label",
]
