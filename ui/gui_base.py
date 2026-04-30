#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI基础模块 - 包含基础工具和函数，供GUI各页面共享使用
"""

import os
import sys
import re
import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("QtAgg")
from matplotlib import font_manager as fm

from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication


# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CORE_DIR_CANDIDATES = [
    os.path.abspath(os.path.join(BASE_DIR, "PythonModule")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "PythonModule_core")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "..", "repos", "PythonModule_core")),
]
for _p in CORE_DIR_CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============ 核心函数缓存 ============
_CORE_FUNC_CACHE = {}


def _get_core_func(module_name: str, func_name: str):
    """获取核心模块函数（带缓存）"""
    key = (module_name, func_name)
    fn = _CORE_FUNC_CACHE.get(key)
    if fn is None:
        mod = __import__(module_name)
        fn = getattr(mod, func_name)
        _CORE_FUNC_CACHE[key] = fn
    return fn


# ============ 数据工具 ============
def _read_matrix_csv_fast(path: str) -> np.ndarray:
    """快速读取CSV矩阵"""
    try:
        df = pd.read_csv(path, header=None, na_filter=False, low_memory=False)
        return df.values
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, OSError):
        arr = np.loadtxt(path, delimiter=",", dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr


def _to_float32_2d(data: np.ndarray) -> np.ndarray:
    """转换为float32二维数组"""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D array")
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return arr


# ============ CSV头部解析 ============
_HEADER_KEYS = [
    "Number of Samples",
    "Number of Traces",
]


def _parse_header_lines(lines):
    if len(lines) < 4:
        return None
    info = {}
    for line in lines[:4]:
        if "=" not in line:
            return None
        left, right = line.split("=", 1)
        key = left.strip()

        # 灵活解析时间窗口（支持 "Time windows" 和 "Time windows (ns)"）
        if "Time window" in key:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", right)
            if m:
                info["Time windows (ns)"] = float(m.group(0))
            continue

        # 灵活解析道间距（支持 "Trace interval" 和 "Trace interval (m)"）
        if "Trace interval" in key:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", right)
            if m:
                info["Trace interval (m)"] = float(m.group(0))
            continue

        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", right)
        if not m:
            return None
        try:
            val = float(m.group(0))
        except ValueError:
            return None
        info[key] = val
    if not all(k in info for k in _HEADER_KEYS):
        return None
    return {
        "a_scan_length": int(info["Number of Samples"]),
        "total_time_ns": float(info.get("Time windows (ns)", 0)),
        "num_traces": int(info["Number of Traces"]),
        "trace_interval_m": float(info.get("Trace interval (m)", 0.01)),
    }


def detect_csv_header(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [f.readline().strip() for _ in range(4)]
    except OSError:
        return None
    return _parse_header_lines(lines)


def _is_numeric_row(line: str) -> bool:
    parts = [p.strip() for p in line.split(",")]
    has_num = False
    for p in parts:
        if p == "":
            continue
        try:
            float(p)
            has_num = True
        except ValueError:
            return False
    return has_num


def _detect_skiprows(path: str, max_lines: int = 10) -> int:
    skip = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                if _is_numeric_row(line.strip()):
                    break
                skip += 1
    except OSError:
        return 0
    return skip


def _select_amp_column(raw_data: np.ndarray) -> int:
    if raw_data.shape[1] > 3:
        return 3
    return raw_data.shape[1] - 1


# ============ 错误消息构建 ============
def _format_explainable_error(
    what_happened: str,
    possible_causes: list,
    next_steps: list,
    technical_detail: str = "",
) -> str:
    lines = [
        f"发生了什么：{what_happened}",
        "可能原因：",
    ]
    for i, item in enumerate(possible_causes[:2], start=1):
        lines.append(f"  {i}. {item}")
    lines.append("下一步建议：")
    for i, item in enumerate(next_steps, start=1):
        lines.append(f"  {i}. {item}")
    if technical_detail:
        lines.append(f"技术详情：{technical_detail}")
    return "\n".join(lines)


def build_csv_load_error_message(err: Exception) -> str:
    return _format_explainable_error(
        what_happened="CSV 加载失败或格式不符合预期。",
        possible_causes=[
            "文件内容不是纯数值矩阵（包含文本、分隔符异常或空行过多）。",
            "CSV 编码/结构异常，导致读取后数据为空或维度不正确。",
        ],
        next_steps=[
            "用 Excel/文本编辑器确认分隔符与列结构一致，并另存为标准 UTF-8 CSV。",
            "先抽取前 50 行做小样本导入，确认可读后再导入完整文件。",
        ],
        technical_detail=str(err),
    )


def build_param_error_message(label: str, raw_value: str, detail: str) -> str:
    return _format_explainable_error(
        what_happened=f"参数'{label}'无效。",
        possible_causes=[
            "输入为空，或类型与参数要求不一致（例如应为数字却输入了文本）。",
            "参数值超出允许范围。",
        ],
        next_steps=[
            "按参数提示输入有效数值，并避免空值。",
            "若不确定，请恢复默认值后重试。",
        ],
        technical_detail=f"输入值={raw_value!r}；{detail}",
    )


def build_processing_error_message(
    err: Exception, method_name: str = "未知方法"
) -> str:
    return _format_explainable_error(
        what_happened=f"处理流程在'{method_name}'步骤执行失败。",
        possible_causes=[
            "worker 执行阶段收到非法输入或中间结果异常。",
            "方法调用失败（依赖函数报错或输出文件未生成）。",
        ],
        next_steps=[
            "先用单步处理验证该方法，再检查参数设置是否合理。",
            "查看日志中的技术详情，必要时切换到其他方法确认数据本身是否可处理。",
        ],
        technical_detail=str(err),
    )


# ============ 取消检查 ============
CANCEL_CHECK_EVERY = 4


class ProcessingCancelled(Exception):
    """用户请求取消当前后台处理"""

    pass


def _check_cancel(
    cancel_checker, step: int, stage: str, every: int = CANCEL_CHECK_EVERY
):
    if cancel_checker is None:
        return
    try:
        every_n = max(1, int(every))
    except Exception:
        every_n = CANCEL_CHECK_EVERY
    if step % every_n != 0:
        return
    if bool(cancel_checker()):
        raise ProcessingCancelled(f"用户已取消（{stage}）")


# ============ 字体配置 ============
def _preferred_cjk_font_candidates() -> list[str]:
    return [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]


def _register_matplotlib_cjk_fonts() -> list[str]:
    discovered: list[str] = []
    keywords = (
        "notosanscjk",
        "noto sans cjk",
        "sourcehansans",
        "source han sans",
        "wenquanyi",
        "simhei",
        "msyh",
        "yahei",
        "droidsansfallback",
        "sarasa",
    )
    try:
        font_paths = fm.findSystemFonts(fontext="ttf")
    except Exception:
        font_paths = []

    for path in font_paths:
        low = path.lower()
        if not any(k in low for k in keywords):
            continue
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            if name and name not in discovered:
                discovered.append(name)
        except Exception:
            continue
    return discovered


def _configure_matplotlib_cjk_fonts() -> None:
    preferred_fonts = _preferred_cjk_font_candidates()
    discovered_fonts = _register_matplotlib_cjk_fonts()

    try:
        installed = {f.name for f in fm.fontManager.ttflist}
    except Exception:
        installed = set()

    ordered = []
    for name in preferred_fonts + discovered_fonts:
        if name not in ordered:
            ordered.append(name)

    available = [name for name in ordered if name in installed]
    fallback_chain = available + ["DejaVu Sans"]

    try:
        matplotlib.rcParams["font.sans-serif"] = fallback_chain
        matplotlib.rcParams["font.family"] = fallback_chain
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _configure_qt_cjk_font(app: QApplication) -> str:
    candidates = _preferred_cjk_font_candidates()
    try:
        families = set(QFontDatabase.families())
    except Exception:
        families = set()

    for family in candidates:
        if family in families:
            app.setFont(QFont(family, 10))
            return family
    return "default"


# ============ 版本信息 ============
def _read_first_existing_text(paths):
    for path in paths:
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8-sig") as f:
                    text = f.read().strip()
                if text:
                    return text
        except Exception:
            continue
    return None


def _get_git_short_sha(base_dir: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", base_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def build_version_string(app_name: str = "GPR_GUI") -> str:
    release_candidates = [
        os.path.join(BASE_DIR, "dist", "RELEASE_VERSION.txt"),
        os.path.join(os.path.dirname(BASE_DIR), "dist", "RELEASE_VERSION.txt"),
        os.path.join(BASE_DIR, "RELEASE_VERSION.txt"),
        os.path.join(BASE_DIR, "VERSION"),
    ]
    release = _read_first_existing_text(release_candidates)
    shortsha = _get_git_short_sha(BASE_DIR)
    if release:
        return f"{app_name} {release} ({shortsha})"
    stamp = datetime.now().strftime("%Y%m%d")
    return f"{app_name} dev-{stamp} ({shortsha})"


# ============ 质量阈值加载 ============
def load_quality_dashboard_thresholds() -> dict:
    from core.preset_profiles import DEFAULT_QUALITY_DASHBOARD_THRESHOLDS

    config_path = os.path.join(BASE_DIR, "config", "quality_gate_thresholds.json")
    if not os.path.isfile(config_path):
        return dict(DEFAULT_QUALITY_DASHBOARD_THRESHOLDS)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return dict(DEFAULT_QUALITY_DASHBOARD_THRESHOLDS)

    direct = payload.get("quality_dashboard", {}).get("thresholds", {})
    if isinstance(direct, dict) and direct:
        merged = dict(DEFAULT_QUALITY_DASHBOARD_THRESHOLDS)
        merged.update(direct)
        return merged

    stolt = payload.get("stolt_golden", {}).get("thresholds", {})
    if isinstance(stolt, dict) and stolt:
        for key in (
            "hyperbola_target",
            "layered_small",
            "clutter_spiky",
            "nan_inf_robustness",
        ):
            th = stolt.get(key)
            if isinstance(th, dict) and th:
                merged = dict(DEFAULT_QUALITY_DASHBOARD_THRESHOLDS)
                merged.update(th)
                return merged
        first = next(iter(stolt.values()))
        if isinstance(first, dict) and first:
            merged = dict(DEFAULT_QUALITY_DASHBOARD_THRESHOLDS)
            merged.update(first)
            return merged

    return dict(DEFAULT_QUALITY_DASHBOARD_THRESHOLDS)


# 配置matplotlib字体
_configure_matplotlib_cjk_fonts()
