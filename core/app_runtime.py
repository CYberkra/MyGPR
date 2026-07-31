# -*- coding: utf-8 -*-
"""Application runtime helpers for logging, QSS sanitising, and UI settings.

The functions in this module are deliberately small and side-effect-light so
``app_qt.py`` can remain a main-window shell rather than a utility dump.
"""

from __future__ import annotations

import json
import logging
import os
import warnings

from core.app_paths import get_logs_dir, get_settings_dir
from core.storage_primitives import atomic_write_json

logger = logging.getLogger(__name__)


def configure_logging() -> str:
    """统一配置应用日志。"""
    log_dir = get_logs_dir()
    log_path = os.path.join(log_dir, "mygpr.log")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    default_level = logging.DEBUG if os.getenv("MYGPR_DEBUG", os.getenv("GPR_GUI_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"} else logging.INFO
    root_logger.setLevel(default_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    for noisy_logger in [
        "matplotlib",
        "matplotlib.font_manager",
        "fontTools",
        "PIL",
        "numexpr",
        "PyQt6.uic",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    warnings.filterwarnings(
        "ignore",
        message=r"Glyph .* missing from font\(s\).*",
        category=UserWarning,
    )

    return log_path


def _sanitize_qss(qss: str) -> str:
    """清理 Qt 不支持或会触发布局警告的样式声明。"""
    cleaned_lines = []
    for line in qss.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("cursor:"):
            continue
        if stripped.startswith("max-width: 16777215"):
            continue
        if stripped.startswith("max-height: 16777215"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _get_settings_path() -> str:
    """获取设置文件路径"""
    settings_dir = get_settings_dir()
    return os.path.join(settings_dir, "gpr_gui_settings.json")


def _load_app_settings_dict() -> dict:
    """加载设置字典；损坏设置不会覆盖或阻断应用启动。"""
    settings_path = _get_settings_path()
    if not os.path.exists(settings_path):
        return {}
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load settings %s: %s", settings_path, exc)
        return {}
    if not isinstance(loaded, dict):
        logger.warning("Ignoring non-object settings document: %s", settings_path)
        return {}
    return loaded


def _save_app_settings_dict(settings: dict) -> None:
    """使用原子替换保存设置，避免异常退出留下半写文件。"""
    if not isinstance(settings, dict):
        logger.warning("Ignoring invalid settings payload of type %s", type(settings).__name__)
        return
    settings_path = _get_settings_path()
    try:
        atomic_write_json(settings_path, settings)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Failed to save settings %s: %s", settings_path, exc)


def _save_last_data_path(path: str):
    """保存上次加载的数据路径。"""
    if not os.path.exists(path):
        return
    settings = _load_app_settings_dict()
    settings["last_data_path"] = path
    _save_app_settings_dict(settings)


def _load_last_data_path() -> str:
    """加载上次的数据路径。"""
    settings = _load_app_settings_dict()
    path = settings.get("last_data_path", "")
    return path if isinstance(path, str) and path and os.path.exists(path) else ""
