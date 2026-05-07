#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Theme manager for light/dark application styling with design tokens."""

import json
import logging
import os
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor
from qfluentwidgets import Theme, setTheme

from core.app_paths import get_settings_dir

logger = logging.getLogger(__name__)


# ============================================================================
# Design Tokens — 单一事实来源
# ============================================================================
TOKENS = {
    "light": {
        "bg_page": "#f3f6fb",
        "bg_card": "#ffffff",
        "bg_input": "#ffffff",
        "bg_hover": "#f7faff",
        "bg_subtle": "#f8fbff",
        "bg_log": "#fafafa",
        "border_subtle": "#dbe4ef",
        "border_default": "#d1d5db",
        "border_input": "#d1d5db",
        "border_focus": "#3b82f6",
        "text_primary": "#1f2937",
        "text_secondary": "#6b7280",
        "text_disabled": "#9ca3af",
        "text_hint": "#718096",
        "accent": "#1677ff",
        "accent_hover": "#0f67eb",
        "accent_light": "#eff5ff",
        "success": "#22c55e",
        "success_bg": "#f0fdf4",
        "success_border": "#bbf7d0",
        "warning": "#f59e0b",
        "warning_bg": "#fffbeb",
        "warning_border": "#fcd34d",
        "error": "#ef4444",
        "error_bg": "#fef2f2",
        "error_border": "#fecaca",
        "info": "#3b82f6",
        "chip_bg": "#f0f7ff",
        "chip_border": "#bfdbfe",
        "chip_text": "#1e40af",
        "shadow": "rgba(0,0,0,0.04)",
        "radius_xl": "14px",
        "radius_lg": "12px",
        "radius_md": "10px",
        "radius_sm": "8px",
        "radius_xs": "6px",
        "font_stack": '"PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI", sans-serif',
        "font_mono": '"JetBrains Mono", "Consolas", "Courier New", monospace',
        "tree_selected_bg": "#eff5ff",
        "tree_selected_text": "#155eef",
        "tree_hover_bg": "#f1f5fb",
        "tab_bg": "#edf3fb",
        "tab_text": "#64748b",
        "tab_selected_bg": "#ffffff",
        "tab_selected_text": "#155eef",
        "tab_border": "#d8e2ef",
        "scroll_bg": "transparent",
        "scroll_handle": "#cbd5e1",
        "scroll_handle_hover": "#94a3b8",
        "splitter": "#e2e8f0",
        "splitter_hover": "#3b82f6",
        "tooltip_bg": "#1e293b",
        "tooltip_text": "#f8fafc",
        "tooltip_border": "#334155",
        "workbench_panel_bg": "#ffffff",
        "workbench_panel_border": "#d9e3ef",
        "workbench_toolbar_bg": "#f6f9fd",
        "workbench_toolbar_border": "#dbe4ef",
        "workbench_tree_bg": "#fcfdff",
        "source_raw": "#1976d2",
        "source_current": "#2e7d32",
        "source_history": "#ef6c00",
        "btn_ghost_bg": "#f8fbff",
    },
    "dark": {
        "bg_page": "#16171a",
        "bg_card": "#25272b",
        "bg_input": "#23252a",
        "bg_hover": "#2d3138",
        "bg_subtle": "#1a1c20",
        "bg_log": "#25282e",
        "border_subtle": "#3a3d43",
        "border_default": "#3c4047",
        "border_input": "#3c4047",
        "border_focus": "#5aa9ff",
        "text_primary": "#f5f5f5",
        "text_secondary": "#b7bcc6",
        "text_disabled": "#6b7280",
        "text_hint": "#9ca3af",
        "accent": "#5aa9ff",
        "accent_hover": "#7abcff",
        "accent_light": "#24364f",
        "success": "#4ade80",
        "success_bg": "#245c39",
        "success_border": "#2f7a4a",
        "warning": "#fbbf24",
        "warning_bg": "#6b4d1f",
        "warning_border": "#8a6528",
        "error": "#f87171",
        "error_bg": "#7f1d1d",
        "error_border": "#b91c1c",
        "info": "#5aa9ff",
        "chip_bg": "#23262c",
        "chip_border": "#3b4048",
        "chip_text": "#c8d0db",
        "shadow": "rgba(0,0,0,0.25)",
        "radius_xl": "14px",
        "radius_lg": "12px",
        "radius_md": "10px",
        "radius_sm": "8px",
        "radius_xs": "6px",
        "font_stack": '"PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI", sans-serif',
        "font_mono": '"JetBrains Mono", "Consolas", "Courier New", monospace',
        "tree_selected_bg": "#2f4f7f",
        "tree_selected_text": "#ffffff",
        "tree_hover_bg": "#31353d",
        "tab_bg": "#292c31",
        "tab_text": "#c2c8d1",
        "tab_selected_bg": "#202226",
        "tab_selected_text": "#ffffff",
        "tab_border": "#3a3d43",
        "scroll_bg": "#23252a",
        "scroll_handle": "#5a606b",
        "scroll_handle_hover": "#7a8291",
        "splitter": "#3a3d43",
        "splitter_hover": "#5aa9ff",
        "tooltip_bg": "#1e293b",
        "tooltip_text": "#f8fafc",
        "tooltip_border": "#334155",
        "workbench_panel_bg": "#212328",
        "workbench_panel_border": "#353941",
        "workbench_toolbar_bg": "#24272d",
        "workbench_toolbar_border": "#434852",
        "workbench_tree_bg": "#23252a",
        "source_raw": "#5aa9ff",
        "source_current": "#65c466",
        "source_history": "#ffb454",
        "btn_ghost_bg": "#2a2d33",
    },
}


# ============================================================================
# Unified QSS Template
# ============================================================================
_QSS_TEMPLATE = """
/* GPR GUI Unified Design System — generated by theme_manager */

/* ========== Global ========== */
QWidget {{
    background: transparent;
    color: {text_primary};
    font-family: {font_stack};
    font-size: 12px;
}}

QMainWindow, QDialog, QWidget:window {{
    background-color: {bg_page};
}}

QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: {bg_page};
}}

/* ========== Typography ========== */
QLabel[class="titleLarge"] {{
    font-size: 16px;
    font-weight: bold;
    color: {text_primary};
    padding: 8px 0;
}}

QLabel[class="titleMedium"] {{
    font-size: 14px;
    font-weight: bold;
    color: {text_primary};
    padding: 6px 0;
}}

QLabel[class="titleSmall"] {{
    font-size: 13px;
    font-weight: 600;
    color: {text_secondary};
    padding: 4px 0;
}}

QLabel[class="textPrimary"] {{
    font-size: 12px;
    color: {text_primary};
}}

QLabel[class="textSecondary"] {{
    font-size: 11px;
    color: {text_secondary};
}}

QLabel[class="textDisabled"] {{
    font-size: 11px;
    color: {text_disabled};
}}

QLabel[class="hintText"] {{
    color: {text_hint};
    font-size: 12px;
}}

QLabel[class="metricLabel"] {{
    color: {text_secondary};
    font-size: 11px;
}}

QLabel[class="metricValue"] {{
    font-size: 14px;
    font-weight: bold;
    color: {text_primary};
}}

QLabel[class="metricGood"] {{ color: {success}; font-weight: 500; }}
QLabel[class="metricWarning"] {{ color: {warning}; font-weight: 500; }}
QLabel[class="metricBad"] {{ color: {error}; font-weight: 500; }}

QLabel[class="statusSuccess"] {{ color: {success}; font-weight: 500; }}
QLabel[class="statusWarning"] {{ color: {warning}; font-weight: 500; }}
QLabel[class="statusError"] {{ color: {error}; font-weight: 500; }}
QLabel[class="statusInfo"] {{ color: {accent}; font-weight: 500; }}

/* ========== Source states ========== */
QLabel[sourceState="raw"] {{ color: {source_raw}; font-weight: 700; }}
QLabel[sourceState="current"] {{ color: {source_current}; font-weight: 700; }}
QLabel[sourceState="history"] {{ color: {source_history}; font-weight: 700; }}

/* ========== Top info bar ========== */
QWidget#topInfoBar {{
    background: {bg_card};
    border-bottom: 1px solid {border_subtle};
    padding: 6px 10px;
}}

QLabel[class="topInfoText"] {{
    color: {text_secondary};
    font-size: 12px;
    font-weight: 500;
}}

QLabel[class="topInfoMeta"] {{
    color: {text_hint};
    font-size: 11px;
    font-weight: 500;
}}

QLabel[class="sectionTitle"] {{
    color: {text_primary};
    font-size: 17px;
    font-weight: 700;
}}

QLabel[class="statusChip"] {{
    background: {chip_bg};
    color: {chip_text};
    border: 1px solid {chip_border};
    border-radius: 9px;
    padding: 3px 10px;
    font-weight: 600;
}}

QLabel[class="statusChip-success"] {{
    background: {success_bg};
    color: {success};
    border: 1px solid {success_border};
    border-radius: 9px;
    padding: 3px 10px;
    font-weight: 600;
}}

/* ========== GroupBox (cards) ========== */
QGroupBox {{
    background: {bg_card};
    border: 1px solid {border_subtle};
    border-radius: {radius_lg};
    margin-top: 14px;
    padding: 10px;
    font-weight: 600;
    color: {text_primary};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {text_primary};
}}

QGroupBox[class="calloutBox"] {{
    background: {accent_light};
    border: 1px solid {accent};
    border-radius: {radius_lg};
}}

QGroupBox[class="calloutBox"]::title {{
    color: {accent};
}}

QGroupBox[class="lowProfileBox"] {{
    background: transparent;
    border: 1px solid {border_subtle};
    border-radius: {radius_sm};
    margin-top: 9px;
    padding: 5px 8px;
    color: {text_secondary};
    font-weight: 500;
}}

QGroupBox[class="lowProfileBox"]::title {{
    color: {text_secondary};
    left: 8px;
    padding: 0 3px;
}}

/* ========== Fallback QPushButton (non-fluent) ========== */
QPushButton {{
    background: {bg_card};
    border: 1px solid {border_default};
    border-radius: {radius_sm};
    padding: 6px 12px;
    color: {text_primary};
    min-height: 30px;
}}

QPushButton:hover {{ background: {bg_hover}; }}
QPushButton:pressed {{ background: {border_subtle}; }}
QPushButton:disabled {{
    background: {bg_page};
    color: {text_disabled};
    border-color: {border_subtle};
}}

QPushButton#btnCancel {{
    color: {error};
    border-color: {error_border};
    background: {error_bg};
}}

QPushButton#btnCancel:hover {{
    background: {error_bg};
    border-color: {error};
}}

/* ========== Inputs ========== */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {bg_input};
    border: 1px solid {border_input};
    border-radius: {radius_xs};
    padding: 5px 8px;
    min-height: 28px;
    color: {text_primary};
}}

QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 2px solid {border_focus};
    padding: 4px 7px;
}}

QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border: 1px solid {border_focus};
}}

QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: {bg_page};
    color: {text_disabled};
    border-color: {border_subtle};
}}

QFormLayout QLabel {{
    color: {text_primary};
    font-weight: 500;
}}

QCheckBox {{
    color: {text_primary};
    spacing: 6px;
    padding: 2px 0;
}}

QRadioButton {{
    color: {text_primary};
    spacing: 8px;
    padding: 2px 0;
}}

QCheckBox:hover, QRadioButton:hover {{
    color: {accent};
}}

QRadioButton:checked {{
    color: {accent};
    font-weight: 700;
}}

QCheckBox:disabled, QRadioButton:disabled {{
    color: {text_disabled};
}}

/* ========== Tab widget ========== */
QTabWidget::pane {{
    border: 1px solid {tab_border};
    border-top: 2px solid {accent};
    border-radius: 0 0 {radius_sm} {radius_sm};
    background: {bg_card};
}}

QTabBar::tab {{
    background: {tab_bg};
    color: {tab_text};
    border: 1px solid {tab_border};
    border-bottom: none;
    border-top-left-radius: {radius_sm};
    border-top-right-radius: {radius_sm};
    padding: 8px 14px;
    margin-right: 3px;
    min-width: 96px;
}}

QTabBar::tab:selected {{
    background: {tab_selected_bg};
    color: {tab_selected_text};
    font-weight: 700;
    border-bottom: none;
}}

QTabBar::tab:hover:!selected {{
    background: {bg_hover};
    color: {accent};
}}

/* ========== Empty state card ========== */
QFrame#emptyStateCard {{
    background: {bg_card};
    border: 1px solid {border_subtle};
    border-radius: {radius_xl};
    border-top: 3px solid {accent};
}}

QLabel[class="emptyTitle"] {{
    color: {text_primary};
    font-size: 20px;
    font-weight: 700;
}}

QLabel[class="emptyBadge"] {{
    color: {accent};
    background: {accent_light};
    border: 1px solid {accent};
    border-radius: 10px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 700;
}}

QLabel[class="emptySubtitle"] {{
    color: {text_secondary};
    font-size: 14px;
    font-weight: 500;
}}

QLabel[class="emptySteps"] {{
    color: {accent};
    background: {accent_light};
    border: 1px solid {accent};
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
    font-size: 13px;
}}

QLabel[class="emptyHint"] {{
    color: {text_hint};
    font-size: 12px;
}}

/* ========== Scroll / List / Text ========== */
QScrollArea {{ border: none; }}

QTextEdit, QListWidget {{
    background: {bg_input};
    border: 1px solid {border_subtle};
    border-radius: {radius_sm};
    padding: 6px;
    color: {text_primary};
}}

QListWidget:item:selected {{
    background: {tree_selected_bg};
    color: {tree_selected_text};
}}

QListWidget:disabled, QTextEdit:disabled {{
    background: {bg_page};
    color: {text_disabled};
    border-color: {border_subtle};
}}

/* ========== Scrollbar ========== */
QScrollBar:vertical {{
    background: {scroll_bg};
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {scroll_handle};
    border-radius: 4px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: {scroll_handle_hover};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background: {scroll_bg};
    height: 8px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {scroll_handle};
    border-radius: 4px;
    min-width: 30px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {scroll_handle_hover};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ========== Splitter ========== */
QSplitter::handle:horizontal {{
    background: {splitter};
    width: 2px;
}}

QSplitter::handle:horizontal:hover {{
    background: {splitter_hover};
}}

/* ========== Tooltip ========== */
QToolTip {{
    background: {tooltip_bg};
    color: {tooltip_text};
    border: 1px solid {tooltip_border};
    border-radius: {radius_xs};
    padding: 5px 10px;
    font-size: 12px;
}}

/* ========== Progress bar ========== */
QProgressBar {{
    background: {border_subtle};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    font-size: 0px;
}}

QProgressBar::chunk {{
    background: {accent};
    border-radius: 4px;
}}

/* ========== Tree Widget ========== */
QTreeWidget {{
    background: {bg_input};
    border: 1px solid {border_subtle};
    border-radius: {radius_sm};
    outline: none;
}}

QTreeWidget::item {{
    height: 28px;
    padding: 4px 8px;
    border-radius: 3px;
    margin: 2px 4px;
}}

QTreeWidget::item:selected {{
    background: {tree_selected_bg};
    color: {tree_selected_text};
}}

QTreeWidget::item:hover {{
    background: {tree_hover_bg};
}}

QTreeWidget::branch {{
    width: 16px;
}}

/* ========== Workbench Specific ========== */
QWidget#workbenchPreviewToolbar,
QLabel#workbenchLogStatus,
QTextEdit#workbenchLogText,
QLineEdit#workbenchSearchBox,
QTreeWidget#workbenchMethodTree {{
    background-color: {workbench_tree_bg};
    color: {text_primary};
    border: 1px solid {workbench_toolbar_border};
    border-radius: {radius_sm};
}}

QWidget#workbenchMethodPanel,
QWidget#workbenchParamPanel,
QWidget#workbenchPreviewPanel {{
    background-color: {workbench_panel_bg};
    border: 1px solid {workbench_panel_border};
    border-radius: {radius_xl};
}}

QGroupBox#basicActionCard,
QGroupBox#basicStatusCard,
QGroupBox#basicMethodCard {{
    background-color: {bg_card};
    border: 1px solid {workbench_panel_border};
    border-radius: {radius_xl};
    margin-top: 14px;
    padding-top: 22px;
}}

QGroupBox#basicActionCard::title,
QGroupBox#basicStatusCard::title,
QGroupBox#basicMethodCard::title {{
    left: 12px;
    padding: 0 8px;
    color: {text_primary};
}}

QGroupBox#basicActionCard PushButton,
QGroupBox#basicStatusCard PushButton,
QGroupBox#basicMethodCard PushButton,
QGroupBox#basicActionCard PrimaryPushButton,
QGroupBox#basicStatusCard PrimaryPushButton,
QGroupBox#basicMethodCard PrimaryPushButton {{
    border-radius: {radius_sm};
}}

QGroupBox#workbenchSourceCard,
QGroupBox#workbenchFavoritesCard {{
    background-color: {bg_subtle};
    border: 1px solid {workbench_panel_border};
    border-radius: {radius_lg};
    margin-top: 12px;
    padding-top: 22px;
}}

QGroupBox#workbenchSourceCard::title,
QGroupBox#workbenchFavoritesCard::title {{
    left: 12px;
    padding: 0 8px;
    color: {text_secondary};
}}

QWidget#workbenchTopBar {{
    background-color: transparent;
}}

QWidget#workbenchPreviewToolbar {{
    border-color: {workbench_toolbar_border};
    background-color: {workbench_toolbar_bg};
}}

QTreeWidget#workbenchMethodTree::item {{
    height: 30px;
    padding: 4px 10px;
    border-radius: {radius_xs};
}}

QTreeWidget#workbenchMethodTree::item:selected {{
    background-color: {tree_selected_bg};
    color: {tree_selected_text};
    border-left: 3px solid {accent};
}}

QTreeWidget#workbenchMethodTree::item:has-children {{
    color: {text_secondary};
    font-weight: 700;
}}

QTreeWidget#workbenchMethodTree::item:hover {{
    background-color: {tree_hover_bg};
}}

QLineEdit#workbenchSearchBox {{
    padding-left: 12px;
    font-size: 13px;
}}

QLabel#workbenchLogStatus {{
    background-color: {workbench_toolbar_bg};
    border-color: {workbench_toolbar_border};
    padding: 8px 10px;
}}

QTextEdit#workbenchLogText {{
    border-radius: {radius_sm};
}}

QTextEdit#basicInfoLog {{
    background-color: {bg_subtle};
    border: 1px solid {workbench_panel_border};
    border-radius: {radius_sm};
    padding: 8px;
}}

QWidget#workbenchTopBar PushButton {{
    min-height: 34px;
    border-radius: {radius_sm};
    padding: 6px 14px;
}}

QGroupBox#basicActionCard PrimaryPushButton[class="basicHeroBtn"] {{
    min-height: 42px;
    font-size: 13px;
}}

QGroupBox#basicActionCard PushButton[class="basicGhostBtn"] {{
    background-color: {btn_ghost_bg};
}}

/* ========== qfluentwidgets overrides ========== */
PrimaryPushButton {{
    background-color: {accent};
    color: #ffffff;
    border: none;
    border-radius: {radius_sm};
    padding: 8px 16px;
    font-size: 12px;
    font-weight: 600;
}}

PrimaryPushButton:hover {{
    background-color: {accent_hover};
}}

PrimaryPushButton:pressed {{
    background-color: {accent};
}}

PrimaryPushButton:disabled {{
    background-color: {border_subtle};
    color: {text_disabled};
}}

PushButton {{
    background-color: {bg_card};
    color: {text_primary};
    border: 1px solid {border_subtle};
    border-radius: {radius_sm};
    padding: 7px 15px;
    font-size: 12px;
}}

PushButton:hover {{
    background-color: {bg_hover};
    border-color: {border_focus};
}}

PushButton:pressed {{
    background-color: {border_subtle};
}}

PushButton:disabled {{
    background-color: {bg_page};
    color: {text_disabled};
    border-color: {border_subtle};
}}

PushButton[class="successBtn"] {{
    background-color: {success_bg};
    color: {success};
    border: 1px solid {success_border};
}}

PushButton[class="successBtn"]:hover {{
    background-color: {success_bg};
    border-color: {success};
}}

PushButton[class="warningBtn"] {{
    background-color: {warning_bg};
    color: {warning};
    border: 1px solid {warning_border};
}}

PushButton[class="warningBtn"]:hover {{
    background-color: {warning_bg};
    border-color: {warning};
}}

/* ========== Log area ========== */
QTextEdit[class="logArea"] {{
    background-color: {bg_log};
    border: 1px solid {border_subtle};
    border-radius: {radius_xs};
    padding: 8px;
    font-family: {font_mono};
    font-size: 11px;
    color: {text_primary};
}}

/* ========== Preview area ========== */
QWidget[class="previewArea"] {{
    background-color: {bg_page};
    border: 1px solid {border_subtle};
    border-radius: {radius_xs};
}}

QLabel[class="previewPlaceholder"] {{
    color: {text_hint};
    font-size: 14px;
    font-style: italic;
}}
"""


class ThemeManager(QObject):
    """主题管理器 — 基于 Design Tokens 生成 QSS"""

    theme_changed = pyqtSignal(str)

    THEMES = {
        "light": {"name": "浅色主题", "icon": "sun"},
        "dark": {"name": "深色主题", "icon": "moon"},
    }

    def __init__(self, base_dir: str = None):
        super().__init__()
        if base_dir is None:
            base_dir = str(Path(__file__).resolve().parents[1])
        self.base_dir = base_dir
        self.config_file = os.path.join(get_settings_dir(), "theme_config.json")
        self.current_theme = self._load_config()

    def _load_config(self) -> str:
        if not os.path.exists(self.config_file):
            return "light"
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                return json.load(f).get("theme", "light")
        except Exception:
            return "light"

    def _save_config(self, theme: str):
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump({"theme": theme}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("保存主题配置失败: %s", e)

    def get_current_theme(self) -> str:
        return self.current_theme

    def get_theme_info(self, theme: str = None) -> dict:
        if theme is None:
            theme = self.current_theme
        return self.THEMES.get(theme, self.THEMES["light"])

    def get_available_themes(self) -> list:
        return list(self.THEMES.keys())

    def _build_stylesheet(self, theme: str) -> str:
        """基于 tokens 生成样式表。"""
        tokens = TOKENS.get(theme, TOKENS["light"])
        return _QSS_TEMPLATE.format(**tokens)

    def _load_extra_stylesheet(self, theme: str) -> str:
        """加载用户扩展 QSS（可选）。"""
        extra_file = os.path.join(self.base_dir, "assets", f"styles_extra_{theme}.qss")
        if os.path.exists(extra_file):
            try:
                with open(extra_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.warning("加载扩展样式失败: %s", e)
        return ""

    def get_theme_stylesheet(self, theme: str = None) -> str:
        """获取完整主题样式表（生成 + 扩展）。"""
        if theme is None:
            theme = self.current_theme
        parts = [self._build_stylesheet(theme), self._load_extra_stylesheet(theme)]
        return "\n".join(p for p in parts if p.strip())

    def set_theme(self, theme: str):
        if theme not in self.THEMES:
            logger.warning("未知主题: %s", theme)
            return
        self.current_theme = theme
        self._save_config(theme)
        self.theme_changed.emit(theme)

    def toggle_theme(self):
        next_theme = "dark" if self.current_theme == "light" else "light"
        self.set_theme(next_theme)
        return next_theme

    def apply_app_theme(self, app, theme: str = None) -> str:
        if theme is None:
            theme = self.current_theme
        theme = theme if theme in self.THEMES else "light"
        fluent_theme = Theme.DARK if theme == "dark" else Theme.LIGHT
        setTheme(fluent_theme)
        stylesheet = self.get_theme_stylesheet(theme)
        app.setStyleSheet(stylesheet)
        return f"qfluentwidgets: {theme.upper()}"

    def apply_theme(self, widget, theme: str = None):
        stylesheet = self.get_theme_stylesheet(theme)
        if stylesheet:
            widget.setStyleSheet(stylesheet)

    def get_metric_color_class(self, value: float, thresholds: tuple) -> str:
        """根据阈值返回 metric 颜色 class 名。
        thresholds: (good_threshold, bad_threshold)
        假设 value 越大越好；如果越小越好，调用前取反或自行判断。
        """
        good, bad = thresholds
        if value >= good:
            return "metricGood"
        if value <= bad:
            return "metricBad"
        return "metricWarning"

    def get_color(self, theme: str, token_name: str) -> QColor:
        """获取指定主题下某个 token 的 QColor。"""
        hex_color = TOKENS.get(theme, TOKENS["light"]).get(token_name, "#000000")
        return QColor(hex_color)


# 全局主题管理器实例
_theme_manager = None


def get_theme_manager() -> ThemeManager:
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager
