#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight visual polish tokens for MyGPR UI.

GX-UI-010 keeps the existing application structure and only applies a
low-risk QSS layer. It intentionally avoids new dependencies and does not
change processing, AutoTune scoring, gprMax, or Evidence behavior.
"""

from __future__ import annotations

from functools import lru_cache

APP_BACKGROUND = "#F7F9FC"
CARD_BACKGROUND = "#FFFFFF"
CARD_BORDER = "#E3EAF3"
PRIMARY = "#13A6A4"
PRIMARY_DARK = "#08776F"
PRIMARY_LIGHT = "#E7FAF7"
TEXT_PRIMARY = "#172033"
TEXT_MUTED = "#64748B"
SUCCESS = "#22A06B"
WARNING = "#F59E0B"
DANGER = "#EF4444"


def is_dark_ui(theme: str | None = None, widget=None) -> bool:
    """Return True when the effective MyGPR UI should use dark colors.

    Explicit LIGHT/DARK values from ``theme_manager`` win. Palette probing is
    only used for automatic/system themes, preventing a nested child palette
    from flipping one page independently.
    """
    theme_name = str(theme or "").strip().lower()

    # qfluentwidgets enums are commonly rendered as ``Theme.DARK`` or
    # ``Theme.LIGHT``. Accept both exact names and contained keywords.
    if theme_name:
        if any(token in theme_name for token in ("dark", "night", "black", "深色", "暗色")):
            return True
        if any(token in theme_name for token in ("light", "white", "day", "浅色", "亮色")):
            return False
        if theme_name not in {"auto", "system", "follow_system", "default", "none", "unknown"}:
            # Unknown but explicit theme names are safer as light than letting a
            # nested child palette flip one page independently.
            return False

    try:
        from PyQt6.QtGui import QPalette
        from PyQt6.QtWidgets import QApplication

        source = widget if widget is not None else QApplication.instance()
        if source is not None:
            color = source.palette().color(QPalette.ColorRole.Window)
            if color.isValid():
                return color.lightness() < 128
    except Exception:
        pass

    return False


def set_dynamic_property(widget, name: str, value) -> bool:
    """Set a Qt dynamic property only when it changes.

    Returns True when a repolish is required, avoiding repeated
    unpolish/polish calls for frequently refreshed status chips.
    """
    if widget is None:
        return False
    if widget.property(name) == value:
        return False
    widget.setProperty(name, value)
    return True


def repolish(widget) -> None:
    """Safely repolish one widget after a dynamic property change."""
    if widget is None:
        return
    try:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
    except Exception:
        return
    widget.update()


def get_effective_theme_key(theme: str | None = None, widget=None) -> str:
    """Return a stable key for local style caches."""
    return "dark" if is_dark_ui(theme, widget=widget) else "light"


def get_app_polish_stylesheet(theme: str | None = None, widget=None) -> str:
    """Return a conservative cached QSS polish layer for light or dark theme."""
    return _get_app_polish_stylesheet_cached(is_dark_ui(theme, widget=widget))


@lru_cache(maxsize=2)
def _get_app_polish_stylesheet_cached(is_dark: bool) -> str:
    """Build the app polish stylesheet once per effective theme."""
    template = _load_app_polish_template()
    for key, value in _theme_tokens(is_dark).items():
        template = template.replace(f"__{key.upper()}__", str(value))
    return template


@lru_cache(maxsize=1)
def _load_app_polish_template() -> str:
    """Load the shared QSS template without keeping 1500+ lines in Python code."""
    try:
        from importlib import resources

        return (
            resources.files(__package__)
            .joinpath("qss/app_polish_template.qss")
            .read_text(encoding="utf-8")
        )
    except Exception:
        from pathlib import Path

        return Path(__file__).with_name("qss").joinpath("app_polish_template.qss").read_text(
            encoding="utf-8"
        )


def _theme_tokens(is_dark: bool) -> dict[str, str]:
    """Return QSS color tokens for the current theme."""
    if is_dark:
        return {
            "bg": "#111418",
            "card": "#1B2027",
            "card_2": "#202630",
            "card_3": "#151A21",
            "border": "#323A46",
            "border_2": "#3B4654",
            "primary": "#60A5FA",
            "primary_dark": "#93C5FD",
            "primary_light": "#172A46",
            "primary_bg": "#172A46",
            "text": "#EAF0F8",
            "muted": "#A6B1C2",
            "subtle": "#7E8A9C",
            "input_bg": "#121820",
            "input_focus": "#172033",
            "tab_bg": "#1A2029",
            "tab_sel": "#242B36",
            "table_alt": "#161C24",
            "good_bg": "#0F2F23",
            "good_text": "#86EFAC",
            "good_border": "#245D43",
            "warn_bg": "#3A2A10",
            "warn_text": "#FBBF24",
            "warn_border": "#72521C",
            "danger_bg": "#3B1517",
            "danger_text": "#FCA5A5",
            "danger_border": "#7F1D1D",
            "neutral_bg": "#202733",
            "neutral_text": "#B7C1D1",
            "neutral_border": "#3A4453",
            "empty_start": "#171D25",
            "empty_end": "#111820",
            "dash": "#3B4654",
            "toolbar_bg": "#14191F",
            "disabled_bg": "#1A2028",
            "disabled_text": "#687386",
            "info_bg": "#172A46",
            "info_text": "#BFDBFE",
            "info_border": "#2B4C7E",
        }

    return {
        "bg": "#F6F8FC",
        "card": CARD_BACKGROUND,
        "card_2": "#F8FAFC",
        "card_3": "#FBFDFF",
        "border": "#E6EBF2",
        "border_2": "#DDE5EF",
        "primary": PRIMARY,
        "primary_dark": PRIMARY_DARK,
        "primary_light": PRIMARY_LIGHT,
        "primary_bg": PRIMARY_LIGHT,
        "text": TEXT_PRIMARY,
        "muted": TEXT_MUTED,
        "subtle": "#94A3B8",
        "input_bg": "#FBFDFF",
        "input_focus": "#FFFFFF",
        "tab_bg": "#F1F5F9",
        "tab_sel": "#FFFFFF",
        "table_alt": "#F8FAFC",
        "good_bg": "#E9F8F0",
        "good_text": "#147A4E",
        "good_border": "#BDEBD1",
        "warn_bg": "#FFF7E8",
        "warn_text": "#B45309",
        "warn_border": "#FDE4B4",
        "danger_bg": "#FEF2F2",
        "danger_text": "#B91C1C",
        "danger_border": "#FECACA",
        "neutral_bg": "#F1F5F9",
        "neutral_text": "#475569",
        "neutral_border": "#D7DEE8",
        "empty_start": "#FBFDFF",
        "empty_end": "#F3F7FE",
        "dash": "#C8D3E1",
        "toolbar_bg": "#FFFFFF",
        "disabled_bg": "#F1F5F9",
        "disabled_text": "#94A3B8",
        "info_bg": "#EFF6FF",
        "info_text": "#2563EB",
        "info_border": "#BFDBFE",
    }
