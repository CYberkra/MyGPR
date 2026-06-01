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

    Important rule: an explicit LIGHT/DARK value from ``theme_manager`` wins.
    Earlier builds also inspected the widget palette even when the manager
    explicitly reported LIGHT. On Windows/qfluentwidgets this can mis-detect a
    dark child palette inside a light application, making AutoTune turn black
    while the rest of the app is light.

    Palette probing is now used only when the explicit theme is missing or set
    to an automatic/system value. This keeps light and dark modes fully
    deterministic.

    /* V0.7.0 brand and report export polish -------------------------- */
    QLabel[class="sectionTitle"] {{
        color: {text};
        font-size: 18px;
        font-weight: 850;
        background: transparent;
    }}
    QLabel[class="hintText"] {{
        color: {muted};
        font-size: 12px;
        background: transparent;
    }}
    QFrame#EvidenceHeroCard {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {card}, stop:1 {card_2});
        border: 1px solid {border};
        border-radius: 18px;
    }}
    QLabel#EvidenceHeroMark {{
        background: {primary_light};
        color: {primary_dark};
        border: 1px solid {border_2};
        border-radius: 15px;
        font-weight: 900;
        letter-spacing: 0.5px;
    }}
    QLabel#EvidenceHeroBadge {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 12px;
        padding: 5px 10px;
        font-weight: 800;
    }}

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

    Returns True when a repolish is required. This avoids repeated
    unpolish/polish calls for status chips that refresh frequently.

    /* V0.7.0 brand and report export polish -------------------------- */
    QLabel[class="sectionTitle"] {{
        color: {text};
        font-size: 18px;
        font-weight: 850;
        background: transparent;
    }}
    QLabel[class="hintText"] {{
        color: {muted};
        font-size: 12px;
        background: transparent;
    }}
    QFrame#EvidenceHeroCard {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {card}, stop:1 {card_2});
        border: 1px solid {border};
        border-radius: 18px;
    }}
    QLabel#EvidenceHeroMark {{
        background: {primary_light};
        color: {primary_dark};
        border: 1px solid {border_2};
        border-radius: 15px;
        font-weight: 900;
        letter-spacing: 0.5px;
    }}
    QLabel#EvidenceHeroBadge {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 12px;
        padding: 5px 10px;
        font-weight: 800;
    }}

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
    """Return a conservative QSS polish layer for light or dark theme.

    The stylesheet is cached by the effective dark/light state. Expensive string
    construction and global ``setStyleSheet`` churn were visible during theme
    toggles and page refreshes, so callers should pass ``widget=self`` when
    possible to resolve the real Qt palette before the cached stylesheet is
    selected.

    /* V0.7.0 brand and report export polish -------------------------- */
    QLabel[class="sectionTitle"] {{
        color: {text};
        font-size: 18px;
        font-weight: 850;
        background: transparent;
    }}
    QLabel[class="hintText"] {{
        color: {muted};
        font-size: 12px;
        background: transparent;
    }}
    QFrame#EvidenceHeroCard {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {card}, stop:1 {card_2});
        border: 1px solid {border};
        border-radius: 18px;
    }}
    QLabel#EvidenceHeroMark {{
        background: {primary_light};
        color: {primary_dark};
        border: 1px solid {border_2};
        border-radius: 15px;
        font-weight: 900;
        letter-spacing: 0.5px;
    }}
    QLabel#EvidenceHeroBadge {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 12px;
        padding: 5px 10px;
        font-weight: 800;
    }}

    """
    return _get_app_polish_stylesheet_cached(is_dark_ui(theme, widget=widget))


@lru_cache(maxsize=2)
def _get_app_polish_stylesheet_cached(is_dark: bool) -> str:
    """Build the app polish stylesheet once per effective theme."""
    if is_dark:
        bg = "#111418"
        card = "#1B2027"
        card_2 = "#202630"
        card_3 = "#151A21"
        border = "#323A46"
        border_2 = "#3B4654"
        primary = "#60A5FA"
        primary_dark = "#93C5FD"
        primary_light = "#172A46"
        text = "#EAF0F8"
        muted = "#A6B1C2"
        subtle = "#7E8A9C"
        input_bg = "#121820"
        input_focus = "#172033"
        tab_bg = "#1A2029"
        tab_sel = "#242B36"
        table_alt = "#161C24"
        good_bg, good_text, good_border = "#0F2F23", "#86EFAC", "#245D43"
        warn_bg, warn_text, warn_border = "#3A2A10", "#FBBF24", "#72521C"
        danger_bg, danger_text, danger_border = "#3B1517", "#FCA5A5", "#7F1D1D"
        neutral_bg, neutral_text, neutral_border = "#202733", "#B7C1D1", "#3A4453"
        empty_start, empty_end = "#171D25", "#111820"
        dash = "#3B4654"
        toolbar_bg = "#14191F"
        disabled_bg, disabled_text = "#1A2028", "#687386"
        info_bg, info_text, info_border = "#172A46", "#BFDBFE", "#2B4C7E"
    else:
        bg = "#F6F8FC"
        card = CARD_BACKGROUND
        card_2 = "#F8FAFC"
        card_3 = "#FBFDFF"
        border = "#E6EBF2"
        border_2 = "#DDE5EF"
        primary = PRIMARY
        primary_dark = PRIMARY_DARK
        primary_light = PRIMARY_LIGHT
        text = "#172033"
        muted = TEXT_MUTED
        subtle = "#94A3B8"
        input_bg = "#FBFDFF"
        input_focus = "#FFFFFF"
        tab_bg = "#F1F5F9"
        tab_sel = "#FFFFFF"
        table_alt = "#F8FAFC"
        good_bg, good_text, good_border = "#E9F8F0", "#147A4E", "#BDEBD1"
        warn_bg, warn_text, warn_border = "#FFF7E8", "#B45309", "#FDE4B4"
        danger_bg, danger_text, danger_border = "#FEF2F2", "#B91C1C", "#FECACA"
        neutral_bg, neutral_text, neutral_border = "#F1F5F9", "#475569", "#D7DEE8"
        empty_start, empty_end = "#FBFDFF", "#F3F7FE"
        dash = "#C8D3E1"
        toolbar_bg = "#FFFFFF"
        disabled_bg, disabled_text = "#F1F5F9", "#94A3B8"
        info_bg, info_text, info_border = "#EFF6FF", "#2563EB", "#BFDBFE"

    return f"""
    QMainWindow {{
        background: {bg};
    }}
    QWidget {{
        color: {text};
        selection-background-color: {primary_light};
        selection-color: {text};
    }}

    /* Product cards --------------------------------------------------- */
    QFrame#Card, QFrame#AutoTuneHeader, QFrame#PreviewCard, QFrame#mainPlotCard,
    QFrame#progressPanel, QFrame#runtimeDrawer {{
        background: {card};
        border: 1px solid {border};
        border-radius: 18px;
    }}
    QFrame#PlotCardHeader {{
        background: transparent;
        border: none;
    }}
    QFrame#mainPlotToolbar {{
        background: {toolbar_bg};
        border: 1px solid {border};
        border-radius: 12px;
    }}
    QLabel#PlotTitle {{
        color: {text};
        font-size: 17px;
        font-weight: 800;
        background: transparent;
        min-height: 24px;
    }}
    QLabel#PlotSubtitle {{
        color: {muted};
        font-size: 12px;
        background: transparent;
        min-height: 0px;
        max-height: 0px;
    }}

    /* Status chips ---------------------------------------------------- */
    QLabel#StatusChip, QLabel#MiniTag, QLabel#PlotStatusChip, QLabel#PlotInfoChip, QLabel#RuntimeSummaryChip, QLabel#PlotModeChip {{
        background: {primary_light};
        color: {primary_dark};
        border: 1px solid {border_2};
        border-radius: 11px;
        padding: 4px 9px;
        font-weight: 700;
    }}
    QLabel#StatusChip[tone="neutral"], QLabel#PlotStatusChip[tone="neutral"], QLabel#RuntimeSummaryChip[tone="neutral"], QLabel#PlotModeChip[tone="neutral"] {{
        background: {neutral_bg};
        color: {neutral_text};
        border-color: {neutral_border};
    }}
    QLabel#StatusChip[tone="good"], QLabel#PlotStatusChip[tone="good"], QLabel#RuntimeSummaryChip[tone="good"], QLabel#PlotModeChip[tone="success"] {{
        background: {good_bg};
        color: {good_text};
        border-color: {good_border};
    }}
    QLabel#StatusChip[tone="info"], QLabel#PlotStatusChip[tone="info"], QLabel#RuntimeSummaryChip[tone="info"], QLabel#PlotModeChip[tone="info"] {{
        background: {info_bg};
        color: {info_text};
        border-color: {info_border};
    }}
    QLabel#StatusChip[tone="warning"], QLabel#PlotStatusChip[tone="warning"], QLabel#RuntimeSummaryChip[tone="warning"], QLabel#PlotModeChip[tone="warning"] {{
        background: {warn_bg};
        color: {warn_text};
        border-color: {warn_border};
    }}
    QLabel#StatusChip[tone="danger"], QLabel#PlotStatusChip[tone="danger"], QLabel#RuntimeSummaryChip[tone="danger"], QLabel#PlotModeChip[tone="danger"] {{
        background: {danger_bg};
        color: {danger_text};
        border-color: {danger_border};
    }}

    /* Empty state ----------------------------------------------------- */
    QFrame#emptyStateCard {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 {empty_start}, stop:1 {empty_end});
        border: 1px dashed {dash};
        border-radius: 18px;
    }}
    QLabel.emptyBadge, QLabel[class="emptyBadge"] {{
        background: {primary_light};
        color: {primary_dark};
        border: 1px solid {border_2};
        border-radius: 12px;
        padding: 5px 12px;
        font-weight: 800;
    }}
    QLabel.emptyTitle, QLabel[class="emptyTitle"] {{
        color: {text};
        font-size: 20px;
        font-weight: 800;
    }}
    QLabel.emptySubtitle, QLabel.emptyHint, QLabel.emptySteps,
    QLabel[class="emptySubtitle"], QLabel[class="emptyHint"], QLabel[class="emptySteps"] {{
        color: {muted};
    }}

    /* Controls -------------------------------------------------------- */
    QPushButton {{
        border-radius: 10px;
        padding: 7px 11px;
        border: 1px solid {border_2};
        background: {card_2};
        color: {text};
        font-weight: 600;
    }}
    QPushButton:hover {{
        background: {primary_light};
        border-color: {primary};
        color: {primary_dark};
    }}
    QPushButton:pressed {{
        background: {input_focus};
    }}
    QPushButton:disabled {{
        background: {disabled_bg};
        color: {disabled_text};
        border-color: {border};
    }}
    QComboBox, QSpinBox, QLineEdit, QTextEdit {{
        border: 1px solid {border_2};
        border-radius: 10px;
        background: {input_bg};
        color: {text};
        padding: 4px 7px;
    }}
    QComboBox:focus, QSpinBox:focus, QLineEdit:focus, QTextEdit:focus {{
        border: 1px solid {primary};
        background: {input_focus};
    }}
    QComboBox QAbstractItemView {{
        background: {card};
        color: {text};
        border: 1px solid {border_2};
        selection-background-color: {primary_light};
        selection-color: {text};
    }}

    /* Tabs and tables ------------------------------------------------- */
    QTabWidget::pane {{
        border: 1px solid {border};
        border-radius: 14px;
        background: {card};
        top: -1px;
    }}
    QTabBar::tab {{
        background: {tab_bg};
        border: 1px solid {border};
        border-bottom: none;
        padding: 8px 12px;
        margin-right: 3px;
        min-width: 66px;
        border-top-left-radius: 11px;
        border-top-right-radius: 11px;
        color: {muted};
        font-weight: 600;
    }}
    QTabBar::tab:selected {{
        background: {tab_sel};
        color: {primary_dark};
        font-weight: 800;
    }}
    QTableWidget {{
        gridline-color: {border};
        selection-background-color: {primary_light};
        selection-color: {text};
        alternate-background-color: {table_alt};
        border: 1px solid {border};
        border-radius: 12px;
        background: {card};
        color: {text};
    }}
    QHeaderView::section {{
        background: {tab_bg};
        color: {text};
        border: none;
        border-right: 1px solid {border};
        border-bottom: 1px solid {border};
        padding: 6px 8px;
        font-weight: 800;
    }}
    QGroupBox {{
        background: {card};
        color: {text};
        border: 1px solid {border};
        border-radius: 14px;
        margin-top: 13px;
        padding: 14px 12px 12px 12px;
        font-weight: 800;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 7px;
        color: {text};
        background: {card};
    }}
    QCheckBox {{
        color: {text};
        spacing: 8px;
    }}
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QLabel#Muted, QLabel#Hint, QLabel#AutoTuneSubtitle {{
        color: {muted};
    }}
    QLabel#PanelTitle, QLabel#CardTitle {{
        color: {text};
        background: transparent;
        font-weight: 800;
    }}
    QLabel#AutoTuneTitle {{
        color: {text};
        background: transparent;
        font-size: 23px;
        font-weight: 800;
    }}
    QLabel#PreviewCanvas {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 {card_3}, stop:1 {card_2});
        border: 1px dashed {dash};
        border-radius: 14px;
        color: {muted};
        font-size: 13px;
        padding: 14px;
    }}


    /* Premium theme consistency additions ---------------------------- */
    QToolBar, QToolBar#mainPlotToolbar, QWidget#mainPlotToolbar {{
        background: {toolbar_bg};
        border: 1px solid {border};
        border-radius: 12px;
        spacing: 4px;
        padding: 4px;
    }}
    QToolButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 5px;
        color: {text};
    }}
    QToolButton:hover {{
        background: {primary_light};
        border-color: {border_2};
    }}
    QToolButton:pressed, QToolButton:checked {{
        background: {primary_light};
        border-color: {primary};
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {border_2};
        border-radius: 5px;
        min-height: 28px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {primary};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
        border: none;
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {border_2};
        border-radius: 5px;
        min-width: 28px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {primary};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: transparent;
        border: none;
        width: 0px;
    }}
    QLabel#EmptyBadge {{
        background: {primary_light};
        color: {primary_dark};
        border: 1px solid {border_2};
        border-radius: 12px;
        padding: 5px 12px;
        font-weight: 800;
    }}
    QLabel#EmptyTitle {{
        color: {text};
        font-size: 20px;
        font-weight: 800;
        background: transparent;
        border: none;
    }}
    QLabel#EmptySubtitle, QLabel#EmptyHint, QLabel#EmptySteps {{
        color: {muted};
        background: transparent;
        border: none;
    }}
    QProgressBar {{
        background: {card_2};
        border: 1px solid {border};
        border-radius: 8px;
        color: {text};
        text-align: center;
        min-height: 16px;
    }}
    QProgressBar::chunk {{
        background: {primary};
        border-radius: 7px;
    }}


    QAbstractScrollArea {{
        background: {card};
        border: 1px solid {border};
        border-radius: 12px;
    }}
    QAbstractScrollArea::viewport {{
        background: {card};
        color: {text};
    }}
    QTableWidget::item {{
        color: {text};
        background: transparent;
        padding: 4px;
    }}
    QTableWidget::item:selected {{
        background: {primary_light};
        color: {text};
    }}
    QTableCornerButton::section {{
        background: {tab_bg};
        border: none;
        border-right: 1px solid {border};
        border-bottom: 1px solid {border};
    }}
    QSpinBox::up-button, QSpinBox::down-button,
    QComboBox::drop-down {{
        background: {card_2};
        border-left: 1px solid {border};
        border-top-right-radius: 8px;
        border-bottom-right-radius: 8px;
        width: 22px;
    }}
    QSpinBox::up-arrow, QSpinBox::down-arrow,
    QComboBox::down-arrow {{
        width: 8px;
        height: 8px;
    }}

    /* Stronger overrides for product workspace and tab crowding ---------- */
    QFrame#mainPlotCard {{
        background: {card};
        border: 1px solid {border};
        border-radius: 18px;
    }}
    QFrame#emptyStateCard {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 {empty_start}, stop:1 {empty_end});
        border: 1px dashed {dash};
        border-radius: 18px;
    }}
    QTabBar::tab {{
        min-width: 66px;
        padding-left: 10px;
        padding-right: 10px;
    }}
    /* Visual language polish ------------------------------------------ */
    QLabel#EmptyIcon {{
        background: {primary_light};
        color: {primary_dark};
        border: 1px solid {border_2};
        border-radius: 24px;
        min-width: 48px;
        min-height: 48px;
        max-width: 48px;
        max-height: 48px;
        font-size: 28px;
        font-weight: 900;
    }}
    QPushButton#PrimaryButton {{
        background: {primary};
        color: white;
        border: 1px solid {primary};
        border-radius: 11px;
        padding: 7px 12px;
        font-weight: 800;
    }}
    QPushButton#PrimaryButton:hover {{
        background: {primary_dark};
        border-color: {primary_dark};
        color: white;
    }}
    QPushButton#SecondaryButton {{
        background: {card_2};
        color: {text};
        border: 1px solid {border_2};
        border-radius: 11px;
        padding: 7px 12px;
        font-weight: 700;
    }}
    QPushButton#SecondaryButton:hover {{
        background: {primary_light};
        color: {primary_dark};
        border-color: {primary};
    }}
    QFrame#CandidateRow {{
        background: {card_2};
        border: 1px solid {border};
        border-radius: 12px;
    }}
    QFrame#CandidateRow:hover {{
        background: {primary_light};
        border-color: {primary};
    }}
    QLabel#CandidateTitle {{
        color: {text};
        font-weight: 800;
        background: transparent;
    }}
    QLabel#CandidateSubtitle {{
        color: {muted};
        font-size: 12px;
        background: transparent;
    }}
    QLabel#MethodTag {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 9px;
        padding: 3px 7px;
        font-size: 11px;
        font-weight: 800;
    }}
    QFrame#RankPanel {{
        background: {card_3};
        border: 1px solid {border};
        border-radius: 12px;
    }}
    QLabel#InlineTitle {{
        color: {text};
        font-weight: 800;
        background: transparent;
    }}
    QLabel#FieldLabel {{
        color: {muted};
        font-size: 12px;
        background: transparent;
    }}

    /* V22 plot workspace refinement ----------------------------------- */
    QSplitter::handle {{
        background: {border};
        margin: 2px;
    }}
    QFrame#mainPlotCard {{
        border-radius: 20px;
    }}
    QFrame#mainPlotToolbar {{
        padding: 1px;
    }}
    QFrame#runtimeDrawer {{
        border-radius: 14px;
    }}
    QLabel#PlotTitle {{
        font-size: 17px;
        letter-spacing: 0.2px;
    }}
    QLabel#PlotStatusChip, QLabel#PlotInfoChip {{
        min-height: 22px;
    }}
    QToolButton {{
        min-width: 26px;
        min-height: 26px;
    }}
    QTextEdit {{
        line-height: 1.35em;
    }}

    /* V23 micro polish: group titles, logs, and secondary tab buttons ----- */
    QGroupBox {{
        margin-top: 0px;
        padding-top: 30px;
        border-radius: 16px;
    }}
    QGroupBox::title {{
        subcontrol-origin: padding;
        subcontrol-position: top left;
        left: 14px;
        top: 8px;
        padding: 0px 2px;
        background: transparent;
        color: {text};
        font-weight: 800;
    }}
    QGroupBox QLabel#Hint,
    QGroupBox QLabel[class="hint"],
    QGroupBox QLabel[class="topInfoMeta"] {{
        background: transparent;
    }}

    QTextEdit, QPlainTextEdit {{
        background: {input_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 8px;
        selection-background-color: {primary_light};
        color: {text};
    }}
    QTextEdit:focus, QPlainTextEdit:focus {{
        border: 1px solid {border_2};
        background: {input_bg};
    }}
    QTextEdit::viewport, QPlainTextEdit::viewport {{
        background: {input_bg};
        border-radius: 10px;
    }}

    /* Runtime drawer and small section selector buttons */
    QPushButton[checkable="true"],
    QPushButton:checked {{
        border-radius: 12px;
    }}
    QFrame#runtimeDrawer {{
        background: {card};
        border: 1px solid {border};
    }}
    QFrame#runtimeDrawer QTabWidget::pane,
    QFrame#runtimeDrawer QAbstractScrollArea {{
        border-radius: 12px;
    }}

    /* De-emphasize focus rectangles / harsh blue borders on passive panes */
    QTableWidget:focus,
    QAbstractScrollArea:focus,
    QGroupBox:focus {{
        border-color: {border};
    }}

    /* Small label strips that previously looked like cut-out white tabs */
    QLabel#PanelTitle,
    QLabel#CardTitle,
    QLabel#InlineTitle {{
        background: transparent;
        border: none;
    }}

    /* Cleaner form cards on the right pages */
    QComboBox, QSpinBox, QLineEdit {{
        min-height: 28px;
    }}
    QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{
        border: 1px solid {primary};
    }}

    /* GX-UI-024/030 consolidated UI styles ---------------------------- */
    QFrame#ResearchNavCard,
    QFrame#MetricCard,
    QLabel#InfoBanner,
    QLabel#WarningBanner,
    QLabel#TruthPreviewBox {{
        background: {card_2};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 8px;
        color: {text};
    }}
    QLabel#WarningBanner {{
        background: {warn_bg};
        color: {warn_text};
        border-color: {warn_border};
    }}
    QLabel#InfoBanner {{
        background: {neutral_bg};
        color: {neutral_text};
        border-color: {neutral_border};
    }}
    QLabel#TruthStatusLabel[tone="good"] {{
        color: {good_text};
        font-weight: 800;
        background: transparent;
    }}
    QLabel#TruthStatusLabel[tone="warning"] {{
        color: {warn_text};
        font-weight: 800;
        background: transparent;
    }}
    QFrame#MetricCard[tone="ok"] {{
        background: {good_bg};
        border: 1px solid {good_border};
        border-radius: 14px;
    }}
    QFrame#MetricCard[tone="warn"] {{
        background: {warn_bg};
        border: 1px solid {warn_border};
        border-radius: 14px;
    }}
    QFrame#MetricCard[tone="neutral"] {{
        background: {neutral_bg};
        border: 1px solid {neutral_border};
        border-radius: 14px;
    }}
    QToolButton#QualityToolButton {{
        background: {card_2};
        border: 1px solid {border_2};
        border-radius: 8px;
        padding: 3px 8px;
        color: {text};
        font-weight: 700;
    }}
    QToolButton#QualityToolButton:checked {{
        background: {primary_light};
        border-color: {primary};
        color: {primary_dark};
    }}
    QLabel#RuntimeSummaryChip {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 10px;
        padding: 4px 9px;
        font-weight: 700;
    }}
    QLabel#NextStepHint {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 12px;
        padding: 7px 10px;
        font-weight: 700;
    }}
    QLabel#NextStepHint[tone="good"] {{
        background: {good_bg};
        color: {good_text};
        border-color: {good_border};
    }}
    QLabel#NextStepHint[tone="warning"] {{
        background: {warn_bg};
        color: {warn_text};
        border-color: {warn_border};
    }}
    QFrame#mainPlotCard {{
        border-radius: 20px;
    }}
    QFrame#runtimeDrawer {{
        border-radius: 14px;
    }}



    /* V0.6.7 quality pass: professional B-scan + inspector + event stream ---- */
    QWidget#mainWorkspacePanel QLabel.topInfoText,
    QLabel[class="topInfoText"] {{
        color: {muted};
        font-weight: 600;
        background: transparent;
    }}
    QLabel[class="topInfoMeta"] {{
        color: {subtle};
        font-weight: 600;
        background: transparent;
    }}
    QWidget#controlPanel QGroupBox#basicMethodCard {{
        background: {card};
        border: 1px solid {border};
        border-radius: 18px;
    }}
    QFrame#MethodInspectorHeader {{
        background: {card_2};
        border: 1px solid {border};
        border-radius: 14px;
    }}
    QLabel#MethodCategoryTag {{
        color: {primary_dark};
        background: {primary_light};
        border: 1px solid {border_2};
        border-radius: 9px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 800;
        max-width: 110px;
    }}
    QLabel#MethodNameLabel {{
        color: {text};
        background: transparent;
        border: none;
        font-size: 14px;
        font-weight: 800;
    }}
    QWidget#InspectorParamPanel {{
        background: {card_3};
        border: 1px solid {border};
        border-radius: 14px;
    }}
    QLabel#ParamFieldLabel {{
        color: {muted};
        background: transparent;
        font-size: 12px;
        font-weight: 700;
    }}
    QLabel#ParamDirtyLabel, QLabel#ApplyFeedbackLabel {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 10px;
        padding: 5px 8px;
        font-size: 12px;
        font-weight: 700;
    }}
    QLabel#ParamDirtyLabel[tone="dirty"], QLabel#ApplyFeedbackLabel[tone="warning"] {{
        background: {warn_bg};
        color: {warn_text};
        border-color: {warn_border};
    }}
    QLabel#ParamDirtyLabel[tone="clean"], QLabel#ApplyFeedbackLabel[tone="neutral"] {{
        background: {neutral_bg};
        color: {neutral_text};
        border-color: {neutral_border};
    }}
    QLabel#ApplyFeedbackLabel[tone="info"] {{
        background: {info_bg};
        color: {info_text};
        border-color: {info_border};
    }}
    QLabel#ApplyFeedbackLabel[tone="success"] {{
        background: {good_bg};
        color: {good_text};
        border-color: {good_border};
    }}
    QLabel#ApplyFeedbackLabel[tone="danger"] {{
        background: {danger_bg};
        color: {danger_text};
        border-color: {danger_border};
    }}
    QWidget#InspectorParamPanel QLineEdit,
    QWidget#InspectorParamPanel QComboBox,
    QWidget#InspectorParamPanel QSpinBox {{
        min-height: 30px;
        border-radius: 10px;
        background: {input_bg};
    }}
    QPushButton[class="basicGhostBtn"] {{
        background: transparent;
        color: {muted};
        border: 1px solid {border};
        border-radius: 11px;
        font-weight: 700;
    }}
    QPushButton[class="basicGhostBtn"]:hover {{
        background: {neutral_bg};
        color: {primary_dark};
        border-color: {border_2};
    }}
    QTextEdit#RuntimeEventLog {{
        background: {card_3};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 7px;
        font-family: "Consolas", "Microsoft YaHei UI", monospace;
        font-size: 12px;
    }}
    QLabel#PlotTitle {{
        font-size: 18px;
        font-weight: 900;
    }}
    QLabel#PlotInfoChip {{
        background: {card_2};
        color: {neutral_text};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 4px 8px;
        font-weight: 700;
    }}
    QFrame#PlotBottomStatusBar {{
        background: {card_3};
        border: 1px solid {border};
        border-radius: 12px;
    }}
    QFrame#MyGPRMark {{
        background: transparent;
        border: none;
    }}
    QLabel#RuntimeSummaryChip[tone="neutral"], QLabel#PlotModeChip[tone="neutral"] {{ background: {neutral_bg}; color: {neutral_text}; border-color: {neutral_border}; }}
    QLabel#RuntimeSummaryChip[tone="info"], QLabel#PlotModeChip[tone="info"] {{ background: {info_bg}; color: {info_text}; border-color: {info_border}; }}
    QLabel#RuntimeSummaryChip[tone="good"], QLabel#PlotModeChip[tone="success"] {{ background: {good_bg}; color: {good_text}; border-color: {good_border}; }}
    QLabel#RuntimeSummaryChip[tone="warning"], QLabel#PlotModeChip[tone="warning"] {{ background: {warn_bg}; color: {warn_text}; border-color: {warn_border}; }}
    QLabel#RuntimeSummaryChip[tone="danger"], QLabel#PlotModeChip[tone="danger"] {{ background: {danger_bg}; color: {danger_text}; border-color: {danger_border}; }}

    /* AutoTune full-theme stylesheet ---------------------------------- */
    QWidget#AutoTuneTuningPage {{
        background: transparent;
        color: {text};
    }}
    QWidget#AutoTuneTuningPage QFrame#AutoTuneHeader,
    QWidget#AutoTuneTuningPage QFrame#PreviewCard {{
        background: {card};
        border: 1px solid {border};
        border-radius: 18px;
    }}
    QWidget#AutoTuneTuningPage QFrame#AutoTuneHeader {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                    stop:0 {card}, stop:1 {card_3});
    }}
    QWidget#AutoTuneTuningPage QLabel#AutoTuneTitle {{
        font-size: 20px;
        font-weight: 800;
        color: {text};
        background: transparent;
    }}
    QWidget#AutoTuneTuningPage QLabel#AutoTuneSubtitle,
    QWidget#AutoTuneTuningPage QLabel#Hint {{
        color: {muted};
        background: transparent;
    }}
    QWidget#AutoTuneTuningPage QLabel#PanelTitle {{
        font-size: 16px;
        font-weight: 800;
        color: {text};
        background: transparent;
    }}
    QWidget#AutoTuneTuningPage QLabel#CardTitle {{
        font-size: 14px;
        font-weight: 800;
        color: {text};
        background: transparent;
    }}
    QWidget#AutoTuneTuningPage QLabel#StatusChip,
    QWidget#AutoTuneTuningPage QLabel#MiniTag {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 11px;
        padding: 5px 9px;
        font-weight: 800;
    }}
    QWidget#AutoTuneTuningPage QLabel#StatusChip[tone="neutral"] {{
        background: {neutral_bg};
        color: {neutral_text};
        border-color: {neutral_border};
    }}
    QWidget#AutoTuneTuningPage QLabel#StatusChip[tone="good"] {{
        background: {good_bg};
        color: {good_text};
        border-color: {good_border};
    }}
    QWidget#AutoTuneTuningPage QLabel#StatusChip[tone="warning"] {{
        background: {warn_bg};
        color: {warn_text};
        border-color: {warn_border};
    }}
    QWidget#AutoTuneTuningPage QLabel#StatusChip[tone="danger"] {{
        background: {danger_bg};
        color: {danger_text};
        border-color: {danger_border};
    }}
    QWidget#AutoTuneTuningPage QLabel#PreviewCanvas {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                    stop:0 {card_3}, stop:1 {card_2});
        border: 1px dashed {border_2};
        border-radius: 14px;
        color: {muted};
        font-size: 13px;
        padding: 14px;
    }}
    QWidget#AutoTuneTuningPage QPushButton#PrimaryButton {{
        background: {primary};
        color: white;
        border: 1px solid {primary};
        border-radius: 11px;
        padding: 6px 10px;
        font-weight: 800;
    }}
    QWidget#AutoTuneTuningPage QPushButton#PrimaryButton:hover {{
        background: {primary_dark};
        color: white;
    }}
    QWidget#AutoTuneTuningPage QPushButton#PrimaryButton:disabled,
    QWidget#AutoTuneTuningPage QPushButton#SecondaryButton:disabled {{
        background: {disabled_bg};
        color: {disabled_text};
        border-color: {border};
    }}
    QWidget#AutoTuneTuningPage QPushButton#SecondaryButton {{
        background: {card};
        color: {text};
        border: 1px solid {border_2};
        border-radius: 11px;
        padding: 6px 10px;
        font-weight: 700;
    }}
    QWidget#AutoTuneTuningPage QPushButton#SecondaryButton:hover {{
        background: {neutral_bg};
        color: {primary_dark};
        border-color: {primary};
    }}
    QWidget#AutoTuneTuningPage QGroupBox {{
        font-weight: 800;
        border: 1px solid {border};
        border-radius: 14px;
        margin-top: 0px;
        padding: 30px 10px 10px 10px;
        background: {card};
        color: {text};
    }}
    QWidget#AutoTuneTuningPage QGroupBox::title {{
        subcontrol-origin: padding;
        subcontrol-position: top left;
        left: 12px;
        top: 8px;
        padding: 0 2px;
        color: {text};
        background: transparent;
    }}
    QWidget#AutoTuneTuningPage QTabWidget::pane {{
        border: 1px solid {border};
        border-radius: 14px;
        background: {card};
        top: -1px;
    }}
    QWidget#AutoTuneTuningPage QTabBar::tab {{
        background: {tab_bg};
        border: 1px solid {border};
        border-bottom: none;
        padding: 8px 11px;
        margin-right: 3px;
        min-width: 56px;
        border-top-left-radius: 11px;
        border-top-right-radius: 11px;
        color: {muted};
        font-weight: 700;
    }}
    QWidget#AutoTuneTuningPage QTabBar::tab:selected {{
        background: {tab_sel};
        color: {primary_dark};
        font-weight: 800;
    }}
    QWidget#AutoTuneTuningPage QAbstractScrollArea {{
        background: {card};
        border: 1px solid {border_2};
        border-radius: 11px;
    }}
    QWidget#AutoTuneTuningPage QAbstractScrollArea::viewport {{
        background: {card};
        color: {text};
    }}
    QWidget#AutoTuneTuningPage QTextEdit,
    QWidget#AutoTuneTuningPage QTableWidget,
    QWidget#AutoTuneTuningPage QComboBox,
    QWidget#AutoTuneTuningPage QSpinBox {{
        border: 1px solid {border_2};
        border-radius: 11px;
        background: {input_bg};
        color: {text};
        selection-background-color: {neutral_bg};
    }}
    QWidget#AutoTuneTuningPage QTextEdit {{
        padding: 7px;
        line-height: 1.35em;
    }}
    QWidget#AutoTuneTuningPage QComboBox,
    QWidget#AutoTuneTuningPage QSpinBox {{
        min-height: 29px;
        padding: 2px 7px;
    }}
    QWidget#AutoTuneTuningPage QComboBox QAbstractItemView {{
        background: {card};
        color: {text};
        border: 1px solid {border_2};
        selection-background-color: {neutral_bg};
    }}
    QWidget#AutoTuneTuningPage QComboBox::drop-down,
    QWidget#AutoTuneTuningPage QSpinBox::up-button,
    QWidget#AutoTuneTuningPage QSpinBox::down-button {{
        background: {tab_bg};
        border-left: 1px solid {border};
        width: 22px;
    }}
    QWidget#AutoTuneTuningPage QCheckBox {{
        spacing: 8px;
        padding: 4px 2px;
        color: {text};
        background: transparent;
    }}
    QWidget#AutoTuneTuningPage QTableWidget {{
        gridline-color: {border};
        alternate-background-color: {tab_bg};
        border: 1px solid {border};
        border-radius: 12px;
        background: {card};
        color: {text};
    }}
    QWidget#AutoTuneTuningPage QTableWidget::item {{
        color: {text};
        background: transparent;
        padding: 4px;
    }}
    QWidget#AutoTuneTuningPage QTableWidget::item:selected {{
        background: {neutral_bg};
        color: {text};
    }}
    QWidget#AutoTuneTuningPage QHeaderView::section {{
        background: {tab_bg};
        color: {text};
        border: none;
        border-right: 1px solid {border};
        border-bottom: 1px solid {border};
        padding: 6px 8px;
        font-weight: 800;
    }}
    QWidget#AutoTuneTuningPage QFrame#CandidateRow {{
        background: {tab_bg};
        border: 1px solid {border};
        border-radius: 12px;
    }}
    QWidget#AutoTuneTuningPage QFrame#CandidateRow:hover {{
        background: {neutral_bg};
        border-color: {primary};
    }}
    QWidget#AutoTuneTuningPage QLabel#CandidateTitle {{
        color: {text};
        font-weight: 800;
        background: transparent;
    }}
    QWidget#AutoTuneTuningPage QLabel#CandidateSubtitle {{
        color: {muted};
        font-size: 12px;
        background: transparent;
    }}
    QWidget#AutoTuneTuningPage QLabel#MethodTag {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 9px;
        padding: 3px 7px;
        font-size: 11px;
        font-weight: 800;
    }}
    QWidget#AutoTuneTuningPage QFrame#RankPanel {{
        background: {card_3};
        border: 1px solid {border};
        border-radius: 12px;
    }}
    QWidget#AutoTuneTuningPage QLabel#InlineTitle {{
        color: {text};
        font-weight: 800;
        background: transparent;
    }}
    QWidget#AutoTuneTuningPage QLabel#FieldLabel {{
        color: {muted};
        font-size: 12px;
        background: transparent;
    }}

    /* Final B-scan workspace interaction polish ------------------------ */
    QFrame#PlotBottomStatusBar {{
        background: {card_2};
        border: 1px solid {border};
        border-radius: 12px;
    }}
    QWidget#PlotToolbarRow {{
        background: transparent;
    }}
    QLabel#PlotInfoChip {{
        min-height: 20px;
        padding: 1px 6px;
        font-size: 11px;
    }}
    QFrame#ProcessingStepperBar {{
        background: {card_3};
        border: 1px solid {border};
        border-radius: 11px;
    }}
    QLabel#ProcessingStepperTitle {{
        color: {muted};
        font-weight: 900;
        padding: 0px 2px;
        font-size: 11px;
        background: transparent;
    }}
    QScrollArea#ProcessingStepperScroll,
    QWidget#ProcessingStepperHost {{
        background: transparent;
        border: none;
    }}
    QLabel#ProcessingStepArrow {{
        color: {muted};
        font-weight: 900;
        background: transparent;
        padding: 0px;
        font-size: 10px;
    }}
    QPushButton#ProcessingStepChip {{
        background: {card_2};
        color: {muted};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 2px 7px;
        font-size: 11px;
        font-weight: 800;
    }}
    QPushButton#ProcessingStepChip:hover {{
        background: {neutral_bg};
        color: {primary_dark};
        border-color: {border_2};
    }}
    QPushButton#ProcessingStepChip[active="true"] {{
        background: {info_bg};
        color: {info_text};
        border-color: {info_border};
    }}
    QPushButton#ProcessingStepChip[current="true"] {{
        font-weight: 900;
    }}
    QPushButton#ProcessingStepChip[active="true"][current="true"] {{
        background: {good_bg};
        color: {good_text};
        border-color: {good_border};
    }}
    QPushButton#ProcessingStepChip[status="raw"] {{
        background: {neutral_bg};
        color: {neutral_text};
        border-color: {neutral_border};
    }}
    QPushButton#ProcessingStepChip[status="applied"] {{
        background: {card_2};
        color: {text};
        border-color: {border_2};
    }}
    QPushButton#ProcessingStepChip[status="current"] {{
        background: {good_bg};
        color: {good_text};
        border-color: {good_border};
    }}
    QPushButton#ProcessingStepChip[status="viewing"],
    QPushButton#ProcessingStepChip[status="viewing_warning"] {{
        background: {info_bg};
        color: {info_text};
        border-color: {info_border};
    }}
    QPushButton#ProcessingStepChip[status="warning"] {{
        background: {warn_bg};
        color: {warn_text};
        border-color: {warn_border};
    }}
    QPushButton#ProcessingStepChip[status="pruned"] {{
        background: {disabled_bg};
        color: {disabled_text};
        border-color: {border};
    }}
    QLabel#ProcessingStepDetail {{
        background: {card_2};
        color: {muted};
        border: 1px solid {border};
        border-radius: 9px;
        padding: 2px 7px;
        font-size: 10px;
        font-weight: 700;
    }}
    QFrame#ProcessingStepItem {{
        background: transparent;
        border: none;
    }}
    QPushButton#ProcessingStepChip[compareSelected="true"] {{
        border-color: {primary_dark};
    }}
    QPushButton#ProcessingStepSelectDot {{
        background: transparent;
        color: {muted};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 0px;
        font-size: 10px;
        font-weight: 900;
    }}
    QPushButton#ProcessingStepSelectDot:hover {{
        background: {info_bg};
        color: {info_text};
        border-color: {info_border};
    }}
    QPushButton#ProcessingStepSelectDot[selected="true"] {{
        background: {primary_light};
        color: {primary_dark};
        border-color: {primary_dark};
    }}
    QPushButton#ProcessingStepSelectDot:disabled {{
        background: transparent;
        color: {disabled_text};
        border-color: {border};
    }}
    QFrame#ProcessingCompareTray {{
        background: transparent;
        border: none;
    }}
    QLabel#ProcessingCompareTrayLabel {{
        background: transparent;
        color: {muted};
        padding: 0px 1px;
        font-size: 10px;
        font-weight: 800;
    }}
    QPushButton#ProcessingStepperAction {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 9px;
        padding: 1px 5px;
        font-size: 10px;
        font-weight: 900;
    }}
    QPushButton#ProcessingStepperAction:hover {{
        background: {info_bg};
        color: {info_text};
        border-color: {info_border};
    }}
    QPushButton#ProcessingStepperAction:checked {{
        background: {primary_light};
        color: {primary_dark};
        border-color: {primary_dark};
    }}
    QPushButton#ProcessingStepperAction:disabled {{
        background: {disabled_bg};
        color: {disabled_text};
        border-color: {border};
    }}

    /* Daily processing modern card groupboxes -------------------------- */
    QGroupBox[cardStyle="modern"] {{
        background: {card};
        border: 1px solid {border};
        border-radius: 16px;
        margin-top: 0px;
        padding-top: 32px;
    }}
    QGroupBox[cardStyle="modern"]::title {{
        subcontrol-origin: padding;
        subcontrol-position: top left;
        left: 14px;
        top: 8px;
        padding: 0px 2px;
        background: transparent;
        color: {text};
        font-weight: 800;
    }}

    /* Evidence checklist ----------------------------------------------- */
    QGroupBox#EvidenceChecklistCard {{
        background: {card};
        border: 1px solid {border};
        border-radius: 16px;
        margin-top: 0px;
        padding-top: 32px;
    }}
    QGroupBox#EvidenceChecklistCard::title {{
        subcontrol-origin: padding;
        subcontrol-position: top left;
        left: 14px;
        top: 8px;
        padding: 0px 2px;
        background: transparent;
        color: {text};
        font-weight: 800;
    }}
    QLabel#EvidenceCheckChip {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 11px;
        padding: 5px 9px;
        font-weight: 700;
    }}
    QLabel#EvidenceCheckChip[tone="warning"] {{
        background: {warn_bg};
        color: {warn_text};
        border-color: {warn_border};
    }}


    /* V0.6.6 calm minimal visual refresh ----------------------------- */
    QWidget#appCentralRoot {{
        background: {bg};
    }}
    QWidget#mainWorkspacePanel,
    QWidget#controlPanel,
    QWidget#basicFlowRoot {{
        background: transparent;
    }}
    QWidget#controlPanelShell {{
        background: transparent;
        border: none;
    }}
    QWidget#topInfoBar {{
        background: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 3px 8px;
    }}
    QWidget#topInfoBar QLabel,
    QLabel[class="topInfoText"] {{
        color: {muted};
        background: transparent;
        font-weight: 600;
    }}
    QLabel[class="topInfoMeta"] {{
        color: {subtle};
        background: transparent;
        font-size: 12px;
    }}
    QSplitter#mainSplitter::handle {{
        background: transparent;
        width: 6px;
        margin: 5px 1px;
        border-radius: 3px;
    }}
    QSplitter#mainSplitter::handle:hover {{
        background: {border_2};
    }}
    QWidget#basicFlowRoot QLabel[class="sectionTitle"] {{
        color: {text};
        background: transparent;
        font-size: 19px;
        font-weight: 900;
        padding: 2px 0px 4px 0px;
    }}
    QGroupBox#basicActionCard,
    QGroupBox#basicMethodCard,
    QGroupBox#basicStatusCard {{
        background: {card};
        border: 1px solid {border};
        border-radius: 18px;
        margin-top: 0px;
        padding-top: 32px;
    }}
    QGroupBox#basicActionCard::title,
    QGroupBox#basicMethodCard::title,
    QGroupBox#basicStatusCard::title {{
        subcontrol-origin: padding;
        subcontrol-position: top left;
        left: 14px;
        top: 8px;
        color: {text};
        background: transparent;
        font-weight: 900;
    }}
    QWidget#basicFlowRoot QLabel[class="hintText"] {{
        color: {subtle};
        background: transparent;
        font-size: 12px;
    }}
    QWidget#basicFlowRoot QLabel[class="statusChip"] {{
        background: {primary_light};
        color: {primary_dark};
        border: 1px solid {border_2};
        border-radius: 11px;
        padding: 6px 9px;
        font-weight: 800;
    }}
    SplitActionButton#basicImportButton,
    SplitActionButton#basicApplyButton,
    QPushButton#basicImportButton,
    QPushButton#basicApplyButton {{
        background: {primary};
        color: white;
        border: 1px solid {primary};
        border-radius: 12px;
        min-height: 36px;
        font-weight: 900;
    }}
    SplitActionButton#basicImportButton:hover,
    SplitActionButton#basicApplyButton:hover,
    QPushButton#basicImportButton:hover,
    QPushButton#basicApplyButton:hover {{
        background: {primary_dark};
        border-color: {primary_dark};
        color: white;
    }}
    QPushButton[class="basicGhostBtn"],
    PushButton[class="basicGhostBtn"] {{
        background: {card_2};
        color: {text};
        border: 1px solid {border_2};
        border-radius: 12px;
        min-height: 34px;
        font-weight: 750;
    }}
    QPushButton[class="basicGhostBtn"]:hover,
    PushButton[class="basicGhostBtn"]:hover {{
        background: {primary_light};
        color: {primary_dark};
        border-color: {primary};
    }}
    QPushButton#btnCancel:enabled {{
        color: {danger_text};
        border-color: {danger_border};
        background: {danger_bg};
    }}
    QComboBox#methodCombo {{
        background: {input_bg};
        border: 1px solid {border_2};
        border-radius: 12px;
        min-height: 34px;
        padding-left: 9px;
        font-weight: 700;
    }}
    QTextEdit#basicInfoLog {{
        background: {card_3};
        border: 1px solid {border};
        border-radius: 12px;
        color: {muted};
        padding: 8px;
        font-size: 12px;
    }}
    QGroupBox#basicStatusCard {{
        background: {card_2};
        border-color: {border};
    }}
    QFrame#mainPlotCard {{
        background: {card};
        border: 1px solid {border};
        border-radius: 22px;
    }}
    QFrame#PlotCardHeader {{
        background: transparent;
        border: none;
        min-height: 32px;
    }}
    QLabel#PlotTitle {{
        color: {text};
        font-size: 18px;
        font-weight: 900;
        background: transparent;
    }}
    QLabel#PlotStatusChip,
    QLabel#PlotInfoChip,
    QLabel#RuntimeSummaryChip,
    QLabel#PlotModeChip {{
        background: {card_2};
        color: {muted};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 4px 9px;
        font-weight: 800;
    }}
    QLabel#PlotStatusChip[tone="good"] {{
        background: {good_bg};
        color: {good_text};
        border-color: {good_border};
    }}
    QLabel#PlotStatusChip[tone="neutral"] {{
        background: {card_2};
        color: {muted};
        border-color: {border};
    }}
    QFrame#PlotBottomStatusBar {{
        background: {card_3};
        border: 1px solid {border};
        border-radius: 14px;
    }}
    QToolBar#mainPlotToolbar,
    QFrame#mainPlotToolbar {{
        background: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 4px;
        spacing: 4px;
    }}
    QToolButton {{
        border-radius: 10px;
        padding: 5px;
    }}
    QToolButton:hover {{
        background: {primary_light};
        color: {primary_dark};
        border-color: {primary};
    }}
    QFrame#runtimeDrawer {{
        background: {card};
        border: 1px solid {border};
        border-radius: 16px;
    }}
    QGroupBox#EvidenceChecklistCard,
    QGroupBox#basicActionCard,
    QGroupBox#basicMethodCard,
    QGroupBox#basicStatusCard,
    QFrame#mainPlotCard,
    QFrame#runtimeDrawer {{
        selection-background-color: {primary_light};
    }}


    /* V0.7.0 brand and report export polish -------------------------- */
    QLabel[class="sectionTitle"] {{
        color: {text};
        font-size: 18px;
        font-weight: 850;
        background: transparent;
    }}
    QLabel[class="hintText"] {{
        color: {muted};
        font-size: 12px;
        background: transparent;
    }}
    QFrame#EvidenceHeroCard {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {card}, stop:1 {card_2});
        border: 1px solid {border};
        border-radius: 18px;
    }}
    QLabel#EvidenceHeroMark {{
        background: {primary_light};
        color: {primary_dark};
        border: 1px solid {border_2};
        border-radius: 15px;
        font-weight: 900;
        letter-spacing: 0.5px;
    }}
    QLabel#EvidenceHeroBadge {{
        background: {neutral_bg};
        color: {neutral_text};
        border: 1px solid {neutral_border};
        border-radius: 12px;
        padding: 5px 10px;
        font-weight: 800;
    }}

    """
