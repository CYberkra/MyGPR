#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matplotlib font fallback helpers for MyGPR UI plots.

Qt widgets use the application font, but Matplotlib manages a separate font
list.  The workbench pages contain Chinese labels in figure titles and empty
states, so importing a page directly in tests or helper tools must still
configure a CJK-capable fallback before the first draw.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def preferred_cjk_font_candidates() -> list[str]:
    """Return ordered CJK-friendly family names used across MyGPR."""
    return [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "Source Han Sans SC",
        "Source Han Serif SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]


@lru_cache(maxsize=1)
def _discover_cjk_font_names() -> tuple[str, ...]:
    """Register common TTF/OTF/TTC CJK fonts and return their family names."""
    try:
        from matplotlib import font_manager as fm
    except Exception:
        return ()

    keywords = (
        "notosanscjk",
        "noto sans cjk",
        "notoserifcjk",
        "noto serif cjk",
        "sourcehansans",
        "source han sans",
        "sourcehanserif",
        "source han serif",
        "wenquanyi",
        "simhei",
        "msyh",
        "yahei",
        "droidsansfallback",
        "sarasa",
    )
    extra_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    ]

    paths: set[str] = set()
    for fontext in ("ttf", "otf", "ttc"):
        try:
            paths.update(fm.findSystemFonts(fontext=fontext))
        except Exception:
            continue
    paths.update(path for path in extra_paths if Path(path).is_file())

    discovered: list[str] = []
    for path in sorted(paths):
        normalized = path.lower().replace("_", "").replace("-", "")
        if not any(keyword.replace(" ", "") in normalized for keyword in keywords):
            continue
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
        except Exception:
            continue
        if name and name not in discovered:
            discovered.append(name)
    return tuple(discovered)


@lru_cache(maxsize=1)
def configure_matplotlib_cjk_fonts() -> tuple[str, ...]:
    """Install a stable Matplotlib fallback chain and return chosen families.

    The helper is intentionally safe to call at module import time.  It does not
    force a Matplotlib backend and only changes font rcParams.
    """
    try:
        import matplotlib
        from matplotlib import font_manager as fm
    except Exception:
        return ()

    discovered = list(_discover_cjk_font_names())
    try:
        installed = {font.name for font in fm.fontManager.ttflist}
    except Exception:
        installed = set()

    ordered: list[str] = []
    for name in preferred_cjk_font_candidates() + discovered:
        if name not in ordered and name in installed:
            ordered.append(name)
    # Do not append non-installed fonts such as plain "Noto Sans".  On Windows
    # that can trigger thousands of Matplotlib findfont warnings during startup,
    # and in some PyQt/Matplotlib combinations it has been observed to end in a
    # native abort before Python can emit a traceback.  Keep only discovered
    # CJK-capable fonts plus Matplotlib's bundled DejaVu Sans fallback.
    if "DejaVu Sans" not in ordered:
        ordered.append("DejaVu Sans")

    try:
        import logging
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        # Matplotlib's ``font.family`` expects generic family groups or concrete
        # family names.  Keep both entries aligned so axes labels, legends and
        # titles all resolve through the same CJK-capable fallback chain.
        matplotlib.rcParams["font.family"] = ordered
        matplotlib.rcParams["font.sans-serif"] = ordered
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        return tuple(ordered)
    return tuple(ordered)


__all__ = ["configure_matplotlib_cjk_fonts", "preferred_cjk_font_candidates"]
