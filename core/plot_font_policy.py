#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless Matplotlib CJK font policy for engineering exports.

This module belongs to ``core`` so report and GIS workers can configure fonts
without importing Qt or UI packages.  It registers common TTC/TTF/OTF CJK
fonts before selecting a concrete family, which is required in stripped-down
Linux field-validation environments where Matplotlib's initial cache may omit
system TTC collections.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


_CANDIDATES = (
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Serif CJK SC",
    "Source Han Sans SC",
    "Source Han Serif SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "AR PL UMing CN",
    "AR PL KaitiM GB",
    "Arial Unicode MS",
)


@lru_cache(maxsize=1)
def configure_matplotlib_cjk_fonts() -> str:
    """Configure Matplotlib and return the selected CJK-capable family."""
    import matplotlib
    from matplotlib import font_manager as fm

    likely_paths = {
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
        "/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf",
    }
    for extension in ("ttf", "otf", "ttc"):
        try:
            likely_paths.update(fm.findSystemFonts(fontext=extension))
        except (OSError, RuntimeError, ValueError):
            pass

    needles = (
        "notosanscjk",
        "notoserifcjk",
        "sourcehansans",
        "sourcehanserif",
        "wenquanyi",
        "simhei",
        "yahei",
        "uming",
        "gbsn",
        "gkai",
    )
    for raw_path in sorted(likely_paths):
        path = Path(raw_path)
        compact = str(path).lower().replace("-", "").replace("_", "").replace(" ", "")
        if not path.is_file() or not any(item in compact for item in needles):
            continue
        try:
            fm.fontManager.addfont(str(path))
        except (OSError, RuntimeError, ValueError):
            continue

    installed = {item.name for item in fm.fontManager.ttflist}
    selected = next((name for name in _CANDIDATES if name in installed), "DejaVu Sans")
    chain = [selected]
    if selected != "DejaVu Sans":
        chain.append("DejaVu Sans")
    matplotlib.rcParams["font.family"] = chain
    matplotlib.rcParams["font.sans-serif"] = chain
    matplotlib.rcParams["axes.unicode_minus"] = False
    return selected


__all__ = ["configure_matplotlib_cjk_fonts"]
