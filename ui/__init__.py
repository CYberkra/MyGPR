"""UI modules for MyGPR."""

from ui.matplotlib_fonts import configure_matplotlib_cjk_fonts

# Configure Matplotlib's separate font fallback as soon as any UI page is
# imported.  This keeps direct page tests and embedded workbench plots from
# rendering Chinese labels with missing-glyph boxes.
configure_matplotlib_cjk_fonts()

__all__ = ["configure_matplotlib_cjk_fonts"]
