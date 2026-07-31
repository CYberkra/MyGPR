from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_windows_launcher_enables_fault_diagnostics_and_stable_graphics_env() -> None:
    text = (ROOT / "scripts" / "mygpr_windows_launcher.py").read_text(encoding="utf-8")
    assert "PYTHONFAULTHANDLER" in text
    assert "-X" in text and "faulthandler" in text
    assert "QT_OPENGL" in text and "software" in text
    assert "MPLCONFIGDIR" in text


def test_matplotlib_font_chain_does_not_force_plain_noto_sans() -> None:
    text = (ROOT / "ui" / "matplotlib_fonts.py").read_text(encoding="utf-8")
    assert 'for fallback in ("Noto Sans", "DejaVu Sans")' not in text
    assert 'append("DejaVu Sans")' in text
