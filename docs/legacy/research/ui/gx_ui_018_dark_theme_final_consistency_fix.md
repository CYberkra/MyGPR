# GX-UI-018 Dark Theme Final Consistency Fix

This patch fixes the dark-theme inconsistency where qfluentwidgets could report `LIGHT` while the effective Qt window palette was dark.

Changes:

- Added `ui.theme.is_dark_ui(...)` to detect the effective dark UI via Qt palette fallback.
- Updated global polish QSS to use effective dark detection rather than trusting the theme-manager label only.
- Updated AutoTune local stylesheet to use effective dark detection.
- Strengthened `QAbstractScrollArea`, `QTableWidget`, headers, table items, combo/spin dropdown controls, and selection colors for dark mode.
- Updated main B-scan direct theme and Matplotlib figure theme to use effective dark detection.

Boundary:

- No processing algorithm changes.
- No AutoTune production scoring changes.
- No gprMax or Evidence changes.
- No PyVista / 3D viewer changes.
