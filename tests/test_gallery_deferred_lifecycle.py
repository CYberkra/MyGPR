"""Deferred gallery deletion must remain safe when Python collects cycles first."""
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import pytest


def test_gallery_timers_remain_safe_during_gc_before_deferred_deletion(tmp_path):
    pytest.importorskip('PyQt6')
    environment = dict(os.environ, QT_QPA_PLATFORM='offscreen',
                       MYGPR_RUNTIME_ROOT=str(tmp_path), MYGPR_LOG_DIR=str(tmp_path / 'logs'),
                       MYGPR_SETTINGS_FILE=str(tmp_path / 'settings.json'))
    program = dedent('''
        import gc
        import sys
        from types import SimpleNamespace
        import numpy as np
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QApplication
        from ui import constants
        from ui.theme_helpers import apply_theme
        from ui.widgets.bscan_gallery import BScanGallery

        errors = []
        sys.excepthook = lambda kind, error, tb: errors.append(str(error))
        app = QApplication([])
        app.setStyle('Fusion')
        apply_theme(constants.THEME_LIGHT)
        bundle = SimpleNamespace(
            matrix=np.ones((8, 6), np.float32), vmin=0., vmax=1., title='source',
            trace_axis_m=None, sample_axis=None, sample_axis_label='',
            trace_count=6, sample_count=8, trace_elevation_m=None, depth_axis_m=None)
        sources = [dict(key=str(i), title=str(i), bundle=bundle) for i in range(5)]
        for rebuild in (False, True):
            for _ in range(3):
                gallery = BScanGallery()
                gallery.set_sources(sources, '0')
                if rebuild:
                    gallery.set_sources(sources[:2], '0')
                gallery.show()
                gallery.close()
                gallery.deleteLater()
                del gallery
                gc.collect()
                app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        gc.collect()
        print('Deferred gallery callback errors:', errors, file=sys.stderr)
        assert errors == [], errors
        # A surviving gallery still completes its staggered fade normally.
        gallery = BScanGallery()
        gallery.set_sources(sources[:2], '0')
        gallery.show()
        QTest.qWait(400)
        assert all(card.graphicsEffect().opacity() == 1. for card in gallery.cards())
        gallery.close()
        gallery.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        assert errors == [], errors
    ''')
    result = subprocess.run([sys.executable, '-c', program], env=environment,
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
