#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Startup import regression tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap


def test_app_qt_import_does_not_eager_load_known_blocking_modules():
    """Importing the GUI shell must not load data/science backends eagerly."""
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    code = textwrap.dedent(
        """
        import json
        import os
        import sys
        sys.path.insert(0, os.getcwd())
        import app_qt
        print(json.dumps({
            "pandas": "pandas" in sys.modules,
            "h5py": "h5py" in sys.modules,
            "pywt": "pywt" in sys.modules,
        }, sort_keys=True))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload == {"h5py": False, "pandas": False, "pywt": False}
