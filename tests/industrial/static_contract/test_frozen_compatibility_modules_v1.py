from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.industrial,
    pytest.mark.static_contract,
    pytest.mark.requirement("REQ-API-002"),
    pytest.mark.risk("RISK-API-DRIFT"),
    pytest.mark.level("contract"),
]


def test_frozen_compatibility_modules_remain_present_until_frontend_cutover() -> None:
    expected = (
        "core/processing_engine.py",
        "PythonModule/motion_compensation_v2.py",
        "PythonModule/motion_compensation_core.py",
        "PythonModule/motion_compensation_height.py",
        "PythonModule/motion_compensation_speed.py",
        "PythonModule/motion_compensation_attitude.py",
    )
    missing = [path for path in expected if not Path(path).is_file()]
    assert not missing, missing
