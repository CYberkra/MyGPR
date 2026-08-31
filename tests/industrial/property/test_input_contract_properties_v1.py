from __future__ import annotations

import json
from pathlib import Path
import random
import string
import zipfile

import numpy as np
import pytest

from core.field_project_models import FieldProjectManifest
from core.field_project_operations import FieldProjectOperationError, restore_project_archive
from core.gpr_data_model import detect_mygpr_airborne_sidecar_csv
from mygpr.application.processing.validation import validate_parameters
from mygpr.domain.common.errors import ParameterValidationError
from mygpr.domain.processing.models import ProcessingMethodDescriptor

pytestmark = [
    pytest.mark.industrial,
    pytest.mark.property,
    pytest.mark.requirement("REQ-PROC-001", "REQ-STO-003"),
    pytest.mark.risk("RISK-PARAMETER-SILENT-IGNORE", "RISK-PATH-TRAVERSAL", "RISK-TEST-INEFFECTIVE"),
    pytest.mark.level("unit"),
]


def _descriptor() -> ProcessingMethodDescriptor:
    return ProcessingMethodDescriptor(
        method_id="contract",
        name="contract",
        parameter_schema={
            "window": {"type": "int", "min": 1, "max": 101},
            "gain_min": {"type": "float", "min": 0.0},
            "gain_max": {"type": "float", "min": 0.0},
            "solver": {"type": "str", "choices": ["auto", "exact"]},
        },
    )


def test_parameter_unknown_range_type_choice_and_cross_field_mutation_oracles() -> None:
    descriptor = _descriptor()
    assert validate_parameters(descriptor, {"window": np.int64(23), "gain_min": 1, "gain_max": 6.0, "solver": "auto"}) == {
        "window": 23, "gain_min": 1, "gain_max": 6.0, "solver": "auto"
    }
    with pytest.raises(ParameterValidationError, match="unknown parameter"):
        validate_parameters(descriptor, {"windwo": 23})
    with pytest.raises(ParameterValidationError, match="below minimum"):
        validate_parameters(descriptor, {"window": 0})
    with pytest.raises(ParameterValidationError, match="exceeds maximum"):
        validate_parameters(descriptor, {"window": 102})
    with pytest.raises(ParameterValidationError, match="must be int"):
        validate_parameters(descriptor, {"window": 23.5})
    with pytest.raises(ParameterValidationError, match="must be one of"):
        validate_parameters(descriptor, {"solver": "unsafe"})
    with pytest.raises(ParameterValidationError, match="must not exceed"):
        validate_parameters(descriptor, {"gain_min": 7.0, "gain_max": 6.0})
    with pytest.raises(ParameterValidationError, match="must be int"):
        validate_parameters(descriptor, {"window": True})

    # Explicit compatibility escape is bounded and retains unknown private/legacy keys.
    assert validate_parameters(descriptor, {"legacy": 1}, reject_unknown=False) == {"legacy": 1}
    assert validate_parameters(descriptor, {"_internal": "ok"}) == {"_internal": "ok"}

    time_descriptor = ProcessingMethodDescriptor(
        method_id="time", name="time",
        parameter_schema={
            "time_start_ns": {"type": "float", "min": 0.0},
            "time_end_ns": {"type": "float", "min": 0.0},
            "low_freq_mhz": {"type": "float"},
            "high_freq_mhz": {"type": "float"},
        },
    )
    assert validate_parameters(time_descriptor, {"time_start_ns": 10.0, "time_end_ns": 0.0})["time_end_ns"] == 0.0
    assert validate_parameters(time_descriptor, {"low_freq_mhz": 10.0, "high_freq_mhz": 20.0})["high_freq_mhz"] == 20.0
    with pytest.raises(ParameterValidationError, match="low_freq_mhz"):
        validate_parameters(time_descriptor, {"low_freq_mhz": 20.0, "high_freq_mhz": 20.0})


def test_airborne_header_parser_fuzz_is_total_and_only_accepts_complete_positive_headers(tmp_path: Path) -> None:
    rng = random.Random(20260722)
    valid = tmp_path / "valid.csv"
    valid.write_text(
        "Number of Samples = 501,,,,\nTime windows (ns) = 700,,,,\nNumber of Traces = 16,,,,\nTrace interval (m) = 0.1,,,,\n",
        encoding="utf-8",
    )
    assert detect_mygpr_airborne_sidecar_csv(valid) is not None
    alphabet = string.ascii_letters + string.digits + "=,.-+_ /\\"
    for index in range(150):
        path = tmp_path / f"fuzz-{index}.csv"
        lines = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80))) for _ in range(rng.randint(0, 8))]
        path.write_text("\n".join(lines), encoding="utf-8")
        result = detect_mygpr_airborne_sidecar_csv(path)
        assert result is None or (int(result["sample_count"]) > 0 and int(result["trace_count"]) > 0)


def test_manifest_identifier_and_zip_restore_traversal_fuzz_are_rejected(tmp_path: Path) -> None:
    unsafe_ids = ["../escape", "..\\escape", "/root", "C:\\outside", "a/b", "a\\b", "\x00bad"]
    for line_id in unsafe_ids:
        payload = {"schema": "mygpr.field_project.v3", "project_id": "project", "name": "bad", "lines": [{"line_id": line_id, "name": "bad"}]}
        with pytest.raises((ValueError, TypeError)):
            FieldProjectManifest.from_dict(payload)

    traversal_names = ["../escape", "../../a", "/absolute", "folder/../../../escape", "C:/outside", "folder\\..\\..\\escape"]
    for index, name in enumerate(traversal_names):
        archive = tmp_path / f"bad-{index}.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("backup_manifest.json", json.dumps({"files": []}))
            handle.writestr(name, "bad")
        with pytest.raises(FieldProjectOperationError):
            restore_project_archive(archive, tmp_path / f"restore-{index}")
