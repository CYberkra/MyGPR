from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mygpr.domain.processing.models import PipelineDefinition, PipelineStep
from mygpr.interfaces.backend import MyGPRBackend
from scripts.verify_yingshan_dataset import main as verify_full_yingshan

pytestmark = [
    pytest.mark.industrial,
    pytest.mark.scientific_validation,
    pytest.mark.requirement("REQ-PROC-002", "REQ-SCI-001", "REQ-SCI-002", "REQ-SCI-003"),
    pytest.mark.risk("RISK-SCIENTIFIC-REGRESSION"),
    pytest.mark.level("acceptance"),
]

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "yingshan_real_v1"


def _quantized_sha256(data: np.ndarray, decimals: int) -> str:
    quantized = np.round(np.asarray(data, dtype=np.float64), decimals=decimals).astype("<f4")
    return hashlib.sha256(np.ascontiguousarray(quantized).tobytes()).hexdigest()


def _pipeline(row: dict[str, object]) -> PipelineDefinition:
    return PipelineDefinition(
        name="Yingshan deterministic preprocessing v1",
        steps=tuple(PipelineStep(str(step["method_id"]), dict(step["params"])) for step in row["preprocessing_v1"]["pipeline"]),
    )


def test_yingshan_subset_hashes_metadata_and_processing_fingerprints_are_stable() -> None:
    manifest = json.loads((FIXTURE / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "yingshan_airborne_gpr_2025_v1"
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        with np.load(FIXTURE / "yingshan_trace_subset_v1.npz", allow_pickle=False) as archive:
            assert set(manifest["lines"]) == {"3", "6", "7", "9", "L1", "X1"}
            for line_id, row in manifest["lines"].items():
                matrix = np.asarray(archive[f"line_{line_id}_matrix"], dtype=np.float32)
                assert list(matrix.shape) == row["subset_shape"]
                assert hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest() == row["subset_sha256"]
                assert np.isfinite(matrix).all()
                result = backend.processing.execute_pipeline(
                    matrix,
                    _pipeline(row),
                    header_info={"total_time_ns": row["time_window_ns"], "length_m": row["trace_interval_m"] * (row["trace_count"] - 1)},
                )
                expected = row["preprocessing_v1"]
                assert _quantized_sha256(result.data, int(expected["quantized_decimals"])) == expected["quantized_sha256"]
                actual_rms = float(np.sqrt(np.mean(np.asarray(result.data, dtype=np.float64) ** 2)))
                assert actual_rms == pytest.approx(float(expected["statistics"]["rms"]), rel=1e-6, abs=1e-10)
                assert [item.metadata.get("implementation_version") for item in result.step_results] == [item["implementation_version"] for item in expected["step_versions"]]
    finally:
        backend.shutdown()


def test_published_yingshan_borehole_baseline_meets_one_metre_criterion() -> None:
    payload = json.loads((FIXTURE / "borehole_truth.json").read_text(encoding="utf-8"))
    maximum = float(payload["criterion"]["maximum_m"])
    errors = []
    for row in payload["observations"]:
        error = abs(float(row["published_gpr_depth_m"]) - float(row["borehole_bedrock_depth_m"]))
        errors.append(error)
        assert error < maximum, row
    assert max(errors) == pytest.approx(0.3)
    assert len(errors) >= 5


def test_full_yingshan_files_match_frozen_hashes_when_external_asset_is_available(tmp_path: Path) -> None:
    source = os.environ.get("MYGPR_YINGSHAN_DATA")
    if not source:
        pytest.skip("set MYGPR_YINGSHAN_DATA to the full Yingshan directory or zip")
    output = tmp_path / "full-validation.json"
    assert verify_full_yingshan([source, "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
