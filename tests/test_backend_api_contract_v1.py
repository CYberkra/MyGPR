from __future__ import annotations

from scripts.check_backend_api_contract import build_contract
import json
from pathlib import Path


def test_backend_api_v1_snapshot_is_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "config" / "backend_api_v1.json").read_text(encoding="utf-8"))
    assert build_contract() == expected


def test_job_dtos_are_json_serializable_without_embedding_matrices() -> None:
    import numpy as np

    from mygpr.application.jobs.models import JobEvent, JobEventType, JobSnapshot, JobStatus

    event = JobEvent.create(
        job_id="job",
        event_type=JobEventType.WARNING,
        sequence=1,
        payload={"matrix": np.ones((4, 3), dtype=np.float32)},
    )
    snapshot = JobSnapshot(
        job_id="job",
        title="demo",
        status=JobStatus.COMPLETED,
        result=np.ones((4, 3), dtype=np.float32),
        error_details={"sample": np.float32(1.25)},
    )
    event_payload = event.to_dict()
    snapshot_payload = snapshot.to_dict()
    json.dumps(event_payload)
    json.dumps(snapshot_payload)
    assert event_payload["payload"]["matrix"]["shape"] == [4, 3]
    assert snapshot_payload["result"]["shape"] == [4, 3]
    assert snapshot_payload["error_details"]["sample"] == 1.25
