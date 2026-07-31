from __future__ import annotations

from core.gpr_data_model import GPRDataSet
from core.manual_processing_chain import ManualProcessingSession


def test_manual_processing_session_append_undo_and_reset() -> None:
    data = GPRDataSet.synthetic("L03", rows=96, cols=72, length_m=36.0)
    session = ManualProcessingSession(data)

    out1, manifest1 = session.append_step("dewow", {"window": 11})
    assert session.step_count == 1
    assert session.current_dataset is out1
    assert manifest1["method_id"] == "dewow"
    assert session.steps[-1].input_shape == tuple(data.matrix.shape)

    out2, manifest2 = session.append_step("subtracting_average_2D", {"ntraces": 9})
    assert session.step_count == 2
    assert session.current_dataset is out2
    assert manifest2["method_id"] == "subtracting_average_2D"
    assert session.steps[-1].input_shape == tuple(out1.matrix.shape)

    payload = session.build_save_payload("subtracting_average_2D", {"ntraces": 9})
    assert payload["manifest"]["processing_mode"] == "manual_step_chain"
    assert payload["manifest"]["chain_step_count"] == 2
    assert len(payload["manifest"]["chain_steps"]) == 2

    assert session.undo_last_step() is True
    assert session.step_count == 1
    assert session.current_dataset is out1

    assert session.reset_to_original() is True
    assert session.step_count == 0
    assert session.current_dataset is data


def test_manual_processing_session_summary_text() -> None:
    data = GPRDataSet.synthetic("L01", rows=64, cols=48, length_m=20.0)
    session = ManualProcessingSession(data)
    assert "原始" in session.summary_text()
    session.append_step("dewow", {"window": 7})
    assert "1 个处理步骤" in session.summary_text()


def test_recompute_middle_step_rebuilds_downstream_chain_atomically() -> None:
    import numpy as np

    data = GPRDataSet.synthetic("L06", rows=96, cols=72, length_m=36.0)
    session = ManualProcessingSession(data)
    session.append_step("dewow", {"window": 9})
    session.append_step("subtracting_average_2D", {"ntraces": 7})
    session.append_step("agcGain", {"window": 21})
    before = np.asarray(session.current_dataset.matrix).copy()

    result, manifest = session.recompute_from_step(
        2,
        method_id="median_background_2D",
        params={"ntraces": 9},
    )

    assert session.step_count == 3
    assert session.steps[1].method_id == "median_background_2D"
    assert session.steps[2].method_id == "agcGain"
    assert manifest["method_id"] == "agcGain"
    assert result is session.current_dataset
    assert not np.allclose(before, np.asarray(result.matrix))


def test_recompute_failure_keeps_last_complete_chain() -> None:
    import numpy as np
    import pytest

    data = GPRDataSet.synthetic("L07", rows=64, cols=48, length_m=20.0)
    session = ManualProcessingSession(data)
    session.append_step("dewow", {"window": 7})
    session.append_step("subtracting_average_2D", {"ntraces": 5})
    before_methods = [step.method_id for step in session.steps]
    before = np.asarray(session.current_dataset.matrix).copy()

    with pytest.raises(Exception):
        session.recompute_from_step(1, method_id="not_a_real_processing_method", params={})

    assert [step.method_id for step in session.steps] == before_methods
    assert session.step_count == 2
    assert np.allclose(before, np.asarray(session.current_dataset.matrix))
