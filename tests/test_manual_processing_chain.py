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
