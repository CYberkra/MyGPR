"""Automatic windows classify partial changes without claiming hardware faults."""

import numpy as np
import pytest
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor


def run(data, header=None, metadata=None, **params):
    return NativeProcessingExecutor().execute(
        ProcessingRequest(
            data=data,
            method_id="short_trace_repair",
            params=params,
            header_info=header or {},
            trace_metadata=metadata or {},
        )
    )


@pytest.mark.parametrize("samples,total_ns", [(501, 900), (502, 300), (101, 250), (12, 9)])
def test_default_auto_windows_cover_all_samples_once_and_use_actual_axis(samples, total_ns):
    axis = np.linspace(17, 17 + total_ns, samples)
    out = run(np.zeros((samples, 65), np.float32), {"time_axis_ns": axis})
    assert out.metadata["window_mode"] == "auto"
    gates = out.metadata["gate_diagnostics"]
    assert len(gates) == 3
    assert gates[0]["sample_start"] == 0 and gates[-1]["sample_stop"] == samples
    assert all(a["sample_stop"] == b["sample_start"] for a, b in zip(gates, gates[1:]))
    assert all(g["sample_stop"] - g["sample_start"] >= 4 for g in gates)
    assert gates[0]["actual_ns"][0] == 17 and gates[-1]["actual_ns"][1] == 17 + total_ns


def test_auto_without_time_axis_does_not_invent_nanoseconds():
    out = run(np.zeros((60, 65), np.float32), {"time_axis_status": "unknown"})
    assert all("actual_ns" not in g for g in out.metadata["gate_diagnostics"])


@pytest.mark.parametrize("offset", [float("nan"), float("inf")])
def test_reconstructed_auto_axis_rejects_nonfinite_offset(offset):
    with pytest.raises(ValueError, match="时间轴"):
        run(np.zeros((60, 65), np.float32), {"time_step_s": 1e-9, "time_cut_offset_ns": offset})


def test_reconstructed_auto_axis_rejects_precision_collapsed_times():
    with pytest.raises(ValueError, match="时间轴"):
        run(np.zeros((60, 65), np.float32), {"time_step_s": 1e-30, "time_cut_offset_ns": 1e20})


def test_explicit_auto_takes_precedence_over_saved_manual_text():
    out = run(np.zeros((60, 65), np.float32), window_mode="auto", time_windows_ns="stale unused text")
    assert out.metadata["window_mode"] == "auto" and out.metadata["manual_windows_ignored"]
    assert len(out.metadata["gate_diagnostics"]) == 3


def test_early_only_change_is_local_and_never_whole_trace_repaired():
    raw = np.zeros((60, 65), np.float32)
    raw[:20, 20:22] = 1
    out = run(raw, mode="interpolate")
    assert not out.trace_metadata["short_trace_candidate_mask"].any()
    assert out.trace_metadata["short_trace_local_change_mask"][20:22].all()
    np.testing.assert_array_equal(out.data, raw)


def test_three_gate_change_can_be_candidate_but_local_neighbor_cannot_anchor_it():
    raw = np.zeros((60, 65), np.float32)
    raw[:, 20] = 1
    raw[:20, 19] = 1
    out = run(raw, mode="interpolate")
    assert out.trace_metadata["short_trace_candidate_mask"][20]
    assert out.trace_metadata["short_trace_local_change_mask"][19]
    assert not out.trace_metadata["short_trace_repaired_mask"].any()
    np.testing.assert_array_equal(out.data, raw)


def test_explicit_full_reproduces_legacy_candidates_and_is_recorded():
    raw = np.zeros((60, 65), np.float32)
    raw[:20, 20] = 1
    out = run(raw, window_mode="full", mode="interpolate")
    assert out.trace_metadata["short_trace_candidate_mask"][20]
    assert out.metadata["window_mode"] == "full"
    assert any(w["code"] == "short_trace_full_window" for w in out.runtime_warnings)


@pytest.mark.parametrize("samples", [1, 4, 8, 11])
def test_short_records_are_review_only_instead_of_fake_three_window_votes(samples):
    raw = np.zeros((samples, 65), np.float32)
    raw[:, 20] = 1
    out = run(raw, mode="interpolate")
    assert not out.trace_metadata["short_trace_candidate_mask"].any()
    assert out.trace_metadata["short_trace_local_change_mask"][20]
    np.testing.assert_array_equal(out.data, raw)


def test_short_record_manual_and_legacy_full_are_explicit_exceptions():
    raw = np.zeros((8, 65), np.float32)
    raw[:, 20] = 1
    manual = run(raw, {"time_axis_ns": np.arange(8.0)}, window_mode="manual", time_windows_ns="0:3;4:7")
    full = run(raw, window_mode="full")
    assert manual.trace_metadata["short_trace_candidate_mask"][20]
    assert full.trace_metadata["short_trace_candidate_mask"][20]


def test_repeated_manual_windows_are_rejected_as_nonindependent_votes():
    with pytest.raises(ValueError, match="重叠"):
        run(
            np.zeros((60, 65), np.float32),
            {"time_axis_ns": np.arange(60.0)},
            window_mode="manual",
            time_windows_ns="0:20;0:20;30:50",
        )


def test_one_manual_window_is_only_local_review():
    raw = np.zeros((60, 65), np.float32)
    raw[:, 20] = 1
    out = run(raw, {"time_axis_ns": np.arange(60.0)}, window_mode="manual", time_windows_ns="0:20", mode="interpolate")
    assert out.trace_metadata["short_trace_local_change_mask"][20]
    assert not out.trace_metadata["short_trace_candidate_mask"].any()
    np.testing.assert_array_equal(out.data, raw)


def test_adjacent_manual_windows_do_not_double_count_a_boundary_impulse():
    raw = np.zeros((60, 65), np.float32)
    raw[20, 20] = 1
    out = run(raw, {"time_axis_ns": np.arange(60.0)}, window_mode="manual", time_windows_ns="0:20;20:40")
    assert not out.trace_metadata["short_trace_candidate_mask"].any()
    assert out.trace_metadata["short_trace_local_change_mask"][20]
    assert out.metadata["gate_diagnostics"][0]["sample_stop"] == 20
    assert out.metadata["gate_diagnostics"][1]["sample_start"] == 20


def test_old_manual_parameters_remain_manual_when_loaded_into_new_editor(qapp):
    from ui.pages.processing_steps import ProcessingSteps
    from ui.pages.processing_page import ProcessingPage
    from ui.controllers.processing_controller import build_method_dicts
    from ui.desktop_backend_facade import method_catalog

    old = {"method_id": "short_trace_repair", "params": {"time_windows_ns": "0:100;100:200;250:850", "mode": "mark"}}
    normalized = ProcessingSteps._editor_parameters(old)
    assert normalized["window_mode"] == "manual" and "window_mode" not in old["params"]
    page = ProcessingPage()
    page.set_methods(build_method_dicts(method_catalog()))
    page._pipeline_list.add_step(old["method_id"], "test", old["params"])
    assert page._param_form.values()["window_mode"] == "manual"
    page.close()


def test_synthetic_real_target_in_all_auto_windows_still_exposes_false_positive_limit():
    clean = np.full((90, 65), 0.25, np.float32)
    raw = clean.copy()
    raw[[10, 40, 70], 20] += 1
    out = run(raw, mode="interpolate")
    assert out.trace_metadata["short_trace_candidate_mask"][20]
    np.testing.assert_array_equal(out.data, clean)


def test_local_mask_survives_equal_distance_resampling():
    from mygpr.infrastructure.processing.algorithms.motion.trace_utils import resample_trace_metadata

    out = resample_trace_metadata(
        {"trace_distance_m": np.arange(5.0), "short_trace_local_change_mask": np.array([0, 1, 0, 0, 0], np.int8)},
        target_trace_distance_m=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    )
    np.testing.assert_array_equal(out["short_trace_local_change_mask"], [0, 1, 1, 1, 0])


def test_local_and_cross_window_masks_have_distinct_preview_layers_and_hide_together(qapp):
    from core.quality_trace_locations import trace_quality_preview_mask, trace_local_change_preview_mask
    from ui.controllers.project_controller import ProjectController
    from ui.widgets.bscan_view import BScanView

    metadata = {
        "short_trace_candidate_mask": np.array([0, 0, 1, 0, 0, 0]),
        "short_trace_local_change_mask": np.array([0, 1, 0, 0, 0, 0]),
        "short_trace_unresolved_mask": np.array([0, 1, 1, 0, 0, 0]),
    }
    red = trace_quality_preview_mask(metadata, 6)
    local = trace_local_change_preview_mask(metadata, 6)
    np.testing.assert_array_equal(red, [0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(local, [0, 1, 0, 0, 0, 0])
    bundle = ProjectController._bundle_from_window(
        np.ones((12, 3), np.float32),
        "L",
        total_samples=12,
        total_traces=6,
        trace_indices=np.array([1, 2, 5]),
        sample_indices=np.arange(12),
        trace_qc_bad_mask=red,
        trace_qc_local_mask=local,
    )
    view = BScanView()
    view.set_bundle(bundle)
    assert len(view._qc_marks) == 1 and len(view._qc_local_marks) == 1
    assert tuple(view._qc_local_marks[0].getRegion()) == (0.0, 1.0)
    assert tuple(view._qc_marks[0].getRegion()) == (1.0, 2.0)
    view.set_qc_marks_visible(False)
    assert not view._qc_marks[0].isVisible() and not view._qc_local_marks[0].isVisible()
    view.set_qc_marks_visible(True)
    assert view._qc_marks[0].isVisible() and view._qc_local_marks[0].isVisible()
    view.clear()
    assert not view._qc_marks and not view._qc_local_marks
    view.close()


def test_new_classification_masks_save_export_and_reimport_without_changing_raw(tmp_path):
    import zipfile
    import json
    from mygpr.interfaces.backend import MyGPRBackend
    from mygpr.domain.processing.models import PipelineDefinition, PipelineStep

    raw = np.zeros((60, 65), np.float32)
    raw[:20, 10] = 1
    raw[:, 20] = 1
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(tmp_path / "project", name="auto")
        pid = project.project_id
        backend.projects.save_line_dataset(pid, "L", raw, time_window_ns=300)
        result = backend.project_processing.execute_pipeline(
            pid, "L", PipelineDefinition(steps=(PipelineStep("short_trace_repair", {}),))
        )
        saved = backend.projects.read_artifact_dataset(pid, "L", result.artifact_id)
        assert saved.trace_metadata["short_trace_local_change_mask"][10]
        assert saved.trace_metadata["short_trace_candidate_mask"][20]
        np.testing.assert_array_equal(saved.data, raw)
        job = backend.submit_processing_result_package(pid, "L", str(tmp_path / "package.zip"))
        exported = backend.jobs.wait(job, timeout=60)
        assert exported.status.value == "completed" and exported.result.verified
        with zipfile.ZipFile(exported.result.archive_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            entry = next(s for s in manifest["steps"] if s["artifact_id"] == result.artifact_id)
            target = tmp_path / "derived.h5"
            target.write_bytes(archive.read(entry["data"]))
        backend.projects.import_line_source(pid, "R", target)
        imported = backend.projects.read_dataset(pid, "R")
        for key in ("short_trace_candidate_mask", "short_trace_local_change_mask", "short_trace_unresolved_mask", "short_trace_zero_mask"):
            np.testing.assert_array_equal(imported.trace_metadata[key], saved.trace_metadata[key])
        np.testing.assert_array_equal(backend.projects.read_dataset(pid, "L").data, raw)
    finally:
        backend.shutdown()
