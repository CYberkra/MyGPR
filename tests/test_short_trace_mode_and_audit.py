"""Mode labels and conservative, persistent anomaly-review contracts."""
import numpy as np
import pytest

from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor


def execute(data, metadata=None, **params):
    return NativeProcessingExecutor().execute(ProcessingRequest(
        data=data, method_id='short_trace_repair', params=params,
        header_info={'time_axis_ns': np.linspace(0,900,data.shape[0])},
        trace_metadata=metadata or {}))


def test_prior_unresolved_provenance_survives_when_a_later_step_does_not_redetect_it():
    raw = np.full((101,65),0.25,np.float32)
    mask = np.zeros(65,np.int8)
    mask[12] = 1
    result = execute(raw, {'short_trace_unresolved_mask':mask})
    assert not result.trace_metadata['short_trace_candidate_mask'].any()
    np.testing.assert_array_equal(result.trace_metadata['short_trace_unresolved_mask'],mask)
    assert result.metadata['unresolved_trace_count'] == 1


def test_historical_unresolved_positions_remain_in_preview():
    from core.quality_trace_locations import trace_quality_preview_mask
    mask = np.array([0,1,0],np.int8)
    np.testing.assert_array_equal(trace_quality_preview_mask({'short_trace_unresolved_mask':mask},3),mask)


def test_repair_clears_only_the_resolved_positions_from_prior_unresolved():
    raw = np.full((101,65),0.25,np.float32)
    raw[:,20] = 1
    mask = np.zeros(65,np.int8)
    mask[[12,20]] = 1
    result = execute(raw, {'short_trace_unresolved_mask':mask}, mode='interpolate')
    assert result.trace_metadata['short_trace_repaired_mask'][20]
    assert result.trace_metadata['short_trace_unresolved_mask'][12]
    assert not result.trace_metadata['short_trace_unresolved_mask'][20]


def test_chinese_mode_selector_returns_internal_values_and_reopens_saved_selection(qapp):
    from qfluentwidgets import ComboBox
    from ui.controllers.processing_controller import build_method_dicts
    from ui.desktop_backend_facade import method_catalog
    from ui.widgets.param_form import ParamForm
    method = next(m for m in build_method_dicts(method_catalog()) if m['method_id']=='short_trace_repair')
    form = ParamForm()
    form.set_schema(method['parameter_schema'])
    editor = form._editors['mode']
    assert isinstance(editor,ComboBox)
    assert [editor.itemText(i) for i in range(editor.count())] == ['仅标记','插补修复']
    assert form.values()['mode'] == 'mark'
    editor.setCurrentIndex(1)
    assert form.values()['mode'] == 'interpolate'
    saved = form.values()
    reopened = ParamForm()
    reopened.set_schema(method['parameter_schema'])
    reopened.set_values(saved)
    assert reopened._editors['mode'].currentText() == '插补修复'
    assert reopened.values() == saved


def test_changing_translated_choice_label_refreshes_the_form(qapp):
    from ui.widgets.param_form import ParamForm
    schema = [{'name':'mode','type':'str','default':'mark','choices':['mark','interpolate'],
               'choice_labels':{'mark':'仅标记','interpolate':'插补修复'}}]
    form = ParamForm()
    form.set_schema(schema)
    schema[0] = {**schema[0],'choice_labels':{'mark':'仅预览标记','interpolate':'插补修复'}}
    form.set_schema(schema)
    assert form._editors['mode'].currentText() == '仅预览标记'
    assert form.values()['mode'] == 'mark'


@pytest.mark.parametrize('length',[1,2,3,4,5,8,15])
def test_coherent_anomaly_lengths_are_detected_but_only_one_to_four_are_repaired(length):
    raw = np.full((101,129),0.25,np.float32)
    raw[:,50:50+length] = 1
    result = execute(raw,mode='interpolate')
    candidates = result.trace_metadata['short_trace_candidate_mask'].astype(bool)
    assert candidates.sum() == length
    if length <= 4:
        assert result.trace_metadata['short_trace_repaired_mask'].sum() == length
        np.testing.assert_array_equal(result.data,np.full_like(raw,0.25))
    else:
        np.testing.assert_array_equal(result.data,raw)
        assert result.trace_metadata['short_trace_unresolved_mask'].sum() == length


@pytest.mark.parametrize('start',[0,1,125,128])
def test_end_anomalies_do_not_extrapolate(start):
    raw = np.zeros((101,129),np.float32)
    length = 1 if start in (0,128) else 3
    raw[:,start:start+length] = 1
    result = execute(raw,mode='interpolate')
    if start == 0 or start+length == raw.shape[1]:
        np.testing.assert_array_equal(result.data,raw)
        assert not result.trace_metadata['short_trace_repaired_mask'].any()


def test_multigate_preserves_local_wave_packet_while_full_window_can_erase_it():
    time = np.linspace(0,900,501)
    raw = np.full((501,129),0.25,np.float32)
    packet = (np.exp(-((time-400)/18)**2)*np.cos((time-400)/5)).astype(np.float32)
    raw[:,60:62] += packet[:,None]
    full = execute(raw,mode='interpolate',window_mode='full')
    gates = execute(raw,mode='interpolate',time_windows_ns='0:100;100:200;250:850')
    assert full.trace_metadata['short_trace_repaired_mask'][60:62].all()
    np.testing.assert_array_equal(gates.data,raw)
    assert not gates.trace_metadata['short_trace_candidate_mask'].any()


def test_separated_real_wave_packets_in_all_gates_are_an_indistinguishable_counterexample():
    time = np.linspace(0,900,501)
    raw = np.full((501,129),0.25,np.float32)
    for center in (50,150,400):
        raw[:,60] += (np.exp(-((time-center)/9)**2)*np.cos((time-center)/3)).astype(np.float32)
    result = execute(raw,mode='interpolate',time_windows_ns='0:100;100:200;250:850')
    assert result.trace_metadata['short_trace_repaired_mask'][60]
    np.testing.assert_array_equal(result.data,np.full_like(raw,0.25))


def test_majority_of_local_reference_can_hide_a_long_coherent_bad_run():
    raw = np.zeros((101,129),np.float32)
    raw[:,50:70] = 1
    result = execute(raw,mode='interpolate')
    assert not result.trace_metadata['short_trace_candidate_mask'].any()
    np.testing.assert_array_equal(result.data,raw)


def test_all_preexisting_bad_anchors_prevent_any_interpolation():
    raw = np.zeros((101,65),np.float32)
    raw[:,20:22] = 1
    result = execute(raw, {'trace_qc_bad_mask':np.ones(65,np.int8)},mode='interpolate')
    assert result.trace_metadata['short_trace_candidate_mask'][20:22].all()
    assert not result.trace_metadata['short_trace_repaired_mask'].any()
    np.testing.assert_array_equal(result.data,raw)


def test_zero_mad_has_strict_threshold_and_no_false_flags_on_constant_data():
    raw = np.ones((101,65),np.float32)
    result = execute(raw)
    assert result.metadata['gate_diagnostics'][0]['threshold'] == 0
    assert not result.trace_metadata['short_trace_candidate_mask'].any()


@pytest.mark.parametrize('length',[1,2,3,4])
def test_correct_known_detection_matches_independent_two_anchor_interpolation(length):
    raw = np.random.default_rng(31).normal(0,.01,(101,129)).astype(np.float32)
    raw[:,50:50+length] = 10
    expected = raw.copy()
    for offset in range(length):
        weight = (offset+1)/(length+1)
        expected[:,50+offset] = ((1-weight)*raw[:,49].astype(float)+weight*raw[:,50+length].astype(float))
    result = execute(raw,mode='interpolate')
    assert result.trace_metadata['short_trace_repaired_mask'].sum() == length
    np.testing.assert_array_equal(result.data,expected)
