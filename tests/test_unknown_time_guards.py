import numpy as np
import pytest

from mygpr.application.autotune.candidate_generators import _build_frequency_filter_trials, _resolve_time_step_ns
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.algorithms.basic import method_zero_time
from mygpr.infrastructure.processing.block_executor import prepare_block_params
from mygpr.infrastructure.processing.native_adapter import prepare_native_params


def test_unknown_time_is_rejected_by_all_zero_time_entry_points():
    data = np.zeros((64, 4), dtype=np.float32)
    header = {'time_axis_status': 'unknown', 'time_window_ns': 250.0}
    with pytest.raises(ValueError, match='时间轴'):
        prepare_native_params(ProcessingRequest(data, 'set_zero_time', {}, header))
    with pytest.raises(ValueError, match='时间轴'):
        prepare_block_params({}, header, data.shape, 'set_zero_time')
    with pytest.raises(ValueError, match='时间轴'):
        method_zero_time(data, {'new_zero_time': 5, '_header_info': header})


def test_unknown_time_is_rejected_before_frequency_autotune():
    data = np.zeros((64, 4), dtype=np.float32)
    with pytest.raises(ValueError, match='时间轴未知'):
        _build_frequency_filter_trials(data, {'time_axis_status': 'unknown',
                                               'time_window_ns': 250.0}, {}, {}, 'coarse')


def test_zero_time_autotune_uses_real_sample_spacing():
    with pytest.raises(ValueError, match='时间轴未知'):
        _resolve_time_step_ns(64, {'time_axis_status': 'unknown', 'time_window_ns': 250.0})
    assert _resolve_time_step_ns(64, {'time_step_s': 2e-9}) == pytest.approx(2.0)
    assert _resolve_time_step_ns(64, {'time_window_ns': 126.0,
                                      'time_axis_endpoint': True}) == pytest.approx(2.0)
