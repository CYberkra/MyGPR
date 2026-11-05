import numpy as np
import pytest

from mygpr.infrastructure.processing.algorithms.basic import method_dewow_native
from mygpr.infrastructure.processing.algorithms.common import normalize_output


def test_dewow_rejects_data_without_any_finite_sample():
    with pytest.raises(ValueError, match='没有有效数值'):
        method_dewow_native(np.full((64, 30), np.nan), {'window': 23})


def test_nonfinite_result_cannot_become_a_successful_zero_result():
    with pytest.raises(ValueError, match='没有有效数值'):
        normalize_output('dewow', np.full((64, 30), np.nan))
