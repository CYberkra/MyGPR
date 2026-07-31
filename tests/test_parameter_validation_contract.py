from __future__ import annotations

import numpy as np
import pytest

from mygpr.domain.common.errors import ParameterValidationError
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.interfaces.backend import MyGPRBackend


def test_processing_rejects_unknown_and_out_of_range_parameters() -> None:
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        data = np.ones((32, 8), dtype=np.float32)
        with pytest.raises(ParameterValidationError, match="unknown parameter"):
            backend.processing.estimate(ProcessingRequest(data, "dewow", {"windwo": 9}))
        with pytest.raises(ParameterValidationError, match="below minimum"):
            backend.processing.estimate(ProcessingRequest(data, "dewow", {"window": 0}))
        with pytest.raises(ParameterValidationError, match="must be int"):
            backend.processing.estimate(ProcessingRequest(data, "dewow", {"window": 9.5}))
    finally:
        backend.shutdown()
