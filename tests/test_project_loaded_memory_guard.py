"""Loaded project pipelines must reject oversized work before reading the matrix."""
from types import SimpleNamespace

import numpy as np
import pytest

from mygpr.application.jobs.context import ExecutionContext
from mygpr.application.processing.service import ProcessingService
from mygpr.application.project.processing_service import ProjectProcessingService
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep, ProcessingRequest
from mygpr.domain.project.models import LineDatasetInfo
from mygpr.infrastructure.processing.block_executor import FileBackedBlockPipelineExecutor
from mygpr.infrastructure.processing.native_adapter import NativeProcessingCatalog, NativeProcessingExecutor
from mygpr.infrastructure.system import resource_policy


class _VirtualProjects:
    """A 400 MB matrix descriptor; reading it is forbidden, with no large allocation."""
    read_attempted = False

    def get_summary(self, *args):
        return SimpleNamespace(read_only=False)

    def get_dataset_info(self, *args):
        return LineDatasetInfo('L01', (10000, 10000), 'float32', 0., 900., 9.)

    def read_dataset(self, *args):
        self.read_attempted = True
        raise AssertionError('The resource guard must run before a full matrix read')


@pytest.mark.parametrize(('steps', 'budget'), [
    (1, 500000000), (8, 2000000000), (1, 4000000000), (8, 8000000000),
], ids=['loaded-step-workspace', 'retained-step-matrices',
        'dewow-float64-workspaces', 'retained-dewow-workspaces'])
def test_intermediate_pipeline_memory_guard_precedes_full_read(monkeypatch, steps, budget):
    monkeypatch.setattr(resource_policy, 'available_memory_bytes', lambda: 64 * 1024 ** 3)
    projects = _VirtualProjects()
    processing = ProcessingService(NativeProcessingCatalog(), NativeProcessingExecutor(),
        FileBackedBlockPipelineExecutor(), resource_policy.LocalProcessingResourcePolicy())
    service = ProjectProcessingService(projects, processing)
    pipeline = PipelineDefinition(steps=tuple(PipelineStep('dewow', {'window': 23}) for _ in range(steps)))
    # The same matrix fits the bounded block path; keeping every step runs loaded.
    assert processing.estimate_block_pipeline((10000, 10000), 'float32', pipeline).memory_bytes < budget
    with pytest.raises(resource_policy.InsufficientProcessingMemory):
        service.execute_pipeline('P', 'L01', pipeline, save_intermediates=True,
            context=ExecutionContext(metadata={'max_memory_bytes': budget}))
    assert not projects.read_attempted


def test_single_dewow_checks_float64_workspace_before_execution(monkeypatch):
    monkeypatch.setattr(resource_policy, 'available_memory_bytes', lambda: 64 * 1024 ** 3)
    executor = NativeProcessingExecutor()

    def forbid_execution(*args):
        raise AssertionError('Oversized Dewow must be rejected before executing its function')

    monkeypatch.setattr(executor, 'execute', forbid_execution)
    processing = ProcessingService(NativeProcessingCatalog(), executor,
        resource_policy=resource_policy.LocalProcessingResourcePolicy())
    # A logical 400 MB input backed by exactly one float32 scalar; no large allocation.
    data = np.broadcast_to(np.array(0., dtype=np.float32), (10000, 10000))
    with pytest.raises(resource_policy.InsufficientProcessingMemory):
        processing.execute_method(ProcessingRequest(data, 'dewow', {'window': 23}),
            context=ExecutionContext(metadata={'max_memory_bytes': 4000000000}))


def test_loaded_shape_estimate_uses_only_scalar_backed_readonly_views(monkeypatch):
    executor = NativeProcessingExecutor()
    estimate = executor.estimate
    calls = []

    def record_request(request):
        assert request.data.shape == (10000, 10000)
        assert request.data.strides == (0, 0)
        assert not request.data.flags.writeable
        assert request.data.base.nbytes == np.dtype('float32').itemsize
        calls.append(request.method_id)
        return estimate(request)

    monkeypatch.setattr(executor, 'estimate', record_request)
    processing = ProcessingService(NativeProcessingCatalog(), executor)
    pipeline = PipelineDefinition(steps=(PipelineStep('dewow', {'window': 23}),
        PipelineStep('dewow', {'window': 23}, enabled=False),
        PipelineStep('sec_gain', {'gain_min': 1., 'gain_max': 3.})))
    result = processing.estimate_pipeline_shape((10000, 10000), 'float32', pipeline,
        allow_chunking=False)
    assert calls == ['dewow', 'sec_gain']
    assert result.memory_bytes >= 6400000000


def test_block_dewow_checks_float64_workspace_before_reading_source(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_policy, 'available_memory_bytes', lambda: 64 * 1024 ** 3)

    class Source:
        shape = (501, 32)
        dtype = 'float32'

        def iter_blocks(self, *, block_rows):
            raise AssertionError('Dewow workspace must be checked before reading any data block')

    processing = ProcessingService(NativeProcessingCatalog(), NativeProcessingExecutor(),
        FileBackedBlockPipelineExecutor(tmp_path), resource_policy.LocalProcessingResourcePolicy())
    pipeline = PipelineDefinition(steps=(PipelineStep('dewow', {'window': 23}),))
    budget = 501 * 32 * 4 * 8
    with pytest.raises(resource_policy.InsufficientProcessingMemory):
        processing.execute_pipeline_blocks(Source(), pipeline,
            consumer=lambda matrix, summary: None,
            context=ExecutionContext(metadata={'max_memory_bytes': budget}))
