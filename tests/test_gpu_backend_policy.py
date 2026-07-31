from __future__ import annotations

import pytest

from mygpr.infrastructure.processing import gpu_policy
from mygpr.infrastructure.processing.gpu_policy import GpuBackendUnavailable, GpuCapability


def test_auto_gpu_request_falls_back_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gpu_policy,
        "query_gpu_capability",
        lambda **_: GpuCapability(False, "test CUDA unavailable"),
    )
    selection = gpu_policy.resolve_gpu_backend("auto", required_bytes=1024)
    assert selection.backend == "cpu"
    assert selection.fallback_reason == "test CUDA unavailable"


def test_explicit_gpu_request_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gpu_policy,
        "query_gpu_capability",
        lambda **_: GpuCapability(False, "test CUDA unavailable"),
    )
    with pytest.raises(GpuBackendUnavailable, match="test CUDA unavailable"):
        gpu_policy.resolve_gpu_backend("gpu")


def test_gpu_memory_budget_reserves_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gpu_policy,
        "query_gpu_capability",
        lambda **_: GpuCapability(True, device_id=0, device_name="test", total_bytes=1000, free_bytes=800),
    )
    fallback = gpu_policy.resolve_gpu_backend("auto", required_bytes=700, reserve_fraction=0.2)
    assert fallback.backend == "cpu"
    assert "insufficient GPU memory" in fallback.fallback_reason
    selected = gpu_policy.resolve_gpu_backend("auto", required_bytes=500, reserve_fraction=0.2)
    assert selected.backend == "gpu"


def test_gpu_dynamic_memory_is_refreshed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def query(**kwargs):
        calls.append(kwargs)
        return GpuCapability(False, "none")

    monkeypatch.setattr(gpu_policy, "query_gpu_capability", query)
    gpu_policy.resolve_gpu_backend("auto", required_bytes=1)
    assert calls[-1]["refresh"] is True


def test_only_resource_failures_are_runtime_fallback_eligible() -> None:
    assert gpu_policy.is_gpu_resource_error(MemoryError("oom"))
    assert gpu_policy.is_gpu_resource_error(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
    assert not gpu_policy.is_gpu_resource_error(RuntimeError("kernel produced invalid value"))
