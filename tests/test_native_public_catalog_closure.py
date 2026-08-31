"""Release contract for the phase-8 native processing catalog closure."""

from core.methods_registry import PROCESSING_METHODS
from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor


def _public_method_ids() -> set[str]:
    return {
        method_id
        for method_id, metadata in PROCESSING_METHODS.items()
        if metadata.get("visibility", "public") == "public"
    }


def test_every_public_processing_method_has_a_native_backend() -> None:
    public_methods = _public_method_ids()
    assert len(public_methods) == 33
    assert public_methods <= set(NATIVE_ALGORITHMS)


def test_native_executor_supports_the_complete_public_catalog() -> None:
    executor = NativeProcessingExecutor()
    unsupported = sorted(
        method_id for method_id in _public_method_ids() if not executor.supports(method_id)
    )
    assert unsupported == []
