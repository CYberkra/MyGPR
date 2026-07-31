"""Optional GPU capability, memory-budget, and fallback policy.

The module never imports CuPy at import time.  CPU-only installations can import
all backend packages without the CUDA runtime or a GPU driver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GpuRequest = Literal["auto", "cpu", "gpu"]


class GpuBackendUnavailable(RuntimeError):
    """Raised when an explicit GPU request cannot be honoured."""


@dataclass(frozen=True, slots=True)
class GpuCapability:
    available: bool
    reason: str = ""
    device_id: int | None = None
    device_name: str = ""
    total_bytes: int = 0
    free_bytes: int = 0


@dataclass(frozen=True, slots=True)
class GpuSelection:
    requested: GpuRequest
    backend: Literal["cpu", "gpu"]
    fallback_reason: str = ""
    capability: GpuCapability = GpuCapability(False, "not queried")
    required_bytes: int = 0


_CAPABILITY_CACHE: GpuCapability | None = None


def query_gpu_capability(*, refresh: bool = False) -> GpuCapability:
    """Return a conservative CuPy/CUDA capability snapshot."""
    global _CAPABILITY_CACHE
    if _CAPABILITY_CACHE is not None and not refresh:
        return _CAPABILITY_CACHE
    try:
        import cupy as cp
    except ImportError as exc:
        result = GpuCapability(False, f"CuPy not installed: {exc}")
        _CAPABILITY_CACHE = result
        return result
    try:
        if not bool(cp.cuda.is_available()):
            result = GpuCapability(False, "CUDA runtime is not available")
            _CAPABILITY_CACHE = result
            return result
        device = cp.cuda.Device()
        device.use()
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        raw_name = properties.get("name", "") if isinstance(properties, dict) else ""
        if isinstance(raw_name, bytes):
            device_name = raw_name.decode("utf-8", errors="replace")
        else:
            device_name = str(raw_name)
        # Verify that allocation and host transfer actually work.  Driver-only
        # visibility is not enough for a production execution decision.
        probe = cp.zeros((16,), dtype=cp.float32)
        probe.get()
        del probe
        result = GpuCapability(
            True,
            device_id=int(device.id),
            device_name=device_name,
            total_bytes=int(total_bytes),
            free_bytes=int(free_bytes),
        )
    except Exception as exc:  # optional runtime can fail in many driver-specific ways
        result = GpuCapability(False, f"GPU initialization failed: {exc}")
    _CAPABILITY_CACHE = result
    return result


def resolve_gpu_backend(
    requested: str,
    *,
    required_bytes: int = 0,
    reserve_fraction: float = 0.15,
    refresh: bool = True,
) -> GpuSelection:
    """Resolve ``auto/cpu/gpu`` with an explicit free-memory contract.

    ``gpu`` is strict and raises when unavailable or under-budget. ``auto``
    falls back to CPU and records the exact reason.
    """
    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "gpu"}:
        raise ValueError("backend must be one of: auto, cpu, gpu")
    request: GpuRequest = normalized  # type: ignore[assignment]
    required = max(0, int(required_bytes))
    if request == "cpu":
        return GpuSelection(request, "cpu", capability=GpuCapability(False, "CPU explicitly requested"), required_bytes=required)

    capability = query_gpu_capability(refresh=refresh)
    reason = capability.reason
    if capability.available and required > 0:
        reserve = int(max(0.0, min(float(reserve_fraction), 0.9)) * capability.total_bytes)
        usable = max(0, capability.free_bytes - reserve)
        if usable < required:
            reason = (
                f"insufficient GPU memory: required={required} bytes, "
                f"usable={usable} bytes, free={capability.free_bytes} bytes"
            )
        else:
            return GpuSelection(request, "gpu", capability=capability, required_bytes=required)
    elif capability.available:
        return GpuSelection(request, "gpu", capability=capability, required_bytes=required)

    if request == "gpu":
        raise GpuBackendUnavailable(f"GPU backend requested but unavailable: {reason}")
    return GpuSelection(request, "cpu", fallback_reason=reason, capability=capability, required_bytes=required)



def is_gpu_resource_error(exc: BaseException) -> bool:
    """Return True only for allocation/resource exhaustion failures."""
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__.lower()
    module = type(exc).__module__.lower()
    message = str(exc).lower()
    if "outofmemory" in name or "out_of_memory" in name:
        return True
    if "cupy" in module and ("memory" in name or "allocation" in message):
        return True
    resource_markers = (
        "out of memory",
        "cuda_error_out_of_memory",
        "memory allocation failed",
        "insufficient memory",
    )
    return any(marker in message for marker in resource_markers)


def clear_gpu_capability_cache() -> None:
    """Clear cached capability data for tests or after device reconfiguration."""
    global _CAPABILITY_CACHE
    _CAPABILITY_CACHE = None


__all__ = [
    "GpuBackendUnavailable",
    "GpuCapability",
    "GpuSelection",
    "clear_gpu_capability_cache",
    "query_gpu_capability",
    "is_gpu_resource_error",
    "resolve_gpu_backend",
]
