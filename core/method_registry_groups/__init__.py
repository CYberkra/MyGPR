"""Declarative processing-method registry groups."""

from core.method_registry_groups.background_denoise import PROCESSING_METHODS_BACKGROUND_DENOISE
from core.method_registry_groups.calibration import PROCESSING_METHODS_CALIBRATION
from core.method_registry_groups.imaging import PROCESSING_METHODS_IMAGING
from core.method_registry_groups.motion import PROCESSING_METHODS_MOTION

__all__ = [
    "PROCESSING_METHODS_CALIBRATION",
    "PROCESSING_METHODS_BACKGROUND_DENOISE",
    "PROCESSING_METHODS_MOTION",
    "PROCESSING_METHODS_IMAGING",
]
