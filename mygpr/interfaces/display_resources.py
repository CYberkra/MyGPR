"""Composition boundary for host resource checks used by desktop rendering."""
from mygpr.domain.processing.models import ResourceEstimate
from mygpr.infrastructure.system.resource_policy import LocalProcessingResourcePolicy


def validate_full_resolution_bscan(shape):
    LocalProcessingResourcePolicy().validate(
        ResourceEstimate(memory_bytes=int(shape[0]) * int(shape[1]) * 4 * 32),
        operation='全分辨率 B-scan 显示（不会自动降采样）')
