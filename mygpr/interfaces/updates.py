"""Desktop distribution entry points, separate from the scientific backend API."""
from pathlib import Path

from mygpr.infrastructure.updates.configuration import UpdateConfiguration
from mygpr.infrastructure.updates.winsparkle import WinSparkle


def update_configuration(path: Path) -> UpdateConfiguration:
    return UpdateConfiguration.load(path)


def update_client(path: Path) -> WinSparkle:
    return WinSparkle(path)
