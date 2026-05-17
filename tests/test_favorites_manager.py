#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for favorite parameter persistence boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from core.favorites_manager import FavoritesManager


def test_favorites_manager_writes_strict_json_for_nonfinite_params(tmp_path: Path):
    manager = FavoritesManager(config_dir=str(tmp_path))

    manager.add_favorite(
        "sec_gain",
        {"gain_min": float("nan"), "gain_max": float("inf"), "enabled": True},
        name="非有限收藏",
    )

    payload = json.loads((tmp_path / "favorites.json").read_text(encoding="utf-8"))
    params = payload["methods"]["sec_gain"][0]["params"]
    assert params["gain_min"] is None
    assert params["gain_max"] is None
    assert params["enabled"] is True
    json.dumps(payload, allow_nan=False)


def test_favorites_export_writes_strict_json_for_nonfinite_params(tmp_path: Path):
    manager = FavoritesManager(config_dir=str(tmp_path / "favorites"))
    manager.add_favorite("agcGain", {"window": float("nan")})

    export_path = tmp_path / "export.json"
    manager.export_favorites(str(export_path))

    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["methods"]["agcGain"][0]["params"]["window"] is None
    json.dumps(payload, allow_nan=False)
