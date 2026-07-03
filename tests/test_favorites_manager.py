#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for core.favorites_manager — parameter favorites CRUD operations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.favorites_manager import FavoritesManager


@pytest.fixture
def manager(tmp_path: Path) -> FavoritesManager:
    """Create a FavoritesManager pointed at a temporary directory."""
    return FavoritesManager(config_dir=str(tmp_path / "favorites"))


# ── Initialization ──────────────────────────────────────────────────────────

class TestInitialization:
    def test_creates_config_dir_if_missing(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nonexistent_config"
        assert not config_dir.exists()
        FavoritesManager(config_dir=str(config_dir))
        assert config_dir.exists()

    def test_empty_state_has_no_favorites(self, manager: FavoritesManager) -> None:
        assert manager.get_favorites() == []

    def test_empty_state_has_no_methods(self, manager: FavoritesManager) -> None:
        assert manager.favorites["methods"] == {}

    def test_empty_state_last_updated_is_none(self, manager: FavoritesManager) -> None:
        assert manager.favorites["last_updated"] is None


# ── Add favorites ────────────────────────────────────────────────────────────

class TestAddFavorite:
    def test_add_single_favorite(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        favs = manager.get_favorites("dewow")
        assert len(favs) == 1
        assert favs[0]["params"] == {"window": 3}

    def test_add_multiple_favorites_same_method(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        manager.add_favorite("dewow", {"window": 5})
        assert len(manager.get_favorites("dewow")) == 2

    def test_add_favorites_different_methods(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        manager.add_favorite("gain", {"agc_window": 10})
        assert len(manager.get_favorites("dewow")) == 1
        assert len(manager.get_favorites("gain")) == 1

    def test_duplicate_params_does_not_create_new(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        manager.add_favorite("dewow", {"window": 3})
        assert len(manager.get_favorites("dewow")) == 1

    def test_duplicate_updates_name(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        manager.add_favorite("dewow", {"window": 3}, name="Updated Name")
        favs = manager.get_favorites("dewow")
        assert favs[0]["name"] == "Updated Name"

    def test_custom_name(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3}, name="我的收藏")
        assert manager.get_favorites("dewow")[0]["name"] == "我的收藏"

    def test_auto_generated_name(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        assert "收藏" in manager.get_favorites("dewow")[0]["name"]

    def test_favorite_has_id(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        fav = manager.get_favorites("dewow")[0]
        assert "id" in fav
        assert fav["id"].startswith("dewow_")

    def test_favorite_has_created_at(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        assert "created_at" in manager.get_favorites("dewow")[0]

    def test_favorite_starts_with_zero_used_count(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        assert manager.get_favorites("dewow")[0]["used_count"] == 0

    def test_last_updated_is_set_after_add(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        assert manager.favorites["last_updated"] is not None

    def test_name_uniqueness_same_base_name(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"w": 1}, name="My Fav")
        manager.add_favorite("dewow", {"w": 2}, name="My Fav")
        names = {f["name"] for f in manager.get_favorites("dewow")}
        assert len(names) == 2  # second gets "My Fav (2)"

    def test_complex_params_dict(self, manager: FavoritesManager) -> None:
        params = {"window": 5, "mode": "aggressive", "nested": {"key": [1, 2, 3]}}
        manager.add_favorite("complex_method", params)
        assert manager.get_favorites("complex_method")[0]["params"] == params


# ── Get favorites ────────────────────────────────────────────────────────────

class TestGetFavorites:
    def test_get_by_method_id_returns_only_that_method(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"w": 1})
        manager.add_favorite("gain", {"w": 2})
        dewow_favs = manager.get_favorites("dewow")
        assert len(dewow_favs) == 1
        assert dewow_favs[0]["params"] == {"w": 1}

    def test_get_all_without_method_id(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"w": 1})
        manager.add_favorite("gain", {"w": 2})
        all_favs = manager.get_favorites()
        assert len(all_favs) == 2
        # Each item should have method_id added
        assert all_favs[0]["method_id"] in ("dewow", "gain")

    def test_get_nonexistent_method_returns_empty(self, manager: FavoritesManager) -> None:
        assert manager.get_favorites("nonexistent") == []


# ── Remove favorites ────────────────────────────────────────────────────────

class TestRemoveFavorite:
    def test_remove_existing_favorite(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        fav_id = manager.get_favorites("dewow")[0]["id"]
        manager.remove_favorite("dewow", fav_id)
        assert manager.get_favorites("dewow") == []

    def test_remove_cleans_up_empty_method(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        fav_id = manager.get_favorites("dewow")[0]["id"]
        manager.remove_favorite("dewow", fav_id)
        assert "dewow" not in manager.favorites["methods"]

    def test_remove_nonexistent_fav_id_does_not_crash(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        manager.remove_favorite("dewow", "nonexistent_id")
        assert len(manager.get_favorites("dewow")) == 1

    def test_remove_from_nonexistent_method_does_not_crash(self, manager: FavoritesManager) -> None:
        manager.remove_favorite("nonexistent", "some_id")  # should not raise


# ── Mark used ────────────────────────────────────────────────────────────────

class TestMarkUsed:
    def test_mark_used_increments_count(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        fav_id = manager.get_favorites("dewow")[0]["id"]
        manager.mark_used("dewow", fav_id)
        manager.mark_used("dewow", fav_id)
        assert manager.get_favorites("dewow")[0]["used_count"] == 2

    def test_mark_used_sets_last_used(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        fav_id = manager.get_favorites("dewow")[0]["id"]
        manager.mark_used("dewow", fav_id)
        assert "last_used" in manager.get_favorites("dewow")[0]

    def test_mark_used_nonexistent_does_not_crash(self, manager: FavoritesManager) -> None:
        manager.mark_used("dewow", "nonexistent")


# ── Get recently used / most used ───────────────────────────────────────────

class TestUsageRanking:
    def test_recently_used_returns_most_recent_first(self, manager: FavoritesManager) -> None:
        import time
        manager.add_favorite("dewow", {"w": 1}, name="A")
        manager.add_favorite("dewow", {"w": 2}, name="B")
        favs = manager.get_favorites("dewow")
        manager.mark_used("dewow", favs[0]["id"])
        time.sleep(0.01)  # ensure distinct timestamps
        manager.mark_used("dewow", favs[1]["id"])
        recent = manager.get_recently_used(limit=2)
        assert recent[0]["name"] == "B"  # most recently used

    def test_most_used_returns_highest_count_first(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"w": 1}, name="A")
        manager.add_favorite("dewow", {"w": 2}, name="B")
        favs = manager.get_favorites("dewow")
        manager.mark_used("dewow", favs[0]["id"])
        manager.mark_used("dewow", favs[0]["id"])
        manager.mark_used("dewow", favs[0]["id"])
        manager.mark_used("dewow", favs[1]["id"])
        most = manager.get_most_used(limit=2)
        assert most[0]["name"] == "A"  # used 3 times vs 1

    def test_recently_used_respects_limit(self, manager: FavoritesManager) -> None:
        for i in range(10):
            manager.add_favorite("m", {"i": i}, name=f"F{i}")
        assert len(manager.get_recently_used(limit=3)) == 3

    def test_most_used_respects_limit(self, manager: FavoritesManager) -> None:
        for i in range(10):
            manager.add_favorite("m", {"i": i}, name=f"F{i}")
        assert len(manager.get_most_used(limit=5)) == 5


# ── Clear all ────────────────────────────────────────────────────────────────

class TestClearAll:
    def test_clear_removes_all_favorites(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"w": 1})
        manager.add_favorite("gain", {"w": 2})
        manager.clear_all()
        assert manager.get_favorites() == []

    def test_clear_resets_methods(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"w": 1})
        manager.clear_all()
        assert manager.favorites["methods"] == {}

    def test_clear_resets_last_updated(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"w": 1})
        manager.clear_all()
        assert manager.favorites["last_updated"] is None


# ── Export / Import ──────────────────────────────────────────────────────────

class TestExportImport:
    def test_export_writes_valid_json(self, manager: FavoritesManager, tmp_path: Path) -> None:
        manager.add_favorite("dewow", {"window": 3})
        export_path = tmp_path / "export.json"
        manager.export_favorites(str(export_path))
        assert export_path.exists()
        data = json.loads(export_path.read_text(encoding="utf-8"))
        assert "dewow" in data["methods"]

    def test_import_merges_favorites(self, manager: FavoritesManager, tmp_path: Path) -> None:
        manager.add_favorite("dewow", {"window": 3})
        # Export to file
        export_path = tmp_path / "export.json"
        manager.export_favorites(str(export_path))
        # Import into a fresh manager
        manager2 = FavoritesManager(config_dir=str(tmp_path / "favorites2"))
        manager2.import_favorites(str(export_path))
        assert len(manager2.get_favorites("dewow")) == 1

    def test_import_skips_duplicates(self, manager: FavoritesManager, tmp_path: Path) -> None:
        manager.add_favorite("dewow", {"window": 3})
        export_path = tmp_path / "export.json"
        manager.export_favorites(str(export_path))
        # Import into same manager → should not duplicate
        count_before = len(manager.get_favorites("dewow"))
        manager.import_favorites(str(export_path))
        assert len(manager.get_favorites("dewow")) == count_before

    def test_import_merges_different_methods(self, manager: FavoritesManager, tmp_path: Path) -> None:
        manager.add_favorite("dewow", {"w": 1})
        export_path = tmp_path / "export.json"
        manager.export_favorites(str(export_path))
        # Another manager with different favorites
        manager2 = FavoritesManager(config_dir=str(tmp_path / "f2"))
        manager2.add_favorite("gain", {"w": 2})
        manager2.import_favorites(str(export_path))
        assert "dewow" in manager2.favorites["methods"]
        assert "gain" in manager2.favorites["methods"]


# ── Persistence ──────────────────────────────────────────────────────────────

class TestPersistence:
    def test_favorites_survive_reload(self, tmp_path: Path) -> None:
        config_dir = str(tmp_path / "persist")
        m1 = FavoritesManager(config_dir=config_dir)
        m1.add_favorite("dewow", {"window": 3})
        m1.add_favorite("gain", {"agc_window": 10})

        # Reload
        m2 = FavoritesManager(config_dir=config_dir)
        assert len(m2.get_favorites()) == 2
        assert m2.get_favorites("dewow")[0]["params"] == {"window": 3}

    def test_favorites_file_is_created(self, manager: FavoritesManager) -> None:
        manager.add_favorite("dewow", {"window": 3})
        assert Path(manager.favorites_file).exists()

    def test_corrupted_favorites_file_is_handled(self, tmp_path: Path) -> None:
        config_dir = str(tmp_path / "corrupt")
        Path(config_dir).mkdir(parents=True, exist_ok=True)
        (Path(config_dir) / "favorites.json").write_text("not valid json {{{", encoding="utf-8")
        # Should not crash, should return empty state
        manager = FavoritesManager(config_dir=config_dir)
        assert manager.get_favorites() == []
