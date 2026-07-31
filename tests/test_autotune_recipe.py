#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for core.autotune_recipe — workflow recipe models and builders."""

from __future__ import annotations

import pytest

from core.autotune_recipe import (
    AutoTuneRecipe,
    AutoTuneRecipeStep,
    _recipe_steps_from_dicts,
    build_workflow_recipe,
    resolve_recipe_goal,
)


# ── AutoTuneRecipeStep ──────────────────────────────────────────────────────

class TestRecipeStep:
    def test_default_values(self) -> None:
        step = AutoTuneRecipeStep(key="test", label="Test", method="auto")
        assert step.key == "test"
        assert step.label == "Test"
        assert step.method == "auto"
        assert step.params == "--"
        assert step.enabled is True
        assert step.source == "auto"

    def test_custom_values(self) -> None:
        step = AutoTuneRecipeStep(
            key="gain", label="AGC", method="agc_gain",
            params="window=5", enabled=False, source="user"
        )
        assert step.params == "window=5"
        assert step.enabled is False
        assert step.source == "user"

    def test_frozen_prevents_mutation(self) -> None:
        step = AutoTuneRecipeStep(key="k", label="L", method="m")
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            step.enabled = False  # type: ignore[misc]


# ── AutoTuneRecipe ──────────────────────────────────────────────────────────

class TestRecipe:
    def _make_steps(self) -> tuple[AutoTuneRecipeStep, ...]:
        return (
            AutoTuneRecipeStep(key="dewow", label="Dewow", method="moving_avg", params="window=3"),
            AutoTuneRecipeStep(key="gain", label="AGC", method="agc_gain", params="window=10"),
        )

    def test_flow_text_joins_enabled_steps(self) -> None:
        recipe = AutoTuneRecipe(
            target_goal="均衡推荐", roi_mode="auto", steps=self._make_steps()
        )
        assert recipe.flow_text == "Dewow → AGC"

    def test_flow_text_skips_disabled_steps(self) -> None:
        steps = (
            AutoTuneRecipeStep(key="a", label="A", method="m", enabled=True),
            AutoTuneRecipeStep(key="b", label="B", method="m", enabled=False),
            AutoTuneRecipeStep(key="c", label="C", method="m", enabled=True),
        )
        recipe = AutoTuneRecipe(target_goal="t", roi_mode="r", steps=steps)
        assert recipe.flow_text == "A → C"

    def test_parameter_text_formats_steps(self) -> None:
        recipe = AutoTuneRecipe(
            target_goal="均衡推荐", roi_mode="auto", steps=self._make_steps()
        )
        text = recipe.parameter_text
        assert "Dewow" in text
        assert "AGC" in text
        assert "moving_avg" in text
        assert "window=3" in text

    def test_default_score_is_zero(self) -> None:
        recipe = AutoTuneRecipe(target_goal="t", roi_mode="r", steps=())
        assert recipe.score == 0.0

    def test_default_data_mode(self) -> None:
        recipe = AutoTuneRecipe(target_goal="t", roi_mode="r", steps=())
        assert recipe.data_mode == "无参考标签"


# ── resolve_recipe_goal ─────────────────────────────────────────────────────

class TestResolveRecipeGoal:
    def test_known_goal_returned_as_is(self) -> None:
        assert resolve_recipe_goal("均衡推荐") == "均衡推荐"

    def test_balanced_alias(self) -> None:
        assert resolve_recipe_goal("balanced") == "均衡推荐"

    def test_default_alias(self) -> None:
        assert resolve_recipe_goal("default") == "均衡推荐"

    def test_anomaly_alias(self) -> None:
        assert resolve_recipe_goal("anomaly") == "局部异常增强"

    def test_local_anomaly_alias(self) -> None:
        assert resolve_recipe_goal("local_anomaly") == "局部异常增强"

    def test_interface_alias(self) -> None:
        assert resolve_recipe_goal("interface") == "连续界面保留"

    def test_fracture_alias(self) -> None:
        assert resolve_recipe_goal("fracture") == "裂隙/破碎带保留"

    def test_wet_weak_zone_alias(self) -> None:
        assert resolve_recipe_goal("wet_weak_zone") == "含水软弱带"

    def test_deep_weak_reflection_alias(self) -> None:
        assert resolve_recipe_goal("deep_weak_reflection") == "深部弱反射增强"

    def test_unknown_goal_falls_back_to_default(self) -> None:
        assert resolve_recipe_goal("some_unknown_goal") == "均衡推荐"

    def test_none_falls_back_to_default(self) -> None:
        assert resolve_recipe_goal(None) == "均衡推荐"

    def test_empty_string_falls_back(self) -> None:
        assert resolve_recipe_goal("") == "均衡推荐"

    def test_whitespace_only_falls_back(self) -> None:
        assert resolve_recipe_goal("   ") == "均衡推荐"

    def test_case_insensitive_aliases(self) -> None:
        assert resolve_recipe_goal("BALANCED") == "均衡推荐"
        assert resolve_recipe_goal("Anomaly") == "局部异常增强"


# ── _recipe_steps_from_dicts ────────────────────────────────────────────────

class TestRecipeStepsFromDicts:
    def test_empty_list_returns_empty(self) -> None:
        assert _recipe_steps_from_dicts([]) == []

    def test_none_returns_empty(self) -> None:
        assert _recipe_steps_from_dicts(None) == []

    def test_parses_dict_entries(self) -> None:
        items = [
            {"key": "dewow", "label": "Dewow", "method": "moving_avg", "params": "w=5"},
        ]
        steps = _recipe_steps_from_dicts(items)
        assert len(steps) == 1
        assert steps[0].key == "dewow"
        assert steps[0].label == "Dewow"

    def test_missing_fields_get_defaults(self) -> None:
        items = [{"key": "gain"}]
        steps = _recipe_steps_from_dicts(items)
        assert steps[0].label == "gain"  # falls back to key
        assert steps[0].method == "auto"
        assert steps[0].params == "--"
        assert steps[0].enabled is True

    def test_existing_recipe_steps_passthrough(self) -> None:
        existing = AutoTuneRecipeStep(key="k", label="L", method="m")
        steps = _recipe_steps_from_dicts([existing])
        assert len(steps) == 1
        assert steps[0] is existing


# ── build_workflow_recipe ───────────────────────────────────────────────────

class TestBuildWorkflowRecipe:
    def test_minimal_args_returns_recipe(self) -> None:
        recipe = build_workflow_recipe(target_goal="均衡推荐", roi_mode="auto")
        assert isinstance(recipe, AutoTuneRecipe)
        assert recipe.target_goal == "均衡推荐"
        assert len(recipe.steps) > 0

    def test_default_goal_produces_five_steps(self) -> None:
        recipe = build_workflow_recipe(target_goal=None, roi_mode=None)
        # 均衡推荐 has 5 pipeline steps
        assert len(recipe.steps) == 5

    def test_fracture_goal_produces_six_steps(self) -> None:
        recipe = build_workflow_recipe(target_goal="裂隙/破碎带保留", roi_mode="manual")
        # 裂隙/破碎带保留 adds denoise → 6 steps
        assert len(recipe.steps) == 6

    def test_best_candidate_overrides_background_method(self) -> None:
        recipe = build_workflow_recipe(
            target_goal="均衡推荐", roi_mode="auto",
            best_candidate_name="my_bg_method",
            best_candidate_params="alpha=0.5",
        )
        bg_step = [s for s in recipe.steps if s.key == "background"][0]
        assert bg_step.method == "my_bg_method"
        assert bg_step.params == "alpha=0.5"

    def test_dash_best_candidate_name_does_not_override(self) -> None:
        recipe = build_workflow_recipe(
            target_goal="均衡推荐", roi_mode="auto",
            best_candidate_name="--",
        )
        bg_step = [s for s in recipe.steps if s.key == "background"][0]
        assert bg_step.method == "auto"  # not overridden

    def test_score_propagates(self) -> None:
        recipe = build_workflow_recipe(
            target_goal="均衡推荐", roi_mode="auto", best_score=0.85
        )
        assert recipe.score == 0.85

    def test_target_response_available_changes_data_mode(self) -> None:
        recipe = build_workflow_recipe(
            target_goal="均衡推荐", roi_mode="auto",
            target_response_available=True,
        )
        assert recipe.data_mode == "有参考响应"

    def test_notes_include_goal_and_roi(self) -> None:
        recipe = build_workflow_recipe(target_goal="局部异常增强", roi_mode="manual")
        notes_text = " ".join(recipe.notes)
        assert "局部异常增强" in notes_text
        assert "手动框选" in notes_text

    def test_notes_include_score_when_present(self) -> None:
        recipe = build_workflow_recipe(
            target_goal="均衡推荐", roi_mode="auto", best_score=0.92
        )
        notes_text = " ".join(recipe.notes)
        assert "0.92" in notes_text

    def test_custom_recipe_steps_override_pipeline(self) -> None:
        custom = [{"key": "custom_filter", "label": "自定义滤波", "method": "lowpass"}]
        recipe = build_workflow_recipe(
            target_goal="均衡推荐", roi_mode="auto", recipe_steps=custom
        )
        assert len(recipe.steps) == 1
        assert recipe.steps[0].key == "custom_filter"

    def test_all_goals_produce_valid_recipes(self) -> None:
        goals = [
            "均衡推荐", "局部异常增强", "连续界面保留",
            "滑坡基覆界面 / 潜在滑移面", "裂隙/破碎带保留",
            "含水软弱带", "深部弱反射增强",
        ]
        for goal in goals:
            recipe = build_workflow_recipe(target_goal=goal, roi_mode="auto")
            assert len(recipe.steps) >= 5
            assert recipe.flow_text  # not empty
