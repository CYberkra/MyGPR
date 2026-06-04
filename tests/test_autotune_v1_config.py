#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke contracts for AutoTune V1 final-candidate profile/recipe config."""

from __future__ import annotations

from core.autotune_v1_config import PROFILE_WEIGHT_FIELDS, load_autotune_v1_config


def test_autotune_v1_config_loads_and_validates_profiles_and_recipes():
    config = load_autotune_v1_config()
    assert config.version.startswith("autotune_v1_final_candidate")
    assert config.status == "design_final_candidate_not_production_default"
    assert set(config.profiles) == {
        "balanced",
        "object_like_anomaly",
        "interface_layer_preservation",
        "landslide_bedrock_sliding_surface",
        "wet_weak_zone",
        "deep_weak_reflector",
    }

    for profile in config.profiles.values():
        assert set(profile.weights) == set(PROFILE_WEIGHT_FIELDS)
        assert abs(sum(profile.normalized_weights.values()) - 1.0) < 1e-9
        assert profile.requires_batch_calibration is True

    for recipe in config.recipes.values():
        assert recipe.steps
        assert recipe.profiles
        assert all(profile_id in config.profiles for profile_id in recipe.profiles)


def test_autotune_v1_profile_aliases_keep_landslide_interface_conservative():
    config = load_autotune_v1_config()
    landslide = config.profile_for_goal("滑坡基覆界面 / 潜在滑移面")
    assert landslide.profile_id == "landslide_bedrock_sliding_surface"
    assert landslide.weights["continuity"] > landslide.weights["background_suppression"]
    assert landslide.weights["depth_weak_reflector"] > landslide.weights["background_suppression"]

    fracture = config.profile_for_goal("裂隙/破碎带保留")
    assert fracture.profile_id == "interface_layer_preservation"

    recipes = config.recipes_for_profile("landslide_interface")
    recipe_ids = {recipe.recipe_id for recipe in recipes}
    assert "interface_preservation" in recipe_ids
    assert "deep_weak_reflector" in recipe_ids


def test_autotune_v1_config_encodes_scoring_boundaries_and_display_only_gain():
    config = load_autotune_v1_config()
    gain_spec = config.candidate_spec["gain"]
    assert "agc" in gain_spec["display_only"]
    assert "AGC display-only" in gain_spec["metric_safe"]

    synthetic = config.scoring_mode_spec("synthetic_paired")
    assert "rmse" in synthetic["full_reference_metrics"]
    assert "ssim_or_ssim_like" in synthetic["full_reference_metrics"]

    real = config.scoring_mode_spec("real_no_prior")
    forbidden = set(real["forbidden_metrics"])
    assert "mse_against_unknown_truth" in forbidden
    assert "ssim_against_unknown_truth" in forbidden
    assert "manual_review_required" in real["claim"]

    assert "manual_review_required" in config.manifest_required_fields
    assert "no_prior_manual_review_required" in config.warning_tags
