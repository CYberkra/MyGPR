#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration loader for AutoTune V1 final-candidate profiles and recipes.

This module intentionally loads a *configuration contract* rather than changing
runtime scoring defaults.  The V1 profile/recipe file is a final-candidate design
artifact that must be batch-calibrated before it becomes the production default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # PyYAML is already a project dependency in the development environment.
    import yaml
except ImportError:  # pragma: no cover - exercised only in stripped deployments
    yaml = None  # type: ignore[assignment]

PROFILE_WEIGHT_FIELDS: tuple[str, ...] = (
    "target_preservation",
    "background_suppression",
    "continuity",
    "contrast",
    "false_positive_penalty",
    "ringing_artifact_penalty",
    "depth_weak_reflector",
)

_SOURCE_AUTOTUNE_V1_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "autotune_v1_profiles.yaml"
)
_PACKAGED_AUTOTUNE_V1_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "mygpr" / "resources" / "config" / "autotune_v1_profiles.yaml"
)
DEFAULT_AUTOTUNE_V1_CONFIG_PATH = (
    _SOURCE_AUTOTUNE_V1_CONFIG_PATH
    if _SOURCE_AUTOTUNE_V1_CONFIG_PATH.is_file()
    else _PACKAGED_AUTOTUNE_V1_CONFIG_PATH
)

# Stable aliases used by UI labels, report labels, CLI strings and previous
# experimental profile names.  The V1 final-candidate set deliberately keeps six
# primary profiles; fracture/broken-zone is mapped conservatively to the
# interface/layer preservation profile rather than becoming a separate default.
PROFILE_ALIASES: dict[str, str] = {
    "balanced": "balanced",
    "balance": "balanced",
    "default": "balanced",
    "均衡推荐": "balanced",
    "均衡处理": "balanced",
    "object_like": "object_like_anomaly",
    "object_like_anomaly": "object_like_anomaly",
    "anomaly": "object_like_anomaly",
    "local_anomaly": "object_like_anomaly",
    "局部异常增强": "object_like_anomaly",
    "interface": "interface_layer_preservation",
    "layer": "interface_layer_preservation",
    "interface_layer": "interface_layer_preservation",
    "interface_layer_preservation": "interface_layer_preservation",
    "连续界面保留": "interface_layer_preservation",
    "层状界面保留": "interface_layer_preservation",
    "fracture": "interface_layer_preservation",
    "broken_zone": "interface_layer_preservation",
    "裂隙/破碎带保留": "interface_layer_preservation",
    "landslide": "landslide_bedrock_sliding_surface",
    "landslide_interface": "landslide_bedrock_sliding_surface",
    "bedrock_interface": "landslide_bedrock_sliding_surface",
    "sliding_surface": "landslide_bedrock_sliding_surface",
    "landslide_bedrock_sliding_surface": "landslide_bedrock_sliding_surface",
    "滑坡基覆界面 / 潜在滑移面": "landslide_bedrock_sliding_surface",
    "滑坡基覆界面 / 潜在滑动面": "landslide_bedrock_sliding_surface",
    "wet_weak_zone": "wet_weak_zone",
    "water_weak_zone": "wet_weak_zone",
    "含水软弱带": "wet_weak_zone",
    "deep_weak": "deep_weak_reflector",
    "weak_deep": "deep_weak_reflector",
    "deep_weak_reflector": "deep_weak_reflector",
    "deep_weak_reflection": "deep_weak_reflector",
    "深部弱反射增强": "deep_weak_reflector",
}


@dataclass(frozen=True)
class AutoTuneV1Profile:
    """A target-oriented scoring profile from the V1 config file."""

    profile_id: str
    label_zh: str
    weights: dict[str, float]
    rationale: str = ""
    requires_batch_calibration: bool = True

    @property
    def normalized_weights(self) -> dict[str, float]:
        total = float(sum(max(0.0, value) for value in self.weights.values()))
        if total <= 0.0:
            return {field_name: 1.0 / len(PROFILE_WEIGHT_FIELDS) for field_name in PROFILE_WEIGHT_FIELDS}
        return {key: float(value) / total for key, value in self.weights.items()}

    def to_dict(self, *, normalized: bool = False) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "label_zh": self.label_zh,
            "weights": dict(self.normalized_weights if normalized else self.weights),
            "rationale": self.rationale,
            "requires_batch_calibration": self.requires_batch_calibration,
        }


@dataclass(frozen=True)
class AutoTuneV1Recipe:
    """A bounded, auditable workflow recipe declaration."""

    recipe_id: str
    name: str
    profiles: tuple[str, ...]
    steps: tuple[str, ...]
    risk: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.recipe_id,
            "name": self.name,
            "profiles": list(self.profiles),
            "steps": list(self.steps),
            "risk": self.risk,
        }


@dataclass(frozen=True)
class AutoTuneV1Config:
    """Validated AutoTune V1 configuration contract."""

    version: str
    status: str
    profiles: dict[str, AutoTuneV1Profile]
    recipes: dict[str, AutoTuneV1Recipe]
    candidate_spec: dict[str, Any] = field(default_factory=dict)
    scoring_spec: dict[str, Any] = field(default_factory=dict)
    manifest_required_fields: tuple[str, ...] = field(default_factory=tuple)
    warning_tags: tuple[str, ...] = field(default_factory=tuple)
    source_path: str = ""

    def resolve_profile_id(self, target_goal: str | None) -> str:
        raw = str(target_goal or "balanced").strip()
        if raw in self.profiles:
            return raw
        return PROFILE_ALIASES.get(raw.lower(), PROFILE_ALIASES.get(raw, "balanced"))

    def profile_for_goal(self, target_goal: str | None) -> AutoTuneV1Profile:
        return self.profiles[self.resolve_profile_id(target_goal)]

    def recipes_for_profile(self, target_goal: str | None) -> tuple[AutoTuneV1Recipe, ...]:
        profile_id = self.resolve_profile_id(target_goal)
        rows = [recipe for recipe in self.recipes.values() if profile_id in recipe.profiles]
        return tuple(rows)

    def scoring_mode_spec(self, mode: str) -> dict[str, Any]:
        return dict(self.scoring_spec.get(mode, {}) or {})

    def to_dict(self, *, normalized_weights: bool = False) -> dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "profiles": {
                profile_id: profile.to_dict(normalized=normalized_weights)
                for profile_id, profile in self.profiles.items()
            },
            "recipes": [recipe.to_dict() for recipe in self.recipes.values()],
            "candidate_spec": self.candidate_spec,
            "scoring_spec": self.scoring_spec,
            "manifest_required_fields": list(self.manifest_required_fields),
            "warning_tags": list(self.warning_tags),
            "source_path": self.source_path,
        }


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"AutoTune V1 config field '{name}' must be a mapping")
    return value


def _as_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"AutoTune V1 config field '{name}' must be a sequence")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load AutoTune V1 YAML config")
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"AutoTune V1 config is not a mapping: {path}")
    return loaded


def validate_autotune_v1_config_data(data: Mapping[str, Any]) -> None:
    """Validate raw config data and raise ``ValueError`` on contract breaks."""

    version = str(data.get("version") or "").strip()
    if not version:
        raise ValueError("AutoTune V1 config missing 'version'")
    status = str(data.get("status") or "").strip()
    if not status:
        raise ValueError("AutoTune V1 config missing 'status'")

    profiles = _as_mapping(data.get("profiles"), "profiles")
    if not profiles:
        raise ValueError("AutoTune V1 config must contain at least one profile")
    for profile_id, row in profiles.items():
        row_map = _as_mapping(row, f"profiles.{profile_id}")
        label = str(row_map.get("label_zh") or "").strip()
        if not label:
            raise ValueError(f"Profile '{profile_id}' missing label_zh")
        weights = _as_mapping(row_map.get("weights"), f"profiles.{profile_id}.weights")
        missing = [field_name for field_name in PROFILE_WEIGHT_FIELDS if field_name not in weights]
        if missing:
            raise ValueError(f"Profile '{profile_id}' missing weight fields: {missing}")
        for field_name in PROFILE_WEIGHT_FIELDS:
            value = weights[field_name]
            if not isinstance(value, (int, float)):
                raise ValueError(f"Profile '{profile_id}' weight '{field_name}' must be numeric")
            if float(value) < 0.0:
                raise ValueError(f"Profile '{profile_id}' weight '{field_name}' must be non-negative")

    recipes = _as_sequence(data.get("recipes"), "recipes")
    recipe_ids: set[str] = set()
    for idx, row in enumerate(recipes):
        row_map = _as_mapping(row, f"recipes[{idx}]")
        recipe_id = str(row_map.get("id") or "").strip()
        if not recipe_id:
            raise ValueError(f"Recipe at index {idx} missing id")
        if recipe_id in recipe_ids:
            raise ValueError(f"Duplicate recipe id: {recipe_id}")
        recipe_ids.add(recipe_id)
        referenced_profiles = tuple(str(item) for item in _as_sequence(row_map.get("profiles"), f"recipes.{recipe_id}.profiles"))
        if not referenced_profiles:
            raise ValueError(f"Recipe '{recipe_id}' must reference at least one profile")
        unknown = [profile_id for profile_id in referenced_profiles if profile_id not in profiles]
        if unknown:
            raise ValueError(f"Recipe '{recipe_id}' references unknown profiles: {unknown}")
        steps = tuple(str(item) for item in _as_sequence(row_map.get("steps"), f"recipes.{recipe_id}.steps"))
        if not steps:
            raise ValueError(f"Recipe '{recipe_id}' must contain at least one step")

    scoring_spec = _as_mapping(data.get("scoring_spec", {}), "scoring_spec")
    synthetic = _as_mapping(scoring_spec.get("synthetic_paired", {}), "scoring_spec.synthetic_paired")
    real = _as_mapping(scoring_spec.get("real_no_prior", {}), "scoring_spec.real_no_prior")
    if "full_reference_metrics" not in synthetic:
        raise ValueError("synthetic_paired scoring spec must list full_reference_metrics")
    forbidden = set(str(item) for item in real.get("forbidden_metrics", []) or [])
    required_forbidden = {
        "mae_against_unknown_truth",
        "mse_against_unknown_truth",
        "psnr_against_unknown_truth",
        "ssim_against_unknown_truth",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("real_no_prior scoring spec must explicitly forbid truth-reference metrics")


def build_autotune_v1_config(data: Mapping[str, Any], *, source_path: str = "") -> AutoTuneV1Config:
    validate_autotune_v1_config_data(data)

    profiles_data = _as_mapping(data.get("profiles"), "profiles")
    profiles: dict[str, AutoTuneV1Profile] = {}
    for profile_id, row in profiles_data.items():
        row_map = _as_mapping(row, f"profiles.{profile_id}")
        weights = {
            field_name: float(_as_mapping(row_map.get("weights"), f"profiles.{profile_id}.weights")[field_name])
            for field_name in PROFILE_WEIGHT_FIELDS
        }
        profiles[str(profile_id)] = AutoTuneV1Profile(
            profile_id=str(profile_id),
            label_zh=str(row_map.get("label_zh") or ""),
            weights=weights,
            rationale=str(row_map.get("rationale") or ""),
            requires_batch_calibration=bool(row_map.get("requires_batch_calibration", True)),
        )

    recipes: dict[str, AutoTuneV1Recipe] = {}
    for row in _as_sequence(data.get("recipes"), "recipes"):
        row_map = _as_mapping(row, "recipes[]")
        recipe_id = str(row_map.get("id") or "")
        recipes[recipe_id] = AutoTuneV1Recipe(
            recipe_id=recipe_id,
            name=str(row_map.get("name") or recipe_id),
            profiles=tuple(str(item) for item in row_map.get("profiles", ()) or ()),
            steps=tuple(str(item) for item in row_map.get("steps", ()) or ()),
            risk=str(row_map.get("risk") or ""),
        )

    return AutoTuneV1Config(
        version=str(data.get("version") or ""),
        status=str(data.get("status") or ""),
        profiles=profiles,
        recipes=recipes,
        candidate_spec=dict(data.get("candidate_spec", {}) or {}),
        scoring_spec=dict(data.get("scoring_spec", {}) or {}),
        manifest_required_fields=tuple(str(item) for item in data.get("manifest_required_fields", ()) or ()),
        warning_tags=tuple(str(item) for item in data.get("warning_tags", ()) or ()),
        source_path=source_path,
    )


def load_autotune_v1_config(path: str | Path | None = None) -> AutoTuneV1Config:
    """Load and validate the AutoTune V1 final-candidate config file."""

    config_path = Path(path) if path is not None else DEFAULT_AUTOTUNE_V1_CONFIG_PATH
    data = _read_yaml(config_path)
    return build_autotune_v1_config(data, source_path=str(config_path))


def profile_weight_table_v1(*, normalized: bool = True) -> dict[str, dict[str, float]]:
    """Return V1 weights keyed by stable profile id."""

    config = load_autotune_v1_config()
    return {
        profile_id: dict(profile.normalized_weights if normalized else profile.weights)
        for profile_id, profile in config.profiles.items()
    }


def recipe_table_v1() -> dict[str, dict[str, Any]]:
    """Return V1 recipes keyed by stable recipe id."""

    config = load_autotune_v1_config()
    return {recipe_id: recipe.to_dict() for recipe_id, recipe in config.recipes.items()}


__all__ = [
    "AutoTuneV1Config",
    "AutoTuneV1Profile",
    "AutoTuneV1Recipe",
    "DEFAULT_AUTOTUNE_V1_CONFIG_PATH",
    "PROFILE_ALIASES",
    "PROFILE_WEIGHT_FIELDS",
    "build_autotune_v1_config",
    "load_autotune_v1_config",
    "profile_weight_table_v1",
    "recipe_table_v1",
    "validate_autotune_v1_config_data",
]
