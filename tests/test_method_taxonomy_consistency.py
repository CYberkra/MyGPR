from __future__ import annotations

from core.methods_registry import (
    METHOD_CATEGORY_LABELS,
    PROCESSING_METHODS,
    get_auto_tune_stage,
    get_method_category,
)


def test_public_methods_have_registered_category_labels():
    for key, method in PROCESSING_METHODS.items():
        if str(method.get("visibility", "public")) != "public":
            continue
        category = get_method_category(key)
        assert category in METHOD_CATEGORY_LABELS, f"{key} category {category} has no label"


def test_frequency_and_artifact_methods_are_not_misfiled_as_background():
    assert get_method_category("fk_filter") == "filtering"
    assert get_auto_tune_stage("fk_filter") == "frequency"
    assert get_method_category("running_average_2D") == "denoising"
    assert get_auto_tune_stage("motion_compensation_vibration") == "artifact"
