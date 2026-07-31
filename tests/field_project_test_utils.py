from __future__ import annotations

from pathlib import Path

from core.field_project_models import FieldLineRecord
from core.field_project_store import FieldProjectStore


def create_test_project(
    root: str | Path,
    *,
    line_ids: tuple[str, ...] = ("L03",),
    name: str = "测试项目",
) -> FieldProjectStore:
    """Create an explicit test-only project fixture without product demo bootstrap."""
    store = FieldProjectStore.create_empty(root, name=name, location="测试工区", operator="测试员")
    for index, line_id in enumerate(line_ids, start=1):
        store.upsert_line(
            FieldLineRecord(
                line_id=line_id,
                name=f"测试测线 {line_id}",
                length_m=80.0 + index * 10.0,
                data_quality="未检查",
                rtk_status="未定位",
                processing_status="未处理",
            )
        )
    return store
