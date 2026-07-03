#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo-project bootstrap and project-level utility mixin."""

from __future__ import annotations

from pathlib import Path


class FieldDemoStoreMixin:
    """Keep demo bootstrap isolated from project manifest persistence."""

    def ensure_demo_artifacts(self, *, sample_csv: str | Path | None = None) -> None:
        self.ensure_structure()
        if sample_csv is not None and Path(sample_csv).exists():
            try:
                self.import_line_file("L03", Path(sample_csv), name="过路口测线", copy_into_project=True)
            except Exception:
                pass
        self.ensure_demo_gpr_artifacts("L03")
        if not self.targets_path("L03").exists():
            self.save_targets("L03", self.default_targets("L03"))
        else:
            try:
                line = self.get_line("L03")
                line.target_count = len(self.load_targets("L03"))
                self.upsert_line(line)
            except KeyError:
                pass
        self.export_spatial_targets_xy("L03")
        self.save_manifest()

    def total_raw_size_mb(self) -> float:
        return sum(line.raw_size_mb for line in self.list_lines())

    def storage_usage_mb(self) -> float:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total / (1024 * 1024)

    def append_log(self, message: str) -> None:
        log_path = self.root / "logs" / "field_workbench.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{self.now()}] {message}\n")


__all__ = ["FieldDemoStoreMixin"]
