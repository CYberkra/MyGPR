#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auto-tune comparison research artifact export tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from core.auto_tune_comparison import run_auto_tune_comparison
from core.auto_tune_comparison_export import export_auto_tune_comparison_artifacts


def _build_export_fixture(samples: int = 84, traces: int = 24) -> np.ndarray:
    rng = np.random.default_rng(29)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
    x = np.linspace(-1.0, 1.0, traces, dtype=np.float64)[None, :]
    data = 0.36 * np.sin(2.0 * np.pi * 0.7 * t)
    data += 0.12 * np.sin(2.0 * np.pi * 9.0 * t)
    data = np.repeat(data, traces, axis=1)
    data += 0.04 * rng.normal(size=(samples, traces))
    hyperbola = 28 + np.round(10.0 * np.square(x)).astype(int).reshape(-1)
    for trace_idx, row in enumerate(hyperbola):
        data[row : row + 3, trace_idx] += np.array([0.25, 0.85, 0.25])
    data[56:60, 7:18] += 0.18 * np.hanning(4)[:, None]
    return data.astype(np.float32)


def test_export_auto_tune_comparison_artifacts_writes_research_bundle(tmp_path: Path):
    raw = _build_export_fixture()
    result = run_auto_tune_comparison(
        raw,
        pipeline=["dewow"],
        manual_params_by_method={"dewow": {"window": 1}},
        roi_spec={
            "mode": "manual",
            "bounds": {
                "time_start_idx": 18,
                "time_end_idx": 68,
                "dist_start_idx": 3,
                "dist_end_idx": 21,
            },
            "label": "synthetic-target-roi",
        },
        search_mode="fast",
    )
    result.metric_delta["comparison_score"] = np.inf
    result.manual.metrics["comparison_score"] = np.nan
    result.automatic.params_by_method["dewow"]["nonfinite_probe"] = np.array(
        [1.0, np.inf]
    )

    bundle = export_auto_tune_comparison_artifacts(
        result,
        out_dir=tmp_path,
        bundle_name="case001",
        input_ref="synthetic://case001",
        notes=["GPRMAX forward-model cases will reuse this export contract."],
    )

    expected = {
        "summary_json",
        "manual_png",
        "auto_png",
        "side_by_side_png",
        "params_csv",
        "metrics_csv",
        "report_md",
    }
    assert expected <= set(bundle["artifacts"])

    for key in expected:
        path = Path(bundle["artifacts"][key])
        assert path.exists(), key
        assert path.stat().st_size > 0, key

    summary = json.loads(
        Path(bundle["artifacts"]["summary_json"]).read_text(encoding="utf-8")
    )
    assert summary["input_ref"] == "synthetic://case001"
    assert summary["verdict"] == result.verdict
    assert summary["roi_info"]["label"] == "synthetic-target-roi"
    assert summary["display_spec"]["locked_scale"] is True
    assert "result" not in summary["manual"]
    assert "result" not in summary["automatic"]
    assert summary["metric_delta"]["comparison_score"] is None
    assert summary["manual"]["metrics"]["comparison_score"] is None
    assert summary["automatic"]["params_by_method"]["dewow"]["nonfinite_probe"] == [
        1.0,
        None,
    ]
    json.dumps(summary, allow_nan=False)

    with Path(bundle["artifacts"]["params_csv"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        params_rows = list(csv.DictReader(handle))
    assert {
        (row["candidate"], row["method_key"], row["param_name"])
        for row in params_rows
    } >= {
        ("manual", "dewow", "window"),
        ("automatic", "dewow", "window"),
    }

    with Path(bundle["artifacts"]["metrics_csv"]).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        metrics_rows = list(csv.DictReader(handle))
    assert "comparison_score" in {row["metric"] for row in metrics_rows}

    report_text = Path(bundle["artifacts"]["report_md"]).read_text(encoding="utf-8")
    assert "# Auto-Tune Comparison Report" in report_text
    assert "synthetic://case001" in report_text
