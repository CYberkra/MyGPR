#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a GPRPy-baseline comparison report for core MyGPR operators."""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gprpy_compat import (  # noqa: E402
    apply_gprpy_agc_gain,
    apply_gprpy_dewow,
    apply_gprpy_rem_mean_trace,
)
from core.processing_engine import run_processing_method  # noqa: E402
from read_file_data import readcsv, save_image  # noqa: E402


DEFAULT_INPUT = ROOT / "sample_data" / "gui_sidecar_all_data_main.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "output" / "gprpy_baseline_reports"


@dataclass(frozen=True)
class BaselineStep:
    """One comparable MyGPR/GPRPy processing operator."""

    key: str
    label: str
    mygpr_method: str
    params: dict[str, Any]
    baseline_func: Callable[[np.ndarray], np.ndarray]
    baseline_name: str


def build_steps(
    *,
    dewow_window: int = 23,
    ntraces: int = 11,
    agc_window: int = 11,
) -> list[BaselineStep]:
    """Build the default GPRPy baseline processing chain."""
    return [
        BaselineStep(
            key="dewow",
            label="低频漂移抑制 / dewow",
            mygpr_method="dewow",
            params={"window": int(dewow_window)},
            baseline_func=lambda data: apply_gprpy_dewow(data, int(dewow_window)),
            baseline_name=f"GPRPy dewow(window={int(dewow_window)})",
        ),
        BaselineStep(
            key="rem_mean_trace",
            label="背景抑制 / remMeanTrace",
            mygpr_method="subtracting_average_2D",
            params={"ntraces": int(ntraces)},
            baseline_func=lambda data: apply_gprpy_rem_mean_trace(data, int(ntraces)),
            baseline_name=f"GPRPy remMeanTrace(ntraces={int(ntraces)})",
        ),
        BaselineStep(
            key="agc_gain",
            label="自动增益 / AGC",
            mygpr_method="agcGain",
            params={"window": int(agc_window)},
            baseline_func=lambda data: apply_gprpy_agc_gain(data, int(agc_window)),
            baseline_name=f"GPRPy agcGain(window={int(agc_window)})",
        ),
    ]


def compare_gprpy_baseline(
    data: np.ndarray,
    *,
    source_label: str,
    dewow_window: int = 23,
    ntraces: int = 11,
    agc_window: int = 11,
) -> dict[str, Any]:
    """Run GPRPy baseline and MyGPR current implementation step-by-step."""
    raw = np.asarray(data, dtype=np.float32)
    if raw.ndim != 2 or raw.size == 0:
        raise ValueError(f"expected non-empty 2D B-scan, got shape={raw.shape}")

    mygpr_current = np.array(raw, copy=True)
    baseline_current = np.array(raw, copy=True)
    records: list[dict[str, Any]] = []

    for index, step in enumerate(
        build_steps(
            dewow_window=dewow_window,
            ntraces=ntraces,
            agc_window=agc_window,
        ),
        start=1,
    ):
        before_mygpr = np.array(mygpr_current, copy=True)
        before_baseline = np.array(baseline_current, copy=True)
        baseline_after = np.asarray(step.baseline_func(baseline_current), dtype=np.float32)
        mygpr_after, meta = run_processing_method(
            mygpr_current,
            step.mygpr_method,
            dict(step.params),
        )
        mygpr_after = np.asarray(mygpr_after, dtype=np.float32)
        diff = mygpr_after - baseline_after
        records.append(
            {
                "index": index,
                "key": step.key,
                "label": step.label,
                "mygpr_method": step.mygpr_method,
                "gprpy_baseline": step.baseline_name,
                "params": dict(step.params),
                "meta": _jsonable(meta),
                "before_mygpr": before_mygpr,
                "before_baseline": before_baseline,
                "mygpr_after": mygpr_after,
                "baseline_after": baseline_after,
                "difference": diff,
                "metrics": _difference_metrics(mygpr_after, baseline_after),
            }
        )
        mygpr_current = mygpr_after
        baseline_current = baseline_after

    overall_metrics = _difference_metrics(mygpr_current, baseline_current)
    return {
        "report_type": "gprpy_baseline_comparison",
        "schema_version": 1,
        "source_label": source_label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_summary": _array_summary(raw),
        "steps": records,
        "overall_metrics": overall_metrics,
        "conclusion": _build_conclusion(records, overall_metrics),
    }


def write_gprpy_baseline_report(
    comparison: dict[str, Any],
    report_dir: str | Path,
    *,
    save_images_flag: bool = True,
) -> dict[str, Any]:
    """Write HTML, JSON, and optional B-scan images for one comparison."""
    output_dir = Path(report_dir)
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    clean_steps: list[dict[str, Any]] = []
    image_records: dict[str, dict[str, str]] = {}
    for step in comparison["steps"]:
        clean = {
            key: _jsonable(value)
            for key, value in step.items()
            if key
            not in {
                "before_mygpr",
                "before_baseline",
                "mygpr_after",
                "baseline_after",
                "difference",
            }
        }
        step_images: dict[str, str] = {}
        if save_images_flag:
            prefix = f"{int(step['index']):02d}-{step['key']}"
            step_images = _save_step_images(step, assets_dir, prefix)
        clean["images"] = step_images
        image_records[str(step["key"])] = step_images
        clean_steps.append(clean)

    summary = {
        key: _jsonable(value)
        for key, value in comparison.items()
        if key != "steps"
    }
    summary["steps"] = clean_steps
    summary["artifacts"] = {
        "html": str((output_dir / "index.html").resolve()),
        "summary_json": str((output_dir / "summary.json").resolve()),
        "assets_dir": str(assets_dir.resolve()),
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    html_path = output_dir / "index.html"
    html_path.write_text(_render_html(summary), encoding="utf-8")
    return summary


def run_report(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dewow_window: int = 23,
    ntraces: int = 11,
    agc_window: int = 11,
    save_images_flag: bool = True,
) -> dict[str, Any]:
    """Load a CSV B-scan and generate a timestamped comparison report."""
    data = readcsv(str(input_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(output_root) / f"gprpy_baseline_{timestamp}"
    comparison = compare_gprpy_baseline(
        data,
        source_label=str(input_path),
        dewow_window=dewow_window,
        ntraces=ntraces,
        agc_window=agc_window,
    )
    return write_gprpy_baseline_report(
        comparison,
        report_dir,
        save_images_flag=save_images_flag,
    )


def _difference_metrics(current: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    current_arr = np.asarray(current, dtype=np.float64)
    baseline_arr = np.asarray(baseline, dtype=np.float64)
    diff = current_arr - baseline_arr
    finite = np.isfinite(diff)
    if not finite.any():
        return {
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
            "rms_diff": float("nan"),
            "relative_rms_diff": float("nan"),
            "correlation": float("nan"),
        }
    diff_f = diff[finite]
    baseline_rms = float(np.sqrt(np.mean(baseline_arr[np.isfinite(baseline_arr)] ** 2)))
    current_f = current_arr[finite].reshape(-1)
    baseline_f = baseline_arr[finite].reshape(-1)
    if current_f.size > 1 and np.std(current_f) > 0 and np.std(baseline_f) > 0:
        corr = float(np.corrcoef(current_f, baseline_f)[0, 1])
    else:
        corr = 1.0 if np.allclose(current_f, baseline_f) else float("nan")
    return {
        "max_abs_diff": float(np.max(np.abs(diff_f))),
        "mean_abs_diff": float(np.mean(np.abs(diff_f))),
        "rms_diff": float(np.sqrt(np.mean(diff_f**2))),
        "relative_rms_diff": float(
            np.sqrt(np.mean(diff_f**2)) / max(baseline_rms, 1.0e-12)
        ),
        "correlation": corr,
    }


def _array_summary(data: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(data)
    finite = arr[np.isfinite(arr)]
    return {
        "shape": [int(value) for value in arr.shape],
        "dtype": str(arr.dtype),
        "finite_ratio": float(finite.size / arr.size) if arr.size else 0.0,
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite)) if finite.size else float("nan"),
    }


def _build_conclusion(
    records: list[dict[str, Any]],
    overall_metrics: dict[str, float],
) -> str:
    max_step_diff = max(
        (float(record["metrics"]["max_abs_diff"]) for record in records),
        default=float("nan"),
    )
    if np.isfinite(max_step_diff) and max_step_diff <= 1.0e-6:
        return "MyGPR 当前 dewow / 背景抑制 / AGC 数值口径已与 GPRPy 基线对齐。"
    if float(overall_metrics.get("relative_rms_diff", 1.0)) < 1.0e-3:
        return "MyGPR 与 GPRPy 基线整体高度接近，但仍存在可见前需审查的微小数值差异。"
    return "MyGPR 与 GPRPy 基线存在明显差异，需要检查算法边界处理或参数解释。"


def _save_step_images(
    step: dict[str, Any],
    assets_dir: Path,
    prefix: str,
) -> dict[str, str]:
    images = {
        "before_mygpr": assets_dir / f"{prefix}-before-mygpr.png",
        "before_baseline": assets_dir / f"{prefix}-before-gprpy.png",
        "mygpr_after": assets_dir / f"{prefix}-after-mygpr.png",
        "baseline_after": assets_dir / f"{prefix}-after-gprpy.png",
        "difference": assets_dir / f"{prefix}-difference.png",
    }
    save_image(step["before_mygpr"], str(images["before_mygpr"]), f"{prefix} MyGPR before")
    save_image(
        step["before_baseline"],
        str(images["before_baseline"]),
        f"{prefix} GPRPy before",
    )
    save_image(step["mygpr_after"], str(images["mygpr_after"]), f"{prefix} MyGPR after")
    save_image(
        step["baseline_after"],
        str(images["baseline_after"]),
        f"{prefix} GPRPy after",
    )
    diff = np.asarray(step["difference"], dtype=np.float32)
    vmax = float(np.max(np.abs(diff))) if diff.size else 1.0
    vmax = max(vmax, 1.0e-12)
    save_image(
        diff,
        str(images["difference"]),
        f"{prefix} MyGPR - GPRPy",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    return {
        key: Path(value).relative_to(assets_dir.parent).as_posix()
        for key, value in images.items()
    }


def _render_html(summary: dict[str, Any]) -> str:
    rows = []
    cards = []
    for step in summary.get("steps", []):
        metrics = step.get("metrics") or {}
        rows.append(
            "<tr>"
            f"<td>{step.get('index')}</td>"
            f"<td>{html.escape(str(step.get('label')))}</td>"
            f"<td>{html.escape(str(step.get('mygpr_method')))}</td>"
            f"<td>{html.escape(str(step.get('gprpy_baseline')))}</td>"
            f"<td>{html.escape(json.dumps(step.get('params'), ensure_ascii=False))}</td>"
            f"<td>{_fmt(metrics.get('max_abs_diff'))}</td>"
            f"<td>{_fmt(metrics.get('relative_rms_diff'))}</td>"
            f"<td>{_fmt(metrics.get('correlation'))}</td>"
            "</tr>"
        )
        images = step.get("images") or {}
        cards.append(_render_step_card(step, images))
    input_summary = summary.get("input_summary") or {}
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>MyGPR / GPRPy Baseline Report</title>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 28px; color: #172033; background: #f6f8fb; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    .panel {{ background: #fff; border: 1px solid #d8e0ea; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ border: 1px solid #d8e0ea; padding: 8px; vertical-align: top; font-size: 13px; }}
    th {{ background: #edf2f7; text-align: left; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; border: 1px solid #d8e0ea; border-radius: 6px; background: #fff; padding: 8px; }}
    img {{ width: 100%; display: block; }}
    figcaption {{ font-size: 12px; color: #526071; margin-top: 6px; }}
    .ok {{ color: #0f766e; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>MyGPR / GPRPy 基线对照报告</h1>
  <section class="panel">
    <h2>结论</h2>
    <p class="ok">{html.escape(str(summary.get("conclusion", "")))}</p>
    <div class="meta">
      <div><b>数据:</b> {html.escape(str(summary.get("source_label", "")))}</div>
      <div><b>生成时间:</b> {html.escape(str(summary.get("created_at", "")))}</div>
      <div><b>输入 shape:</b> {html.escape(str(input_summary.get("shape", "")))}</div>
      <div><b>finite ratio:</b> {_fmt(input_summary.get("finite_ratio"))}</div>
    </div>
  </section>
  <section class="panel">
    <h2>差异指标</h2>
    <table>
      <thead><tr><th>#</th><th>步骤</th><th>MyGPR</th><th>GPRPy baseline</th><th>参数</th><th>max abs diff</th><th>relative RMS diff</th><th>correlation</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </section>
  {''.join(cards)}
</body>
</html>
"""


def _render_step_card(step: dict[str, Any], images: dict[str, str]) -> str:
    if not images:
        return ""
    image_items = [
        ("before_mygpr", "MyGPR 运行前"),
        ("before_baseline", "GPRPy baseline 运行前"),
        ("mygpr_after", "MyGPR 运行后"),
        ("baseline_after", "GPRPy baseline 运行后"),
        ("difference", "差异图 MyGPR - GPRPy"),
    ]
    figures = []
    for key, label in image_items:
        path = images.get(key)
        if path:
            figures.append(
                f'<figure><img src="{html.escape(path)}" alt="{html.escape(label)}">'
                f"<figcaption>{html.escape(label)}</figcaption></figure>"
            )
    return (
        '<section class="panel">'
        f"<h2>{step.get('index')}. {html.escape(str(step.get('label')))}</h2>"
        '<div class="grid">'
        + "".join(figures)
        + "</div></section>"
    )


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "nan"
    if abs(number) >= 1000 or (0 < abs(number) < 0.001):
        return f"{number:.3e}"
    return f"{number:.6f}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, int):
        return int(value)
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="B-scan CSV")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dewow-window", type=int, default=23)
    parser.add_argument("--ntraces", type=int, default=11)
    parser.add_argument("--agc-window", type=int, default=11)
    parser.add_argument("--no-images", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_report(
        input_path=args.input,
        output_root=args.output_root,
        dewow_window=args.dewow_window,
        ntraces=args.ntraces,
        agc_window=args.agc_window,
        save_images_flag=not args.no_images,
    )
    print(summary["artifacts"]["html"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
