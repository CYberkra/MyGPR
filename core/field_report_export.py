#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auditable project report-package export for MyGPR field projects.

The beta report workflow intentionally writes a directory of CSV/JSON/HTML
artifacts and a lightweight PDF summary.  Every generated CSV/JSON/HTML file is plain,
portable and can be inspected or regenerated from the project store.
"""

from __future__ import annotations

import csv
import html
import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from core.field_project_models import FieldLineRecord, local_now, validate_line_id
from core.processing_artifact_index import index_processing_artifacts

REPORT_PACKAGE_SCHEMA = "mygpr.report_package.v1"


@dataclass(frozen=True)
class ReportPackageResult:
    """Result returned after generating one report package."""

    package_dir: str
    manifest_path: str
    html_path: str
    summary_json_path: str
    line_manifest_csv_path: str
    quality_csv_path: str
    targets_csv_path: str
    processing_csv_path: str
    spatial_csv_path: str
    file_count: int
    generated_at: str
    pdf_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(path)


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    except Exception:
        return 0


def _quality_report_to_row(line: FieldLineRecord, report: Any | None) -> dict[str, Any]:
    if report is None:
        return {
            "line_id": line.line_id,
            "line_name": line.name,
            "status": "未质检",
            "sample_count": "",
            "trace_count": "",
            "time_window_ns": "",
            "length_m": f"{float(line.length_m or 0.0):.3f}",
            "finite_ratio": "",
            "nan_ratio": "",
            "trajectory_points": "",
            "orientation": "",
            "orientation_message": "未生成质检报告",
            "issue_count": "",
        }
    return {
        "line_id": line.line_id,
        "line_name": line.name,
        "status": getattr(report, "status_label", "--"),
        "sample_count": int(getattr(report, "sample_count", 0)),
        "trace_count": int(getattr(report, "trace_count", 0)),
        "time_window_ns": f"{float(getattr(report, 'time_window_ns', 0.0)):.3f}",
        "length_m": f"{float(getattr(report, 'length_m', 0.0)):.3f}",
        "finite_ratio": f"{float(getattr(report, 'finite_ratio', 0.0)):.6f}",
        "nan_ratio": f"{float(getattr(report, 'nan_ratio', 0.0)):.6f}",
        "trajectory_points": int(getattr(report, "trajectory_points", 0)),
        "orientation": getattr(report, "orientation", ""),
        "orientation_message": getattr(report, "orientation_message", ""),
        "issue_count": len(getattr(report, "issues", []) or []),
    }


def _line_to_row(line: FieldLineRecord) -> dict[str, Any]:
    return {
        "line_id": line.line_id,
        "name": line.name,
        "length_m": f"{float(line.length_m or 0.0):.3f}",
        "data_quality": line.data_quality,
        "rtk_status": line.rtk_status,
        "processing_status": line.processing_status,
        "target_count": int(line.target_count or 0),
        "raw_rows": int(line.raw_rows or 0),
        "raw_size_mb": f"{float(line.raw_size_mb or 0.0):.3f}",
        "data_format": line.data_format,
        "gpr_dataset_path": line.gpr_dataset_path,
        "trajectory_path": line.trajectory_path,
        "processed_result": line.processed_result,
        "params_path": line.params_path,
        "updated_at": line.updated_at,
    }


def _target_to_row(line_id: str, target: dict[str, Any]) -> dict[str, Any]:
    safe_line_id = validate_line_id(line_id)
    return {
        "target_id": target.get("target_id") or target.get("name", ""),
        "line_id": safe_line_id,
        "distance_m": target.get("distance_m", target.get("mileage", "")),
        "depth_m": target.get("depth_m", target.get("depth", "")),
        "x": target.get("x", ""),
        "y": target.get("y", ""),
        "type": target.get("type", ""),
        "status": target.get("status", ""),
        "confidence": target.get("confidence", ""),
        "source_result_id": target.get("source_result_id", ""),
        "source_mode": target.get("source_mode", ""),
        "source_method_id": target.get("source_method_id", ""),
        "source_manifest_path": target.get("source_manifest_path", ""),
        "note": target.get("note", ""),
    }


def _artifact_to_row(record: Any) -> dict[str, Any]:
    return {
        "artifact_id": record.artifact_id,
        "line_id": record.line_id,
        "method_id": record.method_id,
        "method_name": record.method_name,
        "role": record.role,
        "status": record.status,
        "input_shape": "×".join(str(v) for v in record.input_shape),
        "output_shape": "×".join(str(v) for v in record.output_shape),
        "shape_changed": "yes" if record.shape_changed else "no",
        "created_at": record.created_at,
        "data_path": record.data_path,
        "params_path": record.params_path,
        "manifest_path": record.manifest_path,
        "output_data_sha256": record.output_data_sha256,
        "params_sha256": record.params_sha256,
        "manifest_sha256": record.manifest_sha256,
        "save_schema": record.save_schema,
    }


def _write_html_report(path: Path, *, summary: dict[str, Any], lines: list[dict[str, Any]], quality: list[dict[str, Any]], targets: list[dict[str, Any]], artifacts: list[dict[str, Any]], spatial: list[dict[str, Any]]) -> None:
    def table(headers: list[str], rows: list[dict[str, Any]], limit: int = 30) -> str:
        head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        body_rows = []
        for row in rows[:limit]:
            body_rows.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers) + "</tr>")
        if not body_rows:
            body_rows.append(f"<tr><td colspan='{len(headers)}'>无记录</td></tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    project = summary.get("project", {})
    metrics = summary.get("metrics", {})
    generated_at = summary.get("generated_at", "")
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 28px; color: #1f2937; }
    h1 { margin-bottom: 4px; } h2 { margin-top: 28px; border-bottom: 1px solid #d1d5db; padding-bottom: 6px; }
    .meta, .metric { color: #4b5563; } .metrics { display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0; }
    .metric { border: 1px solid #d1d5db; border-radius: 8px; padding: 10px 12px; min-width: 130px; background: #f9fafb; }
    .metric b { display: block; font-size: 22px; color: #111827; }
    table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 12px; }
    th, td { border: 1px solid #d1d5db; padding: 6px 7px; vertical-align: top; }
    th { background: #eef2ff; text-align: left; } code { background:#f3f4f6; padding:2px 4px; border-radius:4px; }
    """
    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(project.get('name', 'MyGPR 项目报告'))}</title><style>{css}</style></head>
<body>
<h1>{html.escape(project.get('name', 'MyGPR 项目'))} 成果报告</h1>
<div class="meta">生成时间：{html.escape(str(generated_at))}　项目编号：{html.escape(str(project.get('project_no', '--')))}　测区：{html.escape(str(project.get('location', '--')))}</div>
<div class="metrics">
  <div class="metric"><b>{metrics.get('line_count', 0)}</b>测线</div>
  <div class="metric"><b>{metrics.get('imported_line_count', 0)}</b>已导入测线</div>
  <div class="metric"><b>{metrics.get('qc_passed_count', 0)}</b>质检通过</div>
  <div class="metric"><b>{metrics.get('processing_artifact_count', 0)}</b>处理结果</div>
  <div class="metric"><b>{metrics.get('target_count', 0)}</b>目标标注</div>
  <div class="metric"><b>{metrics.get('spatial_export_count', 0)}</b>空间成果</div>
</div>
<h2>1. 项目概况</h2>
<p>操作员：{html.escape(str(project.get('operator', '--')))}；设备：{html.escape(str(project.get('device_model', '--')))}；坐标系统：<code>{html.escape(str(project.get('coordinate_system', '--')))}</code>。</p>
<h2>2. 测线清单</h2>{table(['line_id','name','length_m','data_quality','rtk_status','processing_status','target_count','data_format'], lines)}
<h2>3. 数据质检</h2>{table(['line_id','line_name','status','sample_count','trace_count','length_m','orientation','orientation_message','issue_count'], quality)}
<h2>4. 处理结果</h2>{table(['artifact_id','line_id','method_id','role','status','input_shape','output_shape','data_path'], artifacts)}
<h2>5. 目标标注</h2>{table(['target_id','line_id','distance_m','depth_m','x','y','type','status','source_method_id'], targets)}
<h2>6. 空间成果</h2>{table(['line_id','spatial_csv_path','row_count','has_xy_count','empty_xy_count'], spatial)}
<p class="meta">说明：当前版本输出 HTML/CSV/JSON/PDF 报告包；CSV/JSON 作为可审计源数据，PDF 作为交付摘要。</p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(html_text, encoding="utf-8")
    tmp.replace(path)


def _write_pdf_report(path: Path, *, summary: dict[str, Any], lines: list[dict[str, Any]], quality: list[dict[str, Any]], targets: list[dict[str, Any]], artifacts: list[dict[str, Any]], spatial: list[dict[str, Any]]) -> None:
    """Write a lightweight PDF report using matplotlib's built-in PDF backend.

    The PDF is a readable delivery summary; CSV/JSON/HTML remain the auditable
    source files.  This avoids adding a heavyweight report-generation runtime to
    the one-click launcher.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    def _choose_font() -> str:
        candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "AR PL UMing CN", "AR PL KaitiM GB", "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans"]
        installed = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in installed:
                return name
        return "DejaVu Sans"

    font_name = _choose_font()

    def _page(title: str):
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        fig.text(0.06, 0.955, title, fontsize=18, fontweight="bold", fontname=font_name)
        fig.text(0.06, 0.928, "MyGPR v0.9.9 成果报告", fontsize=9, color="#4b5563", fontname=font_name)
        return fig

    def _kv_lines(fig, start_y: float, rows: list[tuple[str, Any]]) -> float:
        y = start_y
        for key, value in rows:
            fig.text(0.075, y, str(key), fontsize=10, fontweight="bold", fontname=font_name)
            fig.text(0.24, y, str(value), fontsize=10, fontname=font_name)
            y -= 0.027
        return y

    def _table_page(pdf: PdfPages, title: str, headers: list[str], rows: list[dict[str, Any]], limit: int = 24) -> None:
        fig = _page(title)
        ax = fig.add_axes([0.05, 0.08, 0.90, 0.80])
        ax.axis("off")
        data = [[str(row.get(h, ""))[:42] for h in headers] for row in rows[:limit]]
        if not data:
            data = [["无记录"] + [""] * (len(headers) - 1)]
        table = ax.table(cellText=data, colLabels=headers, loc="upper left", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.35)
        for (_row, _col), cell in table.get_celld().items():
            cell.set_edgecolor("#d1d5db")
            cell.set_linewidth(0.4)
            cell.get_text().set_fontname(font_name)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    project = summary.get("project", {})
    metrics = summary.get("metrics", {})
    with PdfPages(tmp) as pdf:
        fig = _page(f"{project.get('name', 'MyGPR 项目')} 成果报告")
        y = _kv_lines(
            fig,
            0.86,
            [
                ("生成时间", summary.get("generated_at", "")),
                ("项目编号", project.get("project_no", "--")),
                ("测区位置", project.get("location", "--")),
                ("操作员", project.get("operator", "--")),
                ("设备型号", project.get("device_model", "--")),
                ("坐标系统", project.get("coordinate_system", "--")),
                ("高程基准", project.get("vertical_datum", "--")),
                ("项目路径", project.get("project_path", "--")),
            ],
        )
        metric_rows = [
            ("测线数量", metrics.get("line_count", 0)),
            ("已导入测线", metrics.get("imported_line_count", 0)),
            ("质检通过", metrics.get("qc_passed_count", 0)),
            ("质检警告", metrics.get("qc_warning_count", 0)),
            ("处理结果", metrics.get("processing_artifact_count", 0)),
            ("目标标注", metrics.get("target_count", 0)),
            ("空间成果", metrics.get("spatial_export_count", 0)),
        ]
        fig.text(0.075, y - 0.035, "核心统计", fontsize=13, fontweight="bold", fontname=font_name)
        _kv_lines(fig, y - 0.075, metric_rows)
        fig.text(0.075, 0.20, "交付说明", fontsize=13, fontweight="bold", fontname=font_name)
        fig.text(0.075, 0.17, "PDF 为交付摘要；CSV、JSON、HTML 文件保留完整可审计数据。", fontsize=10, fontname=font_name)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        _table_page(pdf, "测线清单", ["line_id", "name", "length_m", "data_quality", "rtk_status", "processing_status", "target_count"], lines)
        _table_page(pdf, "数据质检", ["line_id", "line_name", "status", "sample_count", "trace_count", "orientation_message"], quality)
        _table_page(pdf, "处理结果", ["artifact_id", "line_id", "method_id", "role", "status", "output_shape"], artifacts)
        _table_page(pdf, "目标标注", ["target_id", "line_id", "distance_m", "depth_m", "x", "y", "type", "status"], targets)
        _table_page(pdf, "空间成果", ["line_id", "spatial_csv_path", "row_count", "has_xy_count", "empty_xy_count"], spatial)
    tmp.replace(path)


def generate_project_report_package(store: Any, *, package_name: str | None = None) -> ReportPackageResult:
    """Generate an auditable CSV/JSON/HTML report package under reports/."""
    generated_at = local_now()
    stamp = _timestamp()
    package_dir = store.root / "reports" / (package_name or f"report_{stamp}")
    tables_dir = package_dir / "tables"
    json_dir = package_dir / "json"
    html_dir = package_dir / "html"
    for directory in (tables_dir, json_dir, html_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = store.manifest
    lines = store.list_lines()
    artifacts = index_processing_artifacts(store.root)
    artifact_rows = [_artifact_to_row(record) for record in artifacts]
    line_rows = [_line_to_row(line) for line in lines]

    quality_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for line in lines:
        report = store.load_quality_report(line.line_id)
        quality_rows.append(_quality_report_to_row(line, report))
        targets = store.load_targets(line.line_id)
        target_rows.extend(_target_to_row(line.line_id, target) for target in targets)
        spatial_path = store.root / "spatial" / f"{line.line_id}_targets_xy.csv"
        row_count = _count_csv_rows(spatial_path)
        has_xy_count = 0
        empty_xy_count = 0
        if spatial_path.exists():
            try:
                with spatial_path.open("r", encoding="utf-8-sig", newline="") as fh:
                    for row in csv.DictReader(fh):
                        if row.get("x") and row.get("y"):
                            has_xy_count += 1
                        else:
                            empty_xy_count += 1
            except Exception:
                pass
        spatial_rows.append(
            {
                "line_id": line.line_id,
                "spatial_csv_path": _safe_rel(spatial_path, store.root) if spatial_path.exists() else "",
                "row_count": row_count,
                "has_xy_count": has_xy_count,
                "empty_xy_count": empty_xy_count,
            }
        )

    project_payload = {
        "project_id": manifest.project_id,
        "project_no": manifest.project_no,
        "name": manifest.name,
        "location": manifest.location,
        "operator": manifest.operator,
        "device_model": manifest.device_model,
        "coordinate_system": manifest.coordinate_system,
        "vertical_datum": manifest.vertical_datum,
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
        "project_path": str(store.root),
    }
    metrics = {
        "line_count": len(lines),
        "imported_line_count": sum(1 for line in lines if line.gpr_dataset_path or line.raw_path),
        "qc_passed_count": sum(1 for row in quality_rows if row.get("status") == "通过"),
        "qc_warning_count": sum(1 for row in quality_rows if row.get("status") == "警告"),
        "qc_failed_count": sum(1 for row in quality_rows if row.get("status") == "失败"),
        "processing_artifact_count": len(artifact_rows),
        "target_count": len(target_rows),
        "spatial_export_count": sum(1 for row in spatial_rows if row.get("spatial_csv_path")),
    }
    summary = {
        "schema": REPORT_PACKAGE_SCHEMA,
        "generated_at": generated_at,
        "project": project_payload,
        "metrics": metrics,
        "files": {},
    }

    line_csv = tables_dir / "line_manifest.csv"
    quality_csv = tables_dir / "quality_summary.csv"
    targets_csv = tables_dir / "targets_summary.csv"
    processing_csv = tables_dir / "processing_artifacts.csv"
    spatial_csv = tables_dir / "spatial_exports.csv"
    summary_json = json_dir / "project_report_summary.json"
    html_report = html_dir / "project_report.html"
    pdf_report = package_dir / "project_report.pdf"
    report_manifest = package_dir / "report_manifest.json"

    _atomic_write_csv(line_csv, ["line_id", "name", "length_m", "data_quality", "rtk_status", "processing_status", "target_count", "raw_rows", "raw_size_mb", "data_format", "gpr_dataset_path", "trajectory_path", "processed_result", "params_path", "updated_at"], line_rows)
    _atomic_write_csv(quality_csv, ["line_id", "line_name", "status", "sample_count", "trace_count", "time_window_ns", "length_m", "finite_ratio", "nan_ratio", "trajectory_points", "orientation", "orientation_message", "issue_count"], quality_rows)
    _atomic_write_csv(targets_csv, ["target_id", "line_id", "distance_m", "depth_m", "x", "y", "type", "status", "confidence", "source_result_id", "source_mode", "source_method_id", "source_manifest_path", "note"], target_rows)
    _atomic_write_csv(processing_csv, ["artifact_id", "line_id", "method_id", "method_name", "role", "status", "input_shape", "output_shape", "shape_changed", "created_at", "data_path", "params_path", "manifest_path", "output_data_sha256", "params_sha256", "manifest_sha256", "save_schema"], artifact_rows)
    _atomic_write_csv(spatial_csv, ["line_id", "spatial_csv_path", "row_count", "has_xy_count", "empty_xy_count"], spatial_rows)

    summary["files"] = {
        "line_manifest_csv": _safe_rel(line_csv, store.root),
        "quality_summary_csv": _safe_rel(quality_csv, store.root),
        "targets_summary_csv": _safe_rel(targets_csv, store.root),
        "processing_artifacts_csv": _safe_rel(processing_csv, store.root),
        "spatial_exports_csv": _safe_rel(spatial_csv, store.root),
        "html_report": _safe_rel(html_report, store.root),
        "pdf_report": _safe_rel(pdf_report, store.root),
    }
    store.write_json(summary_json, summary)
    _write_html_report(html_report, summary=summary, lines=line_rows, quality=quality_rows, targets=target_rows, artifacts=artifact_rows, spatial=spatial_rows)
    _write_pdf_report(pdf_report, summary=summary, lines=line_rows, quality=quality_rows, targets=target_rows, artifacts=artifact_rows, spatial=spatial_rows)

    files = [p for p in package_dir.rglob("*") if p.is_file()]
    result = ReportPackageResult(
        package_dir=_safe_rel(package_dir, store.root),
        manifest_path=_safe_rel(report_manifest, store.root),
        html_path=_safe_rel(html_report, store.root),
        summary_json_path=_safe_rel(summary_json, store.root),
        line_manifest_csv_path=_safe_rel(line_csv, store.root),
        quality_csv_path=_safe_rel(quality_csv, store.root),
        targets_csv_path=_safe_rel(targets_csv, store.root),
        processing_csv_path=_safe_rel(processing_csv, store.root),
        spatial_csv_path=_safe_rel(spatial_csv, store.root),
        pdf_path=_safe_rel(pdf_report, store.root),
        file_count=len(files) + 2,  # include manifest and latest pointer written below
        generated_at=generated_at,
    )
    manifest_payload = {
        "schema": REPORT_PACKAGE_SCHEMA,
        "generated_at": generated_at,
        "project_id": manifest.project_id,
        "result": result.to_dict(),
        "summary": summary,
    }
    store.write_json(report_manifest, manifest_payload)
    latest_manifest = store.root / "reports" / "latest_report_manifest.json"
    store.write_json(latest_manifest, manifest_payload)
    manifest.reports = {
        "status": "已生成",
        "schema": REPORT_PACKAGE_SCHEMA,
        "generated_at": generated_at,
        "latest_package_dir": result.package_dir,
        "latest_manifest_path": result.manifest_path,
        "latest_html_path": result.html_path,
        "latest_pdf_path": result.pdf_path,
        "file_count": result.file_count,
    }
    store.save_manifest()
    store.append_log(f"生成成果报告包：{result.package_dir}")
    return result


__all__ = ["REPORT_PACKAGE_SCHEMA", "ReportPackageResult", "generate_project_report_package"]
