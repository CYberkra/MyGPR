#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HTML, PDF, XLSX, figure and checksum renderers for project reports."""
from __future__ import annotations

import html
import logging
import uuid
from pathlib import Path
from typing import Any

from core.plot_font_policy import configure_matplotlib_cjk_fonts
from core.report_export_rows import _sha256_file

_LOGGER = logging.getLogger(__name__)
from core.tabular_security import safe_tabular_value


def _write_xlsx_report(
    path: Path,
    *,
    summary: dict[str, Any],
    sheets: list[tuple[str, list[dict[str, Any]]]],
) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
        from openpyxl.utils import get_column_letter
    except Exception as exc:
        raise RuntimeError("生成 Excel 工程汇总需要 openpyxl。") from exc
    wb = Workbook()
    default = wb.active
    wb.remove(default)
    project = summary.get("project", {})
    overview = wb.create_sheet("项目概况")
    overview.append(["MyGPR 工程成果汇总", ""])
    overview.merge_cells("A1:B1")
    overview["A1"].font = Font(bold=True, size=16)
    overview["A1"].alignment = Alignment(horizontal="center")
    for key, label in [
        ("name", "项目名称"), ("project_no", "项目编号"), ("location", "测区位置"),
        ("operator", "编制/操作员"), ("reviewer", "审核人"), ("approver", "批准人"),
        ("device_model", "设备型号"), ("coordinate_system", "坐标系统"),
        ("vertical_datum", "高程基准"), ("generated_at", "生成时间"),
    ]:
        value = summary.get("generated_at", "") if key == "generated_at" else project.get(key, "")
        if key in ('reviewer', 'approver') and not value:
            continue
        overview.append([label, safe_tabular_value(value)])
    overview.column_dimensions["A"].width = 22
    overview.column_dimensions["B"].width = 72
    overview.freeze_panes = "A2"

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9E2F3")
    for name, rows in sheets:
        ws = wb.create_sheet(name[:31])
        headers = list(rows[0].keys()) if rows else ["说明"]
        ws.append(headers)
        if not rows:
            ws.append(["无记录"])
        else:
            for row in rows:
                ws.append([safe_tabular_value(row.get(header, "")) for header in headers])
        for cell in ws[1]:
            cell.fill = header_fill; cell.font = header_font; cell.alignment = Alignment(horizontal="center", vertical="center")
        for row in ws.iter_rows():
            for cell in row:
                cell.border = Border(bottom=thin)
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for idx, header in enumerate(headers, start=1):
            values = [str(header)] + [str(row.get(header, "")) for row in rows[:200]]
            ws.column_dimensions[get_column_letter(idx)].width = min(max(max(len(v) for v in values) + 2, 10), 42)
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        ws.sheet_view.showGridLines = False
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth = 1
        ws.sheet_properties.pageSetUpPr.fitToPage = True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.xlsx")
    wb.save(tmp)
    tmp.replace(path)

def _write_checksums(package_dir: Path, path: Path, *, cancel_checker=None) -> list[dict[str, Any]]:
    rows=[]
    for file in sorted(p for p in package_dir.rglob("*") if p.is_file() and p != path):
        rows.append({
            "path": file.relative_to(package_dir).as_posix(),
            "size_bytes": file.stat().st_size,
            "sha256": _sha256_file(file, cancel_checker=cancel_checker),
        })
    text="".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    path.write_text(text,encoding="utf-8")
    return rows

_BOREHOLE_COLUMNS = ['borehole_id', 'line_id', 'trace_index', 'borehole_depth_m', 'interpreted_depth_m', 'error_m', 'status']
_REPORT_SIGNERS = (('编制', 'compiler'), ('审核', 'reviewer'), ('批准', 'approver'))
_REPORT_HEADERS = {
    'line_id': '测线号', 'name': '名称', 'length_m': '长度（m）', 'data_quality': '数据质检',
    'rtk_status': 'RTK 状态', 'processing_status': '处理状态', 'status': '状态',
    'sample_count': '采样点数', 'trace_count': '道数', 'orientation_message': '道序说明',
    'rtk_coverage_ratio': 'RTK 覆盖率', 'rtk_fixed_ratio': 'RTK 固定解占比',
    'imu_coverage_ratio': 'IMU 覆盖率', 'gap_count': '缺口数', 'warning_count': '警告数',
    'artifact_id': '成果编号', 'method_name': '处理方法', 'input_shape': '输入尺寸（采样点 × 道数）',
    'output_shape': '输出尺寸（采样点 × 道数）', 'version': '版本', 'keypoint_count': '标注点数',
    'coverage_ratio': '覆盖率', 'spatial_csv_path': '空间成果文件', 'row_count': '记录数',
    'has_xy_count': '有坐标的记录数', 'kind': '类型', 'role': '用途', 'crs': '坐标参考',
    'geometry_type': '几何类型', 'borehole_id': '钻孔编号', 'trace_index': '道索引',
    'borehole_depth_m': '钻孔深度（m）', 'interpreted_depth_m': '解释深度（m）',
    'error_m': '误差（m）', 'target_id': '目标编号', 'distance_m': '沿线距离（m）',
    'depth_m': '深度（m）', 'type': '类型',
}
_LINE_PARAMETERS = ('设备', '天线', '频段／中心频率', '功率', '时窗', '采样点／道数')


def _report_sections(lines, quality, interfaces, targets, artifacts, spatial,
                     sensor_sync, gis_layers, boreholes, marker_rows):
    return (
        ("测线台账", ["line_id", "name", "length_m", "data_quality", "rtk_status", "processing_status"], lines),
        ("数据质检", ["line_id", "status", "sample_count", "trace_count", "orientation_message"], quality),
        ("传感器同步", ["line_id", "status", "rtk_coverage_ratio", "rtk_fixed_ratio", "imu_coverage_ratio", "gap_count", "warning_count"], sensor_sync),
        ("处理记录", ["artifact_id", "line_id", "method_name", "status", "input_shape", "output_shape"], artifacts),
        ("界面解释", ["line_id", "status", "version", "keypoint_count", "coverage_ratio"], interfaces),
        ("空间成果", ["line_id", "spatial_csv_path", "row_count", "has_xy_count"], spatial),
        ("GIS 图层", ["name", "kind", "role", "crs", "geometry_type"], gis_layers),
        ('钻孔对比', _BOREHOLE_COLUMNS, boreholes),
        ("历史点目标", ["target_id", "line_id", "distance_m", "depth_m", "type", "status"], targets),
        ("标记与事件区间", ["测线", "图件编号", "名称", "用途", "道号范围", "数据版本", "信息来源", "状态"], marker_rows),
        ("事件采集时段", ["图件编号", "开始采集时间（UTC）", "结束采集时间（UTC）"],
         [row for row in marker_rows if row.get("开始采集时间（UTC）") or row.get("结束采集时间（UTC）")]),
    )


def _step_text(dossier):
    result = []
    for run in dossier['processing']:
        for step in run['steps']:
            if not step.get('enabled', True):
                continue
            params = step.get('params') or {}
            name = step.get('label') or step.get('method_id') or '处理'
            compact = {key: value for key, value in params.items()
                       if isinstance(value, (str, int, float, bool)) and not key.startswith('_')}
            values = ' · '.join(f'{key}={value}' for key, value in compact.items())
            result.append(f'{name} · {values}' if values else name)
        selection = run.get('selection')
        if selection:
            mode = '保留' if selection['mode'] == 'keep' else '剔除'
            ranges = ', '.join(f'{a + 1}–{b}' for a, b in selection.get('ranges', ())[:8])
            more = ' 等' if len(selection.get('ranges', ())) > 8 else ''
            source = selection.get('input_artifact_id') or '原始数据'
            result.append(f'{mode} · 输入 {source} · {ranges}{more} 道 · 选中 {selection["affected_count"]} 道 · 输出 {selection["output_count"]} 道')
            for event in selection.get('intervals', ()):
                label = ' · '.join(filter(None, (event.get('report_code'), event.get('name') or event['uid'])))
                result.append(f'{label} · 区间修订 {event.get("revision", "—")}')
        result.extend(str(warning) for warning in run['warnings'])
    return result


def _event_text(event):
    parts = [event['position'], event['purpose'], event['evidence']]
    if event.get('time_start') or event.get('time_end'):
        parts.append(f'{event["time_start"] or "开放"} 至 {event["time_end"] or "开放"}（UTC）')
    if event.get('note'):
        parts.append(event['note'])
    return ' · '.join(part for part in parts if part)


def _ordered_dossiers(summary):
    dossiers = summary.get('line_dossiers', [])
    featured = [item for item in dossiers if item['featured']]
    if not featured and dossiers:
        featured = sorted(dossiers, key=lambda item: (
            not any(item['analysis'].values()),
            item['quality'].get('status') not in ('失败', '警告'),
        ))[:min(len(dossiers), 3)]
    identifiers = {item['line_id'] for item in featured}
    return featured, [item for item in dossiers if item['line_id'] not in identifiers]


def _write_html_report(path: Path, *, summary, lines, quality, interfaces, targets, artifacts,
                       spatial, sensor_sync=(), gis_layers=(), boreholes=()):
    from core.report_line_content import ANALYSIS_FIELDS

    def esc(value):
        return html.escape('' if value is None else str(value))

    def table(headers, rows):
        head = ''.join(f'<th>{esc(_REPORT_HEADERS.get(key, key))}</th>' for key in headers)
        body = ''.join('<tr>' + ''.join(f'<td>{esc(row.get(key))}</td>' for key in headers) + '</tr>' for row in rows)
        return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'

    def figure(name, label, *, caption=True):
        if not name:
            return ''
        text = f'<figcaption>{esc(label)}</figcaption>' if caption else ''
        return f'<figure><img alt="{esc(label)}" src="../figures/{esc(name)}">{text}</figure>'

    def paragraph(label, value):
        return f'<h3>{esc(label)}</h3><p>{esc(value)}</p>' if value else ''

    def dossier(item):
        qc = item['quality']
        analysis = item['analysis']
        uid = esc(item['line_id'])
        location = (f'{analysis["code"]} · 原始第 {analysis["first_trace"]}–{analysis["last_trace"]} 道'
                    if analysis.get('code') else '')
        descriptions = ''.join(paragraph(label, analysis.get(key)) for key, label in ANALYSIS_FIELDS.items())
        events = ''.join(f'<dt>{esc(event["code"])} · {esc(event["name"])}</dt><dd>{esc(_event_text(event))}</dd>'
                         for event in item['events'])
        steps = ''.join(f'<li>{esc(value)}</li>' for value in _step_text(item))
        notes = ''.join(f'<li>{esc(note)}</li>' for note in item['figure_notes'])
        details = ''.join(figure(name, location) for name in item['detail_figures'])
        notice = ' · '.join(filter(None, (qc.get('status'), qc.get('orientation_message')))) if (
            qc.get('status') in ('失败', '警告')) else ''
        parameters = ''.join(f'<dt>{esc(key)}</dt><dd>{esc(value)}</dd>'
                             for key, value in item['acquisition'].items() if value)
        route = figure(item.get('route_figure'), f'{item["line_id"]} 路线', caption=False)
        if not route and item['map_status'] == '无法在地图定位':
            reason = summary.get('plan_map_note') if len(summary['line_dossiers']) == 1 else ''
            route = f'<p class="meta">{esc(reason or "缺少可用的定位轨迹")}</p>'
        elif not route and item['map_status'] == '已绘制轨迹':
            route = '<p><a href="#survey-map">查看测区路线总图</a></p>'
        return (
            f'<section id="{uid}"><h3>{uid} · {esc(item["name"])}</h3>'
            + (f'<p class="meta">{esc(notice)}</p>' if notice else '')
            + f'<div class="line-header">{route}<dl class="parameters">{parameters}</dl></div>'
            + figure(item['raw_figure'], '原始 B-scan', caption=False)
            + figure(item['processed_figure'], f"处理后 B-scan · {item['primary_name']}", caption=False)
            + ('<p class="meta">仅原始剖面</p>' if item['raw_figure'] and not item['primary_artifact_id'] else '')
            + (f'<ul class="meta">{notes}</ul>' if notes else '')
            + '<h4>分析说明</h4>' + (f'<p>{esc(location)}</p>' if location else '')
            + (descriptions or '<p class="meta">未填写</p>')
            + details + (f'<p class="meta">{esc(item["detail_note"])}</p>' if item['detail_note'] else '')
            + (f'<dl>{events}</dl>' if events else '')
            + (f'<details><summary>处理步骤</summary><ol>{steps}</ol></details>' if steps else '')
            + '</section>'
        )

    project = summary['project']
    signers = '　'.join(f'{label} · {esc(project[key])}' for label, key in _REPORT_SIGNERS if project.get(key))
    narrative = summary['project_analysis']
    focus, others = _ordered_dossiers(summary)
    line_count = len(summary['line_dossiers'])
    tables = _report_sections(lines, quality, interfaces, targets, artifacts, spatial,
                              sensor_sync, gis_layers, boreholes, summary.get('marker_rows', []))
    names = ' · '.join(f'<a href="#{esc(item["line_id"])}">{esc(item["line_id"])}</a>' for item in focus + others)
    map_status = ', '.join(item['line_id'] for item in focus + others if item['map_status'] == '无法在地图定位')
    show_plan = line_count > 1 and (summary['plan_map'] or map_status)
    processing_link = (' · <a href="../json/processing_parameters.json">处理参数与版本</a>'
                       if summary['files'].get('processing_parameters_json') else '')
    css = """
    body{font:15px/1.55 system-ui,'Microsoft YaHei',sans-serif;margin:40px auto;padding:0 20px;max-width:1100px;color:#172336}
    h1,h2,h3,h4{line-height:1.22} h2{margin-top:36px;border-bottom:1px solid #cbd5e1;padding-bottom:8px}
    p,dd{white-space:pre-wrap;overflow-wrap:anywhere} .meta,figcaption{color:#475569;font-size:13px}
    figure{margin:18px 0} img{width:100%;object-fit:contain} figcaption{margin-top:3px}
    table{border-collapse:collapse;width:100%;font-size:12px} th,td{border-bottom:1px solid #cbd5e1;padding:6px;text-align:left}
    th{background:#f1f5f9} .scroll{overflow:auto} section{margin:25px 0} dt{font-weight:600}
    .line-header{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;border-bottom:1px solid #cbd5e1}
    .line-header figure{flex:1 1 280px;max-width:520px;margin:0}
    .parameters{flex:1 1 260px;display:grid;grid-template-columns:max-content 1fr;gap:3px 12px;font-size:13px;margin:0 0 12px}
    .parameters dd{margin:0}
    """
    output = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{esc(project['name'])}</title><style>{css}</style></head><body>
<h1>{esc(project['name'])} 成果报告</h1>
<p class="meta">{esc(project.get('project_no'))} · {esc(project.get('location'))} · {esc(summary.get('generated_at'))}</p>
<h2>项目摘要</h2>
{paragraph('测量目的', narrative.get('objective'))}
{paragraph('主要发现', narrative.get('findings') or '未填写项目发现')}
<p>{line_count} 条测线 · {names}</p>
{'<h2 id="survey-map">测区路线总图</h2>' if show_plan else ''}
{figure(summary['plan_map'], '测区路线总图', caption=False) if show_plan else ''}
{f'<p class="meta">无法在地图定位 · {esc(map_status)}</p>' if show_plan and map_status else ''}
{f'<p class="meta">{esc(summary["plan_map_note"])}</p>' if show_plan and summary.get('plan_map_note') else ''}
<h2>重点测线</h2>{''.join(dossier(item) for item in focus)}
<h2>结论与验证</h2>
<p>{esc(narrative.get('conclusion') or '未填写项目结论')}</p>
{table(_BOREHOLE_COLUMNS, boreholes) if boreholes else ''}
{'<h2>测线图册</h2>' if others else ''}{''.join(dossier(item) for item in others)}
<h2>交付附件</h2>
<p><a href="../tables/acquisition_parameters.csv">采集参数 CSV</a> ·
<a href="../json/line_dossiers.json">逐线记录</a>{processing_link} ·
<a href="../json/markers.csv">标记清单</a> · <a href="../checksums.sha256">文件校验</a></p>
{''.join(f'<details><summary>{title}</summary><div class="scroll">{table(cols, rows)}</div></details>' for title, cols, rows in tables if rows)}
<h2>报告版本</h2>
{f'<p>{signers}</p>' if signers else ''}
<p class="meta">报告 {esc(summary.get('report_revision'))} · 快照 {esc(summary.get('snapshot', {}).get('snapshot_id'))}</p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(output, encoding="utf-8")
    tmp.replace(path)


def _write_pdf_report(path: Path, *, summary: dict[str, Any], boreholes: list[dict[str, Any]],
                      figure_paths: list[Path]) -> None:
    """Write the paginated engineering PDF with cover, approvals and appendices.

    CSV/JSON/XLSX remain the normalized evidence tables and the checksum manifest
    provides independent package-integrity verification.
    """
    import matplotlib
    matplotlib.use("Agg")   # 同上：无头交付红线
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

    configured_font = configure_matplotlib_cjk_fonts()
    font_name = configured_font if configured_font != "DejaVu Sans" else _choose_font()
    page_number = 0

    def _page(title: str):
        nonlocal page_number
        page_number += 1
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        fig.text(0.06, 0.955, title, fontsize=18, fontweight="bold", fontname=font_name)
        fig.text(0.50, 0.025, f"第 {page_number} 页", fontsize=8, color="#64748B", ha="center", fontname=font_name)
        fig.text(0.94, 0.025, str(summary.get("report_revision", "R1")), fontsize=8, color="#64748B", ha="right", fontname=font_name)
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
        table = ax.table(cellText=data, colLabels=[_REPORT_HEADERS.get(key, key) for key in headers],
                         loc="upper left", cellLoc="left")
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
    from core.report_line_content import ANALYSIS_FIELDS

    figures = {item.name: item for item in figure_paths if item.exists()}
    narrative = summary['project_analysis']
    focus, others = _ordered_dossiers(summary)

    def wrap(value, width=73):
        import unicodedata

        rows = []
        for paragraph in str(value or '').splitlines():
            line, used = [], 0
            for char in paragraph:
                size = 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
                if used + size > width:
                    rows.append(''.join(line))
                    line, used = [], 0
                line.append(char)
                used += size
            rows.append(''.join(line))
        return rows

    def write_page(pdf, fig):
        pdf.savefig(fig)
        plt.close(fig)

    def image(fig, name, rect, *, anchor='C'):
        if name in figures:
            ax = fig.add_axes(rect)
            ax.imshow(plt.imread(figures[name]))
            ax.axis('off')
            ax.set_anchor(anchor)
            return ax.get_position().y0
        return rect[1] + rect[3]

    def write_rows(fig, rows, start_y):
        for row, (label, value) in enumerate(rows):
            fig.text(.07, start_y - row * .022, value, fontname=font_name,
                     fontsize=9.5, fontweight='bold' if label else 'normal')

    def write_text_pages(pdf, title, text_rows):
        for offset in range(0, len(text_rows), 38):
            fig = _page(title if offset == 0 else f'{title}（续）')
            write_rows(fig, text_rows[offset:offset + 38], .886)
            write_page(pdf, fig)

    def text_block(rows, label, value):
        if value:
            rows.append((True, label))
            rows.extend((False, row) for row in wrap(value, 88))
            rows.append((False, ''))

    def write_line(pdf, item):
        title = f"{item['line_id']} · {item['name']}"
        fig = _page(title[:44])
        qc = item['quality']
        route_name = item.get('route_figure')
        if route_name:
            image(fig, route_name, [.05, .735, .43, .185], anchor='N')
        elif item['map_status'] != '未选空间章节':
            route_status = '见测区路线总图'
            if item['map_status'] == '无法在地图定位':
                reason = summary.get('plan_map_note') if len(summary['line_dossiers']) == 1 else ''
                route_status = reason or '缺少可用的定位轨迹'
            for row, line in enumerate(wrap(route_status, 38)):
                fig.text(.07, .875 - .019 * row, line, fontname=font_name, fontsize=9, color='#475569')
        parameter_x = .52 if route_name or item['map_status'] != '未选空间章节' else .07
        for row, key in enumerate(_LINE_PARAMETERS):
            value = item['acquisition'].get(key) or '未记录'
            line = f'{key}  {value}'
            short = wrap(line, 48)
            fig.text(parameter_x, .903 - .023 * row, short[0] + ('…' if len(short) > 1 else ''),
                     fontname=font_name, fontsize=9, color='#334155')
        image(fig, item['raw_figure'], [.05, .495, .90, .232])
        if item['processed_figure']:
            image(fig, item['processed_figure'], [.05, .252, .90, .232])
        elif item['raw_figure']:
            message = '所选处理剖面不可用' if item['primary_artifact_id'] else '仅原始剖面'
            fig.text(.07, .485, message, fontname=font_name, fontsize=9)
        baseline = .245 if item['processed_figure'] else .454 if item['raw_figure'] else .723
        notes = list(item['figure_notes'])
        if qc.get('status') in ('失败', '警告'):
            notes.insert(0, qc.get('orientation_message') or qc['status'])
        visible_notes = [line for note in notes for line in wrap(note, 93)]
        for row, line in enumerate(visible_notes[:3]):
            fig.text(.07, baseline - .019 * row, line, fontname=font_name, fontsize=8, color='#475569')
        content = []
        analysis = item['analysis']
        if analysis.get('code'):
            text_block(content, '分析范围',
                       f"{analysis['code']} · 原始第 {analysis['first_trace']}–{analysis['last_trace']} 道")
        if item['detail_note']:
            content.extend((False, row) for row in wrap(item['detail_note']))
        if any(analysis.get(key) for key in ANALYSIS_FIELDS):
            for key, label in ANALYSIS_FIELDS.items():
                text_block(content, label, analysis.get(key))
        else:
            text_block(content, '分析说明', '未填写')
        content.extend((False, line) for line in visible_notes[3:])
        steps = _step_text(item)
        if steps:
            text_block(content, '处理步骤与范围', '\n'.join(steps))
        for event in item['events']:
            text_block(content, f"{event['code']} · {event['name']}", _event_text(event))
        more_parameters = {key: value for key, value in item['acquisition'].items()
                           if value and value != '未记录'
                           and (key not in _LINE_PARAMETERS or len(wrap(f'{key}  {value}', 48)) > 1)}
        if more_parameters:
            text_block(content, '采集参数', '\n'.join(f'{key}  {value}' for key, value in more_parameters.items()))
        next_y = baseline - .021 * len(visible_notes[:3]) - .026
        available = max(0, int((next_y - .055) / .020))
        if len(content) > available:
            available = max(0, available - 2)
        for row, (label, value) in enumerate(content[:available]):
            fig.text(.07, next_y - .020 * row, value, fontname=font_name, fontsize=9,
                     fontweight='bold' if label else 'normal')
        remaining = content[available:]
        if remaining:
            fig.text(.07, .057, '见续页', fontname=font_name, fontsize=8, color='#475569')
        write_page(pdf, fig)
        write_text_pages(pdf, f"{item['line_id']} · 处理与分析", remaining)
        if item['detail_figures']:
            fig = _page(f"{analysis['code']} · 局部剖面")
            fig.text(.07, .898, f"原始第 {analysis['first_trace']}–{analysis['last_trace']} 道",
                     fontname=font_name, fontsize=10)
            for index, name in enumerate(item['detail_figures']):
                image(fig, name, [.05, .57 - .30 * index, .90, .29])
            if item['detail_note']:
                fig.text(.07, .53, item['detail_note'], fontname=font_name, fontsize=9)
            write_page(pdf, fig)

    with PdfPages(tmp) as pdf:
        fig = _page(f"{project.get('name', 'MyGPR 项目')} 成果报告"[:47])
        _kv_lines(fig, .887, [
            ('项目编号', project.get('project_no') or '—'),
            ('测区位置', str(project.get('location') or '—')[:55]),
            ('交付范围', f"{metrics.get('line_count', 0)} 条测线 · "
             f"{len(summary.get('report_profile', {}).get('selected_artifact_ids', ()))} 项处理成果"),
            ('生成时间', summary.get('generated_at', '')),
        ])
        overview = []
        text_block(overview, '测量目的', narrative.get('objective'))
        text_block(overview, '主要发现', narrative.get('findings') or '未填写项目发现')
        text_block(overview, '重点测线', ' · '.join(
            f"{item['line_id']} {item['name']}" for item in focus))
        warnings = [f"{item['line_id']} {item['quality'].get('orientation_message') or item['quality'].get('status')}"
                    for item in focus + others if item['quality'].get('status') in ('失败', '警告')]
        if warnings:
            text_block(overview, '质检提示', '\n'.join(warnings))
        for row, (label, value) in enumerate(overview[:26]):
            fig.text(.075, .727 - row * .023, value, fontname=font_name, fontsize=10,
                     fontweight='bold' if label else 'normal')
        write_page(pdf, fig)
        write_text_pages(pdf, '项目摘要（续）', overview[26:])

        dossiers = focus + others
        missing = [item['line_id'] for item in dossiers if item['map_status'] == '无法在地图定位']
        if (summary['plan_map'] and len(dossiers) > 1) or (missing and len(dossiers) > 1):
            fig = _page('测区路线总图')
            y = image(fig, summary['plan_map'], [.05, .30, .90, .59], anchor='N') - .027
            map_rows = wrap('无法在地图定位 · ' + '、'.join(missing), 92) if missing else []
            map_rows.extend(wrap(summary.get('plan_map_note'), 92))
            write_rows(fig, [(False, row) for row in map_rows[:5]], y)
            write_page(pdf, fig)
            write_text_pages(pdf, '地图定位状态（续）', [(False, row) for row in map_rows[5:]])

        for item in focus:
            write_line(pdf, item)
        conclusions = []
        text_block(conclusions, '结论', narrative.get('conclusion') or '未填写项目结论')
        if boreholes:
            count = sum(row.get('status') == 'passed' for row in boreholes)
            text_block(conclusions, '钻孔对比', f'{count} / {len(boreholes)} 项误差达标，逐项数值见钻孔对比表。')
        write_text_pages(pdf, '结论与验证', conclusions)
        for item in others:
            write_line(pdf, item)

        fig = _page('交付附件与版本')
        attachment_rows = [
            ('采集参数', 'tables/acquisition_parameters.csv'),
            ('逐线资料', 'json/line_dossiers.json'),
            ('标记清单', 'json/markers.csv'),
            ('文件校验', 'checksums.sha256'),
            ('报告快照', (summary.get('snapshot') or {}).get('snapshot_id', '')),
        ]
        if summary['files'].get('processing_parameters_json'):
            attachment_rows.insert(2, ('处理参数', 'json/processing_parameters.json'))
        _kv_lines(fig, .878, attachment_rows)
        y = .65
        for role, key in _REPORT_SIGNERS:
            if not project.get(key):
                continue
            fig.text(.075, y, role, fontsize=10, fontweight='bold', fontname=font_name)
            fig.text(.24, y, project[key], fontsize=10, fontname=font_name)
            y -= .027
        write_page(pdf, fig)
        for offset in range(0, len(boreholes), 24):
            title = f'钻孔对比 · {offset // 24 + 1}' if len(boreholes) > 24 else '钻孔对比'
            _table_page(pdf, title, _BOREHOLE_COLUMNS, boreholes[offset:offset + 24])
    tmp.replace(path)

__all__ = ['_write_xlsx_report', '_write_checksums', '_write_html_report', '_write_pdf_report']
