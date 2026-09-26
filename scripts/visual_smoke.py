# -*- coding: utf-8 -*-
"""Windows 前端视觉冒烟（visual smoke）——CI 截图验证。

在真平台渲染处理页 v2 的关键场景并输出 PNG 到 ``output/visual_smoke/``，
供 CI 以 workflow artifact 形式上传、人工目检（本项目约定：视觉验收
优于程序化指标）。**不依赖真实数据**（合成剖面，种子固定，输出确定）。

用法（gui-windows job 在 pytest 通过后执行）::

    python scripts/visual_smoke.py --out output/visual_smoke

退出码：0 = 全部场景截图成功；2 = 任一场景失败（CI 可见）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

# Windows runner 有桌面会话用真平台；Linux 无显示则退回 offscreen
os.environ.setdefault('QT_QPA_PLATFORM', 'windows')

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from ui.pages.processing_page import ProcessingPage  # noqa: E402
from ui.theme_helpers import apply_theme  # noqa: E402

N_SAMPLES, N_TRACES = 700, 900


_APP: QApplication | None = None


def _make_app() -> QApplication:
    """进程内唯一 QApplication——必须模块级持有，否则被 GC 后
    qconfig 的 C++ 对象随之删除（themeChangedFinished 崩溃）。"""
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _profile(shift: float) -> np.ndarray:
    """合成剖面：直达波 + 双曲线 + 噪声（种子确定，输出可复现）。"""
    axis = np.linspace(0.0, 120.0, N_SAMPLES)
    rng = np.random.default_rng(int(shift) or 1)
    direct = np.exp(-axis / 12.0)[:, None] * np.ones((1, N_TRACES))
    hyper = np.zeros((N_SAMPLES, N_TRACES))
    for t0, amp in ((260, 0.16), (430, 0.08), (560, 0.04)):
        for trace in range(N_TRACES):
            off = (trace - N_TRACES / 2) * 1.1
            row = int(np.sqrt(max(off, 0.0) ** 2 + (t0 + shift) ** 2))
            if row < N_SAMPLES:
                hyper[row, trace] += amp
    noise = rng.normal(0, 0.010, (N_SAMPLES, N_TRACES))
    return (direct * 0.7 + hyper + noise).astype('float32')


def _bundle(tag: float, title: str):
    matrix = _profile(tag)
    return SimpleNamespace(
        matrix=matrix,
        vmin=float(np.nanpercentile(matrix, 2)),
        vmax=float(np.nanpercentile(matrix, 98)),
        title=title, x_label='道数', y_label='双程走时 (ns)',
        trace_axis_m=np.arange(N_TRACES, dtype=float) * 0.05,
        sample_axis=np.linspace(0.0, 120.0, N_SAMPLES),
        sample_axis_label='双程走时 (ns)',
        trace_count=N_TRACES, sample_count=N_SAMPLES,
        trace_elevation_m=None, depth_axis_m=None)


def _artifact(aid: str, method: str, group: str, step: int, *,
              kind: str = 'intermediate'):
    return SimpleNamespace(
        artifact_id=aid, line_id='L01',
        name=(f'run 步骤{step}_{method}' if kind == 'intermediate'
              else f'run_{method}'),
        method_id=method, method_name=method,
        created_at=f'2026-09-26T10:0{step}:00',
        manifest={'params': {'artifact_kind': kind, 'run_group_id': group,
                             'run_step_index': step}})


def _build_page() -> ProcessingPage:
    """构造处理页并喂四步链 + 运行后成果（含一步禁用）。"""
    apply_theme('dark')
    page = ProcessingPage()
    page.resize(1560, 980)
    page.show()
    app = _make_app()
    app.processEvents()
    page.set_original_bundle(_bundle(0.0, 'L01 原始数据'))
    # 链条 chips（与运行成果一致的步骤序列）
    for mid, label in (('dewow', '零时校正 (Dewow)'),
                       ('sec_gain', 'SEC 增益 (AGC)'),
                       ('bandpass', '带通滤波 (Bandpass)'),
                       ('agc', '自动增益控制 (AGC)')):
        page._pipeline_list.add_step(mid, label, {})
    page.set_artifacts([
        _artifact('S1', 'dewow', 'G1', 1),
        _artifact('S2', 'sec_gain', 'G1', 2),
        _artifact('S3', 'bandpass', 'G1', 3),
        _artifact('F1', 'agc', 'G1', 4, kind='processing'),
    ])
    for i, aid in enumerate(('S1', 'S2', 'S3', 'F1')):
        page.set_artifact_bundle(aid, _bundle(30.0 + 20 * i, aid))
    app.processEvents()
    return page


def _capture(page: ProcessingPage, out: Path, name: str) -> Path:
    app = _make_app()
    app.processEvents()
    app.processEvents()
    path = out / f'{name}.png'
    page.grab().save(str(path), 'PNG')
    return path


def _scenarios(out: Path) -> list[Path]:
    """渲染场景清单：v2 空态 / 四步运行后网格 / 选中同步高亮。"""
    produced = []
    app = _make_app()
    page = _build_page()

    # 场景 1：运行后六卡网格（输入 + 4 步，2 列大图）
    produced.append(_capture(page, out, '01-results-grid-after-run'))
    page.hide()

    # 场景 2：选中第 2 步 → chip 滑动胶囊 + 对应结果卡高亮
    page.show()
    page._pipeline_list.select_step(1)
    app.processEvents()
    produced.append(_capture(page, out, '02-selection-sync'))
    page.hide()

    # 场景 3：关闭全部步骤成果（删步骤 → 只剩输入 + 占位）
    page._pipeline_list._remove_step(0)
    app.processEvents()
    page.show()
    produced.append(_capture(page, out, '03-after-step-removal'))
    page.hide()
    page.close()
    return produced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='output/visual_smoke')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _make_app()
    try:
        produced = _scenarios(out)
    except Exception as exc:  # noqa: BLE001 - CI 需要明确失败原因
        import traceback
        traceback.print_exc()
        print(f'[visual-smoke] FAILED: {exc!r}')
        return 2
    for path in produced:
        print(f'[visual-smoke] {path}')
    missing = [p for p in produced if not p.exists() or p.stat().st_size == 0]
    if missing:
        print(f'[visual-smoke] FAILED: missing {missing}')
        return 2
    print(f'[visual-smoke] OK: {len(produced)} screenshots')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
