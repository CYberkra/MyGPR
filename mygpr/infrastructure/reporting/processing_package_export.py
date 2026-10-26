"""Stream raw input and saved results into a verified, atomically published ZIP."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import zipfile

import h5py
import numpy as np

from mygpr.application.jobs.context import ExecutionContext
from mygpr.domain.project.result_package import (
    PACKAGE_SCHEMA, RESULT_H5_SCHEMA, ProcessingResultPackage, direct_parent,
)


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path, value):
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _write_h5(path, dataset, artifact, context):
    rows, cols = dataset.matrix.shape
    label = f'导出矩阵：{artifact.name}' if artifact is not None else f'导出原始测线：{dataset.line_id}'
    with h5py.File(path, 'x') as handle:
        handle.attrs['schema'] = RESULT_H5_SCHEMA
        handle.attrs['line_id'] = dataset.line_id
        handle.attrs['artifact_id'] = artifact.artifact_id if artifact is not None else ''
        handle.attrs['dataset_role'] = 'processing_result' if artifact is not None else 'raw_input'
        handle.attrs['time_window_ns'] = dataset.time_window_ns
        handle.attrs['dielectric_constant'] = dataset.dielectric_constant
        handle.attrs['metadata_json'] = json.dumps(_json_value(dataset.to_metadata()), ensure_ascii=False, allow_nan=False)
        matrix = handle.create_dataset('matrix', shape=(rows, cols), dtype='float32',
                                       chunks=(min(rows, 256), min(cols, 1024)),
                                       compression='gzip', compression_opts=1, fletcher32=True)
        total = ((rows + 255) // 256) * ((cols + 1023) // 1024)
        completed = 0
        for row in range(0, rows, 256):
            for col in range(0, cols, 1024):
                context.raise_if_cancelled()
                region = np.s_[row:min(row + 256, rows), col:min(col + 1024, cols)]
                matrix[region] = np.asarray(dataset.matrix[region], dtype=np.float32)
                completed += 1
                context.report_progress(completed, total, label)
        for key in ('distance_axis_m', 'time_axis_ns', 'depth_axis_m'):
            values = np.asarray(getattr(dataset, key))
            expected = cols if key == 'distance_axis_m' else rows
            if values.ndim != 1 or len(values) != expected or not np.all(np.isfinite(values)):
                raise ValueError(f'{label}：无效坐标轴 {key}')
            handle.create_dataset(key, data=values)


def _axis_edges(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.array([values[0] - .5, values[0] + .5])
    mids = (values[1:] + values[:-1]) / 2
    return np.r_[values[0] - (mids[0] - values[0]), mids, values[-1] + (values[-1] - mids[-1])]


def _monotone(values):
    diffs = np.diff(values)
    return len(values) > 1 and (np.all(diffs > 0) or np.all(diffs < 0))


def _write_preview(path, dataset, artifact, context):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    context.raise_if_cancelled()
    data, samples, traces = dataset.preview_window(max_samples=800, max_traces=1200)
    figure = Figure(figsize=(9, 5), dpi=120)
    canvas = FigureCanvasAgg(figure)
    ax = figure.subplots()
    finite = data[np.isfinite(data)]
    limit = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    limit = max(limit, np.finfo(np.float32).tiny)
    time = np.asarray(dataset.time_axis_ns)[samples]
    # Distance may be a placeholder; label it as trace index unless explicitly known.
    status = dataset.metadata.get('distance_axis_status', 'unknown')
    distance = np.asarray(dataset.distance_axis_m)[traces]
    known = status in ('known', 'measured', 'valid', 'physical', 'provided_length') and _monotone(distance)
    timed = dataset.time_window_ns > 0 and _monotone(time)
    x_edges = _axis_edges(distance if known else traces)
    y_edges = _axis_edges(time if timed else samples)
    image = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(data), shading='flat',
                          cmap='seismic', vmin=-limit, vmax=limit, rasterized=True)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[-1], y_edges[0])
    ax.set_xlabel('Distance (m)' if known else 'Trace index (0-based)')
    ax.set_ylabel('Time (ns)' if timed else 'Sample index (0-based)')
    ax.set_title(f'{artifact.line_id} / {artifact.artifact_id}')
    figure.colorbar(image, ax=ax, label='Amplitude')
    figure.tight_layout()
    context.raise_if_cancelled()
    canvas.print_png(str(path))
    figure.clear()


def _sha256(path, context):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            context.raise_if_cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def _create_verified_zip(stage, files, archive_path, context):
    checksums = {p.relative_to(stage).as_posix(): _sha256(p, context) for p in files}
    _write_json(stage / 'checksums.json', checksums)
    files = [*files, stage / 'checksums.json']
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for index, path in enumerate(files):
            context.raise_if_cancelled()
            with path.open('rb') as source, archive.open(path.relative_to(stage).as_posix(), 'w', force_zip64=True) as dest:
                while chunk := source.read(1024 * 1024):
                    context.raise_if_cancelled()
                    dest.write(chunk)
            context.report_progress(index + 1, len(files) * 2, '压缩处理结果')
    with archive_path.open('rb+') as handle:
        os.fsync(handle.fileno())
    with zipfile.ZipFile(archive_path, 'r') as archive:
        for index, (name, expected) in enumerate(checksums.items()):
            digest = hashlib.sha256()
            with archive.open(name) as source:
                while chunk := source.read(1024 * 1024):
                    context.raise_if_cancelled()
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise OSError(f'压缩包校验失败：{name}')
            context.report_progress(len(files) + index + 1, len(files) * 2, '校验压缩包')
        if json.loads(archive.read('checksums.json')) != checksums:
            raise OSError('压缩包校验清单不一致')


def _publish(archive, destination, context):
    """Atomic no-clobber publication, including concurrent exports to one name."""
    for index in range(10000):
        context.raise_if_cancelled()
        candidate = destination if not index else destination.with_name(f'{destination.stem}_{index}{destination.suffix}')
        try:
            if os.name == 'nt':
                os.rename(archive, candidate)
            else:
                os.link(archive, candidate)
        except FileExistsError:
            logging.getLogger(__name__).debug('Keeping existing result package: %s', candidate)
            continue
        context.mark_committed()
        return candidate
    raise FileExistsError('同名打包文件过多，请选择其他文件名。')


def export_processing_package(*, project_id, line_id, artifacts, load_dataset, load_raw_dataset,
                              destination, target, warnings, context: ExecutionContext):
    destination = Path(destination).expanduser().absolute()
    if destination.suffix.lower() != '.zip':
        destination = destination.with_name(destination.name + '.zip')
    if not destination.parent.is_dir():
        raise ValueError('保存目录不存在，请重新选择保存位置。')
    context.raise_if_cancelled()
    manifest = dict(schema=PACKAGE_SCHEMA, project_id=project_id, line_id=line_id,
                    scope='to_step' if target else 'line', target_artifact_id=target,
            created_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"), warnings=list(warnings), steps=[],
                    sensor_metadata='No persisted per-step sensor snapshots; source sensors are not substituted.')
    with tempfile.TemporaryDirectory(prefix='.mygpr-package-', dir=destination.parent) as folder:
        stage = Path(folder)
        raw = stage / 'raw'
        raw.mkdir()
        try:
            raw_dataset = load_raw_dataset()
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f'无法读取原始测线 {line_id}，请修复数据后重新打包：{exc}') from exc
        _write_h5(raw / 'data.h5', raw_dataset, None, context.child(0, len(artifacts) + 2))
        _write_json(raw / 'source.json', dict(
            kind='project_raw_snapshot', line_id=line_id, metadata=raw_dataset.to_metadata()))
        manifest['raw_input'] = dict(
            kind='project_raw_snapshot', line_id=line_id, shape=list(raw_dataset.matrix.shape),
            data='raw/data.h5', source='raw/source.json')
        for index, artifact in enumerate(artifacts):
            context.raise_if_cancelled()
            # Internal paths use ordinal IDs, never user-controlled file names.
            step = stage / 'steps' / f'{index + 1:03d}'
            step.mkdir(parents=True)
            dataset = load_dataset(artifact.artifact_id)
            step_context = context.child(index + 1, len(artifacts) + 2)
            _write_h5(step / 'data.h5', dataset, artifact, step_context)
            _write_preview(step / 'bscan.png', dataset, artifact, step_context)
            _write_json(step / 'parameters.json', artifact.manifest)
            relative = step.relative_to(stage).as_posix()
            manifest['steps'].append(dict(
                artifact_id=artifact.artifact_id, name=artifact.name, method_id=artifact.method_id,
                branch_id=artifact.branch_id, direct_parent_artifact_id=direct_parent(artifact),
                shape=list(dataset.matrix.shape), source_record_sha256=artifact.sha256,
                data=f'{relative}/data.h5', preview=f'{relative}/bscan.png', parameters=f'{relative}/parameters.json'))
        _write_json(stage / 'manifest.json', manifest)
        (stage / 'README.txt').write_text(
            'MyGPR 处理结果包\nraw/data.h5 是打包时工程中未处理的测线输入，每包只保存一份。\n'
            '原始矩阵及坐标轴以可重新导入的 H5 保存；raw/source.json 记录来源元数据。\n'
            '这份输入快照不是原始采集文件的逐字节副本，也不证明它与所有历史成果的输入相同。\n'
            '每个 steps 目录含实际保存的矩阵 H5、B-scan PNG 和参数 JSON。\n'
            'manifest.json 记录分支与可靠的直接来源；未知来源保留为 null。\n'
            '仅包含已保存结果；未保存的中间步骤不会重新计算。H5 可在新版 MyGPR 中导入。\n'
            'PNG 为降采样预览，色标按该结果幅度的 99% 分位显示；完整数值以 H5 为准。\n'
            '未保存逐步传感器快照，因此不将原始传感器数据当作处理后的快照。\n'
            '本包是结果交付包，不替代整个工程备份。\n' + '\n'.join(warnings), encoding='utf-8')
        files = sorted(p for p in stage.rglob('*') if p.is_file())
        archive = stage / 'result.zip'
        _create_verified_zip(stage, files, archive, context.child(len(artifacts) + 1, len(artifacts) + 2))
        final = _publish(archive, destination, context)
    return ProcessingResultPackage(str(final), line_id, tuple(a.artifact_id for a in artifacts), tuple(warnings), True)
