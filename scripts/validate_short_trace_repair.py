"""Local field validation, saved artifact roundtrip, and known-trace holdouts.

Run from the repository: python -m scripts.validate_short_trace_repair
--source FILE --output NEW_DIRECTORY --time-windows-ns '0:100;100:200;250:850'
No uploads, in-place source writes, or changes to the user's active projects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from core.gpr_data_model import load_gpr_dataset
from core.storage_primitives import atomic_write_bytes, atomic_write_json
from mygpr.domain.processing.models import PipelineDefinition, PipelineStep, ProcessingRequest
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor
from mygpr.interfaces.backend import MyGPRBackend


def _verify_package_roundtrips(backend, pid, package, output, raw, marked, mark_data, repaired):
    roundtrips = []
    with zipfile.ZipFile(package.archive_path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        entries = [('raw', manifest['raw_input']['data'], raw)]
        for step in manifest['steps']:
            expected = mark_data if step['artifact_id'] == marked.artifact_id else repaired
            entries.append((step['artifact_id'], step['data'], expected))
        for index, (name, member, expected) in enumerate(entries):
            destination = output / f'{name}.h5'
            atomic_write_bytes(destination, archive.read(member))
            loaded = load_gpr_dataset(destination)
            np.testing.assert_array_equal(loaded.matrix, expected.data)
            np.testing.assert_array_equal(loaded.time_axis_ns, expected.header_info['time_axis_ns'])
            # Core GPRDataSet exposes preview arrays, not processing trace metadata.
            # Verify masks through the real project reimport/read contract.
            roundtrip_line = f'L{20 + index}'
            backend.projects.import_line_source(pid, roundtrip_line, destination)
            imported = backend.projects.read_dataset(pid, roundtrip_line)
            for key in ('short_trace_candidate_mask', 'short_trace_repaired_mask', 'short_trace_unresolved_mask'):
                if key in expected.trace_metadata:
                    np.testing.assert_array_equal(imported.trace_metadata[key], expected.trace_metadata[key])
            roundtrips.append({'name': name, 'path': str(destination), 'matrix_time_masks_exact': True})
    return roundtrips


def validate(source: Path, output: Path, time_windows_ns: str, baseline: Path | None = None) -> dict:
    source = source.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('验证输出目录必须不存在或为空，避免覆盖已有结果')
    output.mkdir(parents=True, exist_ok=True)
    # Only this process's disposable registry/settings are redirected.
    os.environ['LOCALAPPDATA'] = str(output / 'local_app_data')
    os.environ['MYGPR_RUNTIME_ROOT'] = str(output / 'runtime')
    os.environ['MYGPR_SETTINGS_FILE'] = str(output / 'settings.json')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    params = {'window_traces': 31, 'robust_z': 8., 'time_windows_ns': time_windows_ns, 'max_run_traces': 4}
    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(output / 'project', name='Short trace validation')
        pid = project.project_id
        backend.projects.import_line_source(pid, 'L04', source)
        raw = backend.projects.read_dataset(pid, 'L04')
        marked = backend.project_processing.execute_pipeline(pid, 'L04', PipelineDefinition(
            steps=(PipelineStep('short_trace_repair', params),)))
        mark_data = backend.projects.read_artifact_dataset(pid, 'L04', marked.artifact_id)
        np.testing.assert_array_equal(mark_data.data, raw.data)
        final = backend.project_processing.execute_pipeline(pid, 'L04', PipelineDefinition(
            steps=(PipelineStep('short_trace_repair', {**params, 'mode': 'interpolate'}),)),
            input_artifact_id=marked.artifact_id)
        repaired = backend.projects.read_artifact_dataset(pid, 'L04', final.artifact_id)
        flag = repaired.trace_metadata['short_trace_candidate_mask'].astype(bool)
        changed = repaired.trace_metadata['short_trace_repaired_mask'].astype(bool)
        np.testing.assert_array_equal(raw.data[:, ~changed], repaired.data[:, ~changed])
        np.testing.assert_array_equal(backend.projects.read_dataset(pid, 'L04').data, raw.data)
        np.testing.assert_array_equal(raw.header_info['time_axis_ns'], repaired.header_info['time_axis_ns'])
        for key in ('trace_id', 'trace_timestamp_s'):
            if key in raw.trace_metadata:
                np.testing.assert_array_equal(raw.trace_metadata[key], repaired.trace_metadata[key])
        candidates = (np.flatnonzero(flag) + 1).tolist()
        baseline_matches = None
        if baseline:
            expected = json.loads(baseline.read_text(encoding='utf-8'))
            baseline_matches = candidates == expected['flagged_trace_ids']
            assert baseline_matches, '独立研究脚本与方法库候选道不一致'
        # Hide valid measured traces; evaluate only where the actual waveform is known.
        holdouts = []
        time_axis = np.asarray(raw.header_info['time_axis_ns'])
        early = (time_axis >= 0) & (time_axis <= 100)
        for start, length in ((180, 1), (280, 2), (480, 4), (680, 1), (880, 4)):
            first = start - 1
            stop = first + length
            if stop >= raw.data.shape[1] or flag[max(0, first-16):min(len(flag), stop+16)].any():
                continue
            hidden = np.array(raw.data, copy=True)
            hidden[:, first:stop] = 0
            result = NativeProcessingExecutor().execute(ProcessingRequest(
                data=hidden, method_id='short_trace_repair', params={**params, 'mode': 'interpolate'},
                header_info=raw.header_info))
            recovered = result.trace_metadata['short_trace_repaired_mask'][first:stop].astype(bool)
            error = result.data[early, first:stop].astype(float) - raw.data[early, first:stop]
            relative = float(np.sqrt(np.mean(error**2) / np.mean(raw.data[early, first:stop].astype(float)**2)))
            holdouts.append({'start_trace': start, 'end_trace': start+length-1,
                             'all_hidden_traces_detected_and_repaired': bool(recovered.all()),
                             'relative_RMS_error_0_100ns': relative})
            assert recovered.all(), '人工遮挡的正常道未完成检测/插补'
        job = backend.submit_processing_result_package(pid, 'L04', str(output / 'original_and_derived.zip'))
        exported = backend.jobs.wait(job, timeout=60)
        if exported.status.value != 'completed':
            raise RuntimeError(exported.error_message)
        package = exported.result
        assert package.verified
        roundtrips = _verify_package_roundtrips(
            backend, pid, package, output, raw, marked, mark_data, repaired)
        after = hashlib.sha256(source.read_bytes()).hexdigest()
        assert before == after
        result = {'source': str(source), 'source_sha256_before': before, 'source_sha256_after': after,
                  'shape_samples_by_traces': list(raw.data.shape), 'parameters': params, 'candidate_trace_ids': candidates,
                  'repaired_trace_count': int(changed.sum()), 'unmodified_other_traces_exact': True,
                  'research_baseline_candidates_match': baseline_matches, 'normal_trace_hide_and_restore': holdouts,
                  'roundtrips': roundtrips, 'package_verified': True, 'package_path': str(package.archive_path),
                  'warning': '已知正常道遮挡误差只适用于这些区段；不是异常道真值或目标损伤率。',
                  'desktop_acceptance': 'Not performed; Qt tests/preview rendering do not substitute for live Windows GUI acceptance.'}
        atomic_write_json(output / 'validation.json', result)
        return result
    finally:
        backend.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-windows-ns', default='0:100;100:200;250:850')
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.source, args.output, args.time_windows_ns, args.baseline), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
