"""Small local scientific audit; detection and known-mask interpolation are separate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
from core.gpr_data_model import load_gpr_dataset
from core.storage_primitives import atomic_write_json
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor
from mygpr.infrastructure.processing.algorithms.extended.short_trace_repair import _windows

GATES = '0:100;100:200;250:850'

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()

def execute(data, time, width=31, z=8., gates=GATES):
    return NativeProcessingExecutor().execute(ProcessingRequest(data=data, method_id='short_trace_repair',
        params={'mode':'interpolate','window_traces':width,'robust_z':z,'time_windows_ns':gates,'max_run_traces':4,
                'window_mode':'manual' if gates.strip() else 'full'},
        header_info={'time_axis_ns':time}))

def known_mask_fill(data, first, length):
    """Independent two-anchor equation, conditional on a known correct mask.

    Production equivalence is pinned by test_correct_known_detection_matches_independent_two_anchor_interpolation.
    This operator evaluation bypasses detection on purpose; its errors cannot be reported as detector accuracy.
    """
    weights = np.arange(1,length+1,dtype=float)[None,:]/(length+1)
    return ((1-weights)*data[:,first-1,None].astype(float)+weights*data[:,first+length,None].astype(float)).astype(np.float32)

def errors(truth, filled, time):
    result = {}
    for name, low, high in [('early',0,100),('middle',100,250),('late',250,850),('late_weak',500,850)]:
        gate = (time >= low) & (time <= high)
        original, estimate = truth[gate].astype(float), filled[gate].astype(float)
        rms = float(np.sqrt(np.mean(original**2)))
        error = float(np.sqrt(np.mean((estimate-original)**2)))
        ticks = time[gate]
        shift = ticks[np.argmax(abs(estimate),axis=0)]-ticks[np.argmax(abs(original),axis=0)]
        result[name] = {'ns':[low,high], 'truth_rms_stored_units':rms, 'error_rms_stored_units':error,
                        'relative_rms_error':error/rms if rms else None,
                        'abs_peak_time_shift_max_ns':float(abs(shift).max()),
                        'abs_peak_time_shift_median_ns':float(np.median(abs(shift)))}
    return result

def packet(time, center, width=10.):
    u = (time[:,None]-np.asarray(center)[None,:])/width
    return np.where(abs(u)<=4, (1-2*u*u)*np.exp(-u*u),0.).astype(np.float32)

def _plot_controlled_examples(plot_cases, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,3,figsize=(12,6),sharex=True,sharey=True,layout='constrained')
    for column,(name,original,processed) in enumerate(plot_cases):
        for row,matrix in enumerate((original,processed)):
            axes[row,column].imshow(matrix,aspect='auto',extent=[1,129,900,0],cmap='gray',vmin=-.02,vmax=.02)
            axes[row,column].set_title(name if row==0 else 'After three-gate repair')
            axes[row,column].set_xlabel('Trace')
    axes[0,0].set_ylabel('Known target only, time (ns)')
    axes[1,0].set_ylabel('Output minus known background, ns')
    fig.suptitle('Controlled examples only: identical amplitude scale; not field truth')
    fig.savefig(output/'controlled_reflection_counterexamples.png',dpi=160)
    plt.close(fig)


def audit(source: Path, output: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be new/empty')
    output.mkdir(parents=True,exist_ok=True)
    before = digest(source)
    loaded = load_gpr_dataset(source)
    data = np.asarray(loaded.matrix,dtype=np.float32)
    time = np.asarray(loaded.time_axis_ns)
    if data.shape != (501,1051):
        raise ValueError('This small-resource audit is scoped to the verified G2_02 file only')
    baseline = execute(data,time)
    flag = baseline.trace_metadata['short_trace_candidate_mask'].astype(bool)
    reference_ids = set((np.flatnonzero(flag)+1).tolist())
    sensitivity = []
    for width in (21,31,41):
        for z in (6.,8.,10.):
            result = execute(data,time,width,z)
            ids = set((np.flatnonzero(result.trace_metadata['short_trace_candidate_mask'])+1).tolist())
            sensitivity.append({'width':width,'z':z,'candidate_count':len(ids),
                                'repaired_count':int(result.trace_metadata['short_trace_repaired_mask'].sum()),
                                'jaccard_to_31_8':len(ids&reference_ids)/len(ids|reference_ids),
                                'added_trace_ids':sorted(ids-reference_ids),'removed_trace_ids':sorted(reference_ids-ids)})
    full = execute(data,time,gates='')
    full_ids = set((np.flatnonzero(full.trace_metadata['short_trace_candidate_mask'])+1).tolist())
    delta = data.astype(float)-median_filter(data.astype(float),size=(1,31),mode='reflect')
    detail = []
    for indices, info in _windows(GATES,{'time_axis_ns':time},data.shape[0]):
        score = np.sqrt(np.mean(delta[indices]**2,axis=0))
        center = np.median(score)
        sigma = 1.4826*np.median(abs(score-center))
        detail.append({**info,'median_rms':float(center),'sigma_mad':float(sigma),
                       'threshold':float(center+8*sigma),'differing_traces':[
                           {'trace':j,'rms_difference':float(score[j-1]),
                            'robust_z':float((score[j-1]-center)/sigma) if sigma else None,
                            'passes':bool(score[j-1]>center+8*sigma)} for j in sorted(full_ids^reference_ids)]})
    holdouts = []
    for trace in (180,280,480,680,880):
        for length in (1,2,3,4):
            first = trace-1
            if flag[max(0,first-16):min(len(flag),first+length+16)].any():
                continue
            truth = data[:,first:first+length]
            filled = known_mask_fill(data,first,length)
            holdouts.append({'trace_start':trace,'length':length,'metrics':errors(truth,filled,time)})
    synthetic_time = np.linspace(0,900,501)
    x = np.arange(129)
    background = np.repeat((.1*np.cos(synthetic_time/9)*np.exp(-synthetic_time/80)).astype(np.float32)[:,None],len(x),axis=1)
    local = np.zeros_like(background)
    local[:,60:62] = .02*packet(synthetic_time,np.array([400.,400.]))
    crossed = local.copy()
    crossed[:,60:62] += .02*packet(synthetic_time,np.array([50.,50.]))+.02*packet(synthetic_time,np.array([150.,150.]))
    hyperbola = .02*packet(synthetic_time,np.sqrt(350.**2+(8*(x-64))**2))
    dipping = .02*packet(synthetic_time,80+6*x)
    synthetic_cases = []
    plot_cases = []
    for name, target in [('single_late_local',local),('multigate_local_counterexample',crossed),
                          ('hyperbola',hyperbola),('steep_event',dipping)]:
        total = background+target
        result = execute(total,synthetic_time)
        detected = result.trace_metadata['short_trace_candidate_mask'].astype(bool)
        repaired = result.trace_metadata['short_trace_repaired_mask'].astype(bool)
        true_columns = np.any(target!=0,axis=0)
        residual = result.data-background
        synthetic_cases.append({'name':name,'target_column_count':int(true_columns.sum()),
            'marked_target_columns':int((detected&true_columns).sum()),'marked_total':int(detected.sum()),
            'repaired_target_columns':int((repaired&true_columns).sum()),
            'target_relative_rms_change':float(np.sqrt(np.mean((residual-target)**2)/np.mean(target**2))),
            'full_window_marked_target_columns':int((execute(total,synthetic_time,gates='').trace_metadata['short_trace_candidate_mask'].astype(bool)&true_columns).sum())})
        if name != 'steep_event':
            plot_cases.append((name,target,residual))
    synthetic_fill = []
    for name, target in [('hyperbola',hyperbola),('steep_event',dipping)]:
        for length in (1,2,3,4):
            first = 63
            estimate = known_mask_fill(target,first,length)
            truth = target[:,first:first+length]
            shifts = synthetic_time[np.argmax(abs(estimate),axis=0)]-synthetic_time[np.argmax(abs(truth),axis=0)]
            synthetic_fill.append({'case':name,'start_trace':first+1,'length':length,
                'relative_target_rms_error':float(np.sqrt(np.mean((estimate-truth).astype(float)**2)/np.mean(truth.astype(float)**2))),
                'max_abs_peak_time_shift_ns':float(abs(shifts).max())})
    _plot_controlled_examples(plot_cases, output)
    after = digest(source)
    assert before == after
    result = {'source':str(source),'source_sha256_unchanged':after,
        'baseline_candidate_trace_ids':sorted(reference_ids),'sensitivity':sensitivity,
        'full_vs_three_gates':{'full_candidates':len(full_ids),'full_repaired':int(full.trace_metadata['short_trace_repaired_mask'].sum()),
                              'three_candidates':len(reference_ids),'three_repaired':int(baseline.trace_metadata['short_trace_repaired_mask'].sum()),
                              'full_only_trace_ids':sorted(full_ids-reference_ids),'per_gate_diagnostics':detail},
        'known_mask_field_holdouts':holdouts,'controlled_detector_counterexamples':synthetic_cases,
        'known_mask_synthetic_interpolation':synthetic_fill,
        'metric_definitions':{'relative_rms':'sqrt(mean((fill-truth)^2))/sqrt(mean(truth^2)); all samples and held-out traces in named gate, no gain/normalization',
            'peak_time_shift':'absolute-maximum sample within the stated gate; field weak-wave peak is not a validated reflector arrival',
            'known_mask':'prescribed hidden positions bypass detector; ordinary raw traces are known only in the held-out experiment',
            'target_relative_rms_change':'constructed target only; output minus known synthetic background; zero means unchanged target',
            'limits':'No manually verified bad-trace truth. Candidate26 is a parameter-dependent output. Contrived counterexamples do not establish field sensitivity/specificity.'}}
    atomic_write_json(output/'audit.json',result)
    return result

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = audit(args.source,args.output)
    print(json.dumps({'output':str(args.output),'candidates':len(report['baseline_candidate_trace_ids']),
                      'field_holdouts':len(report['known_mask_field_holdouts']),
                      'counterexamples':report['controlled_detector_counterexamples']},ensure_ascii=False))
