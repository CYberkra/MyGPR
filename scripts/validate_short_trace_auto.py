"""Local cohort/classification comparison; counts are not detector accuracy."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from core.gpr_data_model import load_gpr_dataset
from core.storage_primitives import atomic_write_json
from mygpr.domain.processing.models import ProcessingRequest
from mygpr.infrastructure.processing.native_adapter import NativeProcessingExecutor


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def execute(data, time, window_mode, width, z, limit, mode="mark", spec=""):
    params = {
        "window_mode": window_mode,
        "window_traces": width,
        "robust_z": z,
        "max_run_traces": limit,
        "mode": mode,
        "time_windows_ns": spec,
    }
    return NativeProcessingExecutor().execute(
        ProcessingRequest(data=data, method_id="short_trace_repair", params=params, header_info={"time_axis_ns": time})
    )


def summary(result):
    keys = (
        "short_trace_candidate_mask",
        "short_trace_local_change_mask",
        "short_trace_zero_mask",
        "short_trace_repaired_mask",
        "short_trace_unresolved_mask",
    )
    return {key: (np.flatnonzero(result.trace_metadata[key]) + 1).tolist() for key in keys}


def validate(manifest, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new/empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads(manifest.read_text(encoding="utf-8"))
    preflight_path = manifest.parent / "preflight.json"
    preflight = {}
    if preflight_path.exists():
        preflight = {r["id"]: r for r in json.loads(preflight_path.read_text(encoding="utf-8")).get("records", [])}
    results = []
    seen = {}
    g2 = None
    for record in records:
        source = Path(record["path"])
        before = sha(source)
        entry = {"id": record["id"], "source": str(source), "sha256": before}
        known = preflight.get(record["id"], {})
        if known.get("file_sha256") == before and known.get("object_tree_error"):
            entry["known_source_container_warning"] = known["object_tree_error"]
        if before in seen:
            entry["identical_file_to"] = seen[before]
        seen.setdefault(before, record["id"])
        try:
            loaded = load_gpr_dataset(source)
            data = np.asarray(loaded.matrix, dtype=np.float32)
            time = np.asarray(loaded.time_axis_ns)
            if data.size > 50_000_000:
                raise ValueError("Outside this small-resource validation scope")
            entry.update(shape=list(data.shape), declared_time_ns=[float(time[0]), float(time[-1])])
            zero_columns = ~np.any(data, axis=0)
            entry["zero_trace_ids"] = (np.flatnonzero(zero_columns) + 1).tolist()
            if not np.any(data):
                entry["excluded"] = (
                    "all-zero stopped record; readable matrix is not proof the full container is healthy"
                )
            else:
                comparisons = []
                for name, scheme, width, z, limit, spec in (
                    ("user_old_full", "full", 15, 5.0, 3, ""),
                    ("user_auto", "auto", 15, 5.0, 3, ""),
                    ("previous_full", "full", 31, 8.0, 4, ""),
                    ("new_default", "auto", 31, 8.0, 4, ""),
                    ("previous_manual", "manual", 31, 8.0, 4, "0:100;100:200;250:850"),
                ):
                    if scheme == "manual" and (time[0] > 0 or time[-1] < 850):
                        comparisons.append({"name": name, "not_applicable": "fixed ns windows outside this record"})
                        continue
                    result = execute(data, time, scheme, width, z, limit, spec=spec)
                    np.testing.assert_array_equal(result.data, data)
                    repair = execute(data, time, scheme, width, z, limit, mode="interpolate", spec=spec)
                    changed = repair.trace_metadata["short_trace_repaired_mask"].astype(bool)
                    local = result.trace_metadata["short_trace_local_change_mask"].astype(bool)
                    assert not (changed & local).any()
                    np.testing.assert_array_equal(repair.data[:, ~changed], data[:, ~changed])
                    counts = {
                        "name": name,
                        "window_mode": scheme,
                        "width": width,
                        "z": z,
                        "max_run": limit,
                        "cross_or_legacy_count": int(result.trace_metadata["short_trace_candidate_mask"].sum()),
                        "local_count": int(local.sum()),
                        "eligible_and_repaired_count": int(changed.sum()),
                        "mark_samples_exact": True,
                        "nonrepaired_samples_exact": True,
                        "gates": result.metadata["gate_diagnostics"],
                        "zero_trace_cross_ids": (
                            np.flatnonzero(
                                zero_columns & result.trace_metadata["short_trace_candidate_mask"].astype(bool)
                            )
                            + 1
                        ).tolist(),
                    }
                    if record["id"] == "G2_02":
                        counts["masks"] = summary(result)
                        counts["repair_masks"] = summary(repair)
                    comparisons.append(counts)
                entry["comparisons"] = comparisons
                if record["id"] == "G2_02":
                    g2 = (data.copy(), time.copy())
        except (OSError, ValueError, KeyError) as exc:
            entry["error"] = repr(exc)
        assert before == sha(source)
        entry["source_sha256_unchanged"] = True
        results.append(entry)
    t = np.linspace(0, 900, 501)
    x = np.arange(129)
    background = np.repeat((0.02 * np.cos(t / 13) * np.exp(-t / 100)).astype(np.float32)[:, None], 129, axis=1)

    def pulse(center):
        u = (t - center) / 10
        return np.where(abs(u) <= 4, (1 - 2 * u * u) * np.exp(-u * u), 0).astype(np.float32)

    controlled = []
    for name, centers in [
        ("early_local", [50]),
        ("late_local", [400]),
        ("boundary_local", [300]),
        ("three_real_packets", [150, 450, 750]),
    ]:
        data = background.copy()
        for center in centers:
            data[:, 60:62] += pulse(center)[:, None]
        row = {"name": name, "true_target_traces": [61, 62]}
        for mode in ("full", "auto"):
            result = execute(data, t, mode, 31, 8.0, 4, mode="interpolate")
            row[mode] = {
                **summary(result),
                "relative_known_signal_change": float(
                    np.sqrt(
                        np.mean((result.data - data).astype(float) ** 2)
                        / np.mean((data - background).astype(float) ** 2)
                    )
                ),
            }
        controlled.append(row)
    for name, centers in [("hyperbola", np.sqrt(350.0**2 + (8 * (x - 64)) ** 2)), ("steep_event", 80 + 6 * x)]:
        data = background + np.stack([pulse(center) for center in centers], axis=1)
        row = {"name": name, "true_target_trace_count": 129}
        for mode in ("full", "auto"):
            result = execute(data, t, mode, 31, 8.0, 4, mode="interpolate")
            row[mode] = {
                **summary(result),
                "relative_known_signal_change": float(
                    np.sqrt(
                        np.mean((result.data - data).astype(float) ** 2)
                        / np.mean((data - background).astype(float) ** 2)
                    )
                ),
            }
        controlled.append(row)
    report = {
        "cohort": results,
        "controlled_examples": controlled,
        "controlled_background": "Finite nonzero common waveform; relative_known_signal_change denominator is constructed stored target (data minus common background). Whole-zero synthetic anchors are not treated as healthy references.",
        "default_choice": "auto equal-sample thirds,31-trace reference,z8,max4,mark only; uniform sampling corresponds to time thirds. Fixed proportions tested separately before implementation.",
        "interpretation": "No field bad-trace truth. Counts/stability are classification comparisons, not sensitivity/specificity. Local class is review-only. Cross-window candidate is not a hardware-fault diagnosis. Same interpolation equation retains its earlier steep-event fidelity limitations.",
    }
    atomic_write_json(output / "validation.json", report)
    if g2:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
        data, time = g2
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
        for row, scheme in enumerate(("full", "auto")):
            result = execute(data, time, scheme, 15, 5.0, 3)
            for col, (left, right) in enumerate(((549, 620), (699, 735))):
                gate = (time >= 0) & (time <= 150)
                part = data[gate, left:right]
                bound = float(np.percentile(abs(part), 99))
                ax = axes[row, col]
                ax.imshow(
                    part,
                    aspect="auto",
                    cmap="gray",
                    vmin=-bound,
                    vmax=bound,
                    extent=[left + 0.5, right + 0.5, float(time[gate][-1]), float(time[gate][0])],
                )
                for key, color in (("short_trace_candidate_mask", "red"), ("short_trace_local_change_mask", "orange")):
                    for index in np.flatnonzero(result.trace_metadata[key][left:right]):
                        ax.axvspan(left + index + 0.5, left + index + 1.5, color=color, alpha=0.2, lw=0)
                ax.set_title(("旧全窗候选" if scheme == "full" else "自动三窗分类") + "｜同为15道/5倍/最长3道，仅标记")
                ax.set_xlabel("道号（1起）")
                ax.set_ylabel("时间 ns")
        fig.legend(
            handles=[
                Patch(color="red", alpha=0.25, label="红：跨窗或旧全窗候选"),
                Patch(color="orange", alpha=0.25, label="橙：局部变化，仅审阅、不插补"),
            ],
            loc="outside lower center",
            ncol=2,
        )
        fig.suptitle("同一原始波形，仅比较分类覆盖；不因候选减少认定算法更准确")
        fig.savefig(output / "before_after_classification.png", dpi=160)
        plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.manifest, args.output)
    print(
        json.dumps(
            {
                "g2": next(r for r in report["cohort"] if r["id"] == "G2_02"),
                "excluded_or_errors": [r for r in report["cohort"] if "excluded" in r or "error" in r],
            },
            ensure_ascii=False,
        )
    )
