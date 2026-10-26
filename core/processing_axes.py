"""Recover saved coordinates without borrowing a newer raw revision's axes."""

from datetime import datetime

import h5py
import numpy as np


def artifact_axes(source, manifest, fallback, load_manifest):
    metadata = manifest.get("input_dataset") or {}
    axes = []
    for name in ("distance_axis_m", "time_axis_ns", "depth_axis_m"):
        value = metadata.get(name)
        # Older JSON summaries contain only {shape, dtype}, not coordinates.
        axis = np.asarray(value, dtype=np.float32) if isinstance(value, (list, tuple)) else None
        if axis is not None and (axis.ndim != 1 or not axis.size or not np.isfinite(axis).all()):
            raise ValueError(f"成果持久化坐标轴无效：{name}")
        axes.append(axis)
    if all(axis is not None for axis in axes):
        return tuple(axes)
    if source.exists():
        with h5py.File(source, "r", libver="latest") as handle:
            raw = handle.get("raw")
            raw_stamp = raw.attrs.get("committed_at") if raw is not None else None
        # Raw commit metadata survives backups; container mtimes do not. Inline
        # result deletion also changes the file mtime without replacing raw.
        if not raw_stamp and any(axis is None for axis in axes):
            raise ValueError("成果缺少完整坐标轴及原始测线版本记录，无法安全恢复物理坐标轴。")
        raw_at = datetime.fromisoformat(str(raw_stamp).replace("Z", "+00:00"))
        seen = set()
        for _ in range(128):
            stamp = manifest.get("committed_at")
            if not stamp:
                raise ValueError("成果缺少坐标轴来源版本记录，无法安全恢复物理坐标轴。")
            saved_at = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
            if saved_at.tzinfo is None or raw_at.tzinfo is None:
                raise ValueError("成果坐标轴来源时间缺少时区，无法核对原始测线版本。")
            if raw_at > saved_at:
                raise ValueError(
                    "原始测线已更新，旧成果缺少完整坐标轴快照；无法安全恢复物理坐标轴。"
                    "请恢复生成该成果时的项目备份，或基于当前输入重新处理。"
                )
            parent = (manifest.get("input_dataset") or {}).get("parent_artifact_id") or manifest.get(
                "parent_artifact_id"
            )
            if not parent:
                break
            if parent in seen:
                raise ValueError("成果坐标轴来源记录存在循环，无法安全恢复物理坐标轴。")
            seen.add(parent)
            manifest = load_manifest(str(parent))
        else:
            raise ValueError("成果坐标轴来源记录过长，无法安全恢复物理坐标轴。")
    return tuple(saved if saved is not None else current for saved, current in zip(axes, fallback))
