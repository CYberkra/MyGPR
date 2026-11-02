"""Read-only1.1.15 companions: source-row markers and CRC-bound CSV snapshots."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import json
import math
import uuid
import zlib

import h5py
import numpy as np

CSV_FORMAT = "cdut-acquisition-csv"
FLOAT_FIELDS = (
    "lon",
    "lat",
    "alt",
    "height",
    "roll",
    "pitch",
    "yaw",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "timestamp_ms",
)
MARKER_FIELDS = ("marker_id", "trace_no", "marker_type", "label", "marked_at", "lon", "lat", "alt", "height")
MAX_MARKERS = 10000


def _integer(value, *, low=0, high=2**64 - 1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError("CDUT integer field is invalid")
    return int(value)


def _json_bytes(contents):
    def bad_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    return json.loads(contents, parse_constant=bad_constant)


def _json_file(path, limit=1048576):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"CDUT companion must be a regular file: {path.name}")
    with path.open("rb") as stream:
        contents = stream.read(limit + 1)
    if len(contents) > limit:
        raise ValueError(f"CDUT companion exceeds size limit: {path.name}")
    return _json_bytes(contents), hashlib.sha256(contents).hexdigest()


def _marker_rows(rows, count):
    if not isinstance(rows, list) or len(rows) > MAX_MARKERS:
        raise ValueError("采集标记数量或类型无效")
    result = []
    seen_ids = set()
    unique_types = set()
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("采集标记记录无效")
        record = dict(item)
        mid = _integer(record.get("marker_id"), low=1, high=2**63 - 1)
        if mid in seen_ids:
            raise ValueError("采集标记ID重复")
        seen_ids.add(mid)
        record["marker_id"] = mid
        record["trace_no"] = _integer(record.get("trace_no"), low=1, high=count if count is not None else 2**63 - 1)
        kind = record.get("marker_type", "general")
        if kind not in ("start", "general", "end") or (kind != "general" and kind in unique_types):
            raise ValueError("采集标记类型无效或开始/结束重复")
        unique_types.add(kind)
        record["marker_type"] = kind
        for field, limit in (("label", 512), ("marked_at", 128)):
            if not isinstance(record.get(field), str) or len(record[field]) > limit:
                raise ValueError(f"采集标记文本无效: {field}")
        when = datetime.fromisoformat(record["marked_at"])
        if when.tzinfo is None:
            raise ValueError("采集标记时间缺少时区")
        for field in ("lon", "lat", "alt", "height"):
            value = record.get(field)
            if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
                record[field] = None
            elif (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
            ):
                raise ValueError(f"采集标记位置数值无效: {field}")
            else:
                record[field] = float(value)
        result.append(record)
    return result


def read_hdf5_markers(handle, trace_count):
    if "markers" not in handle:
        if "markers_schema_version" in handle.attrs:
            raise ValueError("HDF5声明了采集标记，但缺少/markers")
        return []
    if not isinstance(handle.get("markers", getlink=True), h5py.HardLink):
        raise ValueError("HDF5采集标记禁止外部/软链接")
    group = handle["markers"]
    if not isinstance(group, h5py.Group):
        raise ValueError("/markers must be a group")
    for attrs, key in (
        (handle.attrs, "markers_schema_version"),
        (handle.attrs, "trace_number_base"),
        (group.attrs, "schema_version"),
        (group.attrs, "trace_number_base"),
    ):
        if _integer(attrs.get(key), low=1, high=1) != 1:
            raise ValueError("Unsupported marker schema/base")
    lengths = set()
    for key in MARKER_FIELDS:
        if key not in group or not isinstance(group.get(key, getlink=True), h5py.HardLink):
            raise ValueError(f"HDF5采集标记字段缺失或为链接: {key}")
        ds = group[key]
        if not isinstance(ds, h5py.Dataset) or ds.ndim != 1 or len(ds) > MAX_MARKERS:
            raise ValueError(f"HDF5采集标记字段维度/数量无效: {key}")
        if key in ("marker_id", "trace_no") and ds.dtype.kind not in "iu":
            raise ValueError("Marker IDs must be integers")
        lengths.add(len(ds))
    if len(lengths) != 1:
        raise ValueError("HDF5采集标记字段长度不一致")
    rows = []
    for i in range(lengths.pop()):
        row = {}
        for key in MARKER_FIELDS:
            value = group[key][i]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            row[key] = value
        rows.append(row)
    return _marker_rows(rows, trace_count)


def csv_session(path, trace_count):
    path = Path(path)
    side = path.with_name(path.name + ".session.json")
    if not side.exists():
        return None
    binding, _ = _json_file(side, 65536)
    if (
        not isinstance(binding, dict)
        or _integer(binding.get("schema"), low=1, high=1) != 1
        or binding.get("data_file") != path.name
    ):
        raise ValueError("CSV测次绑定格式或文件名不匹配")
    identity = uuid.UUID(str(binding.get("acquisition_id", "")))
    if not identity.int:
        raise ValueError("CSV测次UUID无效")
    if _integer(binding.get("committed_trace_count")) != trace_count:
        raise ValueError("CSV测次提交行数不一致；请结束/恢复采集后再导入")
    return str(identity)


def marker_sidecars(path, trace_count=None):
    """Only explicit data_file basename matches are associated; never trust paths."""
    path = Path(path)
    candidates = sorted(path.parent.glob("*_markers.json"))
    if len(candidates) > 128:
        raise ValueError("标记文件过多，请将对应测次放入独立目录")
    matched = []
    for candidate in candidates:
        try:
            payload, digest = _json_file(candidate)
        except (ValueError, OSError, UnicodeError):
            # Unrelated markers do not establish a relationship to this source.
            if candidate.name == path.stem.removesuffix("_streaming") + "_markers.json":
                raise
            continue
        if not isinstance(payload, dict) or payload.get("data_file") != path.name:
            continue
        prefix = payload.get("file_prefix")
        if (
            not isinstance(prefix, str)
            or candidate.name != prefix + "_markers.json"
            or _integer(payload.get("schema_version"), low=1, high=1) != 1
            or _integer(payload.get("trace_number_base"), low=1, high=1) != 1
        ):
            raise ValueError("CSV采集标记绑定/版本无效")
        matched.append((_marker_rows(payload.get("markers"), trace_count), candidate, digest))
    if len(matched) > 1:
        raise ValueError("多个采集标记文件指向同一数据文件，拒绝猜测")
    return matched[0] if matched else ([], None, None)


def csv_companion_metadata(path, matrix, *, cancel_requested=None):
    path = Path(path)
    count = matrix.shape[1]
    identity = csv_session(path, count)
    marks, marker_file, marker_hash = marker_sidecars(path, count)
    result = {"acquisition_markers": marks}
    if marker_file:
        result["acquisition_marker_sources"] = [
            {"file_name": marker_file.name, "sha256": marker_hash, "association": "data_file_basename_only"}
        ]
    journal = path.with_name(path.name + ".meta.jsonl")
    if not journal.exists():
        if identity:
            raise ValueError("带测次身份的CSV缺少原始元数据日志")
        return result
    if journal.is_symlink():
        raise ValueError("CSV元数据日志禁止链接")
    from core.chunked_gpr_io import check_cancel

    fields = {key: [None] * count for key in (*FLOAT_FIELDS, "trace_id")}
    seen = set()
    digest = hashlib.sha256()
    with journal.open("rb") as stream:
        while True:
            check_cancel(cancel_requested)
            line = stream.readline(65537)
            if not line:
                break
            if len(line) > 65536:
                raise ValueError("CSV元数据日志单行过长")
            digest.update(line)
            if not line.strip():
                continue
            record = _json_bytes(line)
            if not isinstance(record, dict):
                raise ValueError("CSV元数据日志记录无效")
            row = _integer(record.get("row"), low=1, high=count)
            if row in seen:
                raise ValueError("CSV元数据日志道号重复")
            seen.add(row)
            if identity and record.get("acquisition_id") != identity:
                raise ValueError("CSV元数据UUID不匹配")
            if not identity and record.get("acquisition_id"):
                raise ValueError("CSV原记录声明UUID但缺少测次绑定")
            crc = zlib.crc32(np.asarray(matrix[:, row - 1], dtype="<f4").tobytes()) & 0xFFFFFFFF
            if _integer(record.get("samples_crc32"), high=2**32 - 1) != crc:
                raise ValueError(f"CSV元数据与波形CRC不一致（道{row}）")
            fields["trace_id"][row - 1] = _integer(record.get("trace_id"))
            for key in FLOAT_FIELDS:
                if key not in record:
                    raise ValueError(f"CSV元数据字段缺失: {key}")
                value = record[key]
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                ):
                    raise ValueError(f"CSV元数据数值无效: {key}")
                fields[key][row - 1] = float(value) if value is not None else None
    if len(seen) != count:
        raise ValueError("CSV原始元数据日志不完整")
    result.update(
        source_format=CSV_FORMAT,
        trace_count=count,
        trace_meta=fields,
        csv_metadata_status="uuid_and_crc_verified" if identity else "crc_verified_unbound",
        csv_journal_sha256=digest.hexdigest(),
        distance_axis_status="unknown",
    )
    if identity:
        result["acquisition_id"] = identity
    return result


def remove_owned_companions(directory, names):
    """Remove only validated companion basenames in the caller-owned directory."""
    directory = Path(directory)
    if not isinstance(names, (list, tuple)):
        raise ValueError("项目伴随文件清单无效")
    for name in names:
        if (
            not isinstance(name, str)
            or name != Path(name).name
            or not name.endswith((".sfcw.json", ".s21.jsonl", ".session.json", ".meta.jsonl", "_markers.json"))
        ):
            raise ValueError("项目伴随文件路径无效")
        target = directory / name
        if target.resolve().parent != directory.resolve():
            raise ValueError("项目伴随文件越界")
        target.unlink(missing_ok=True)


def copy_acquisition_companions(source, destination, previous_files, copy_file, **kwargs):
    """Project-local copies and stale companion removal stay inside its transaction."""
    source, destination = Path(source), Path(destination)
    suffixes = (".sfcw.json", ".s21.jsonl", ".session.json", ".meta.jsonl")
    relevant = (
        suffixes[:2]
        if source.suffix.lower() in {".h5", ".hdf5"}
        else suffixes[2:]
        if source.suffix.lower() in {".csv", ".txt"}
        else ()
    )
    files = [
        source.with_name(source.name + suffix) for suffix in relevant if source.with_name(source.name + suffix).exists()
    ]
    _, marker_file, _ = marker_sidecars(source) if source.suffix.lower() in {".csv", ".txt"} else ([], None, None)
    if marker_file:
        files.append(marker_file)
    incoming = {p.name for p in files}
    if not isinstance(previous_files, (list, tuple)):
        raise ValueError("项目伴随文件清单无效")
    stale = set(previous_files or []) | {destination.name + suffix for suffix in suffixes}
    remove_owned_companions(destination.parent, list(stale - incoming))
    for file in files:
        if file.is_symlink() or not file.is_file():
            raise ValueError("采集伴随文件必须是普通文件")
        copy_file(file, destination.parent / file.name, **kwargs)
    return sorted(incoming)
