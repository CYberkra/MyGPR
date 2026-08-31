#!/usr/bin/env python3
"""Verify the six full Yingshan field CSV files without loading them into RAM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def _hash_stream(handle) -> tuple[str, int]:
    digest = hashlib.sha256(); size = 0
    for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
        digest.update(block); size += len(block)
    return digest.hexdigest(), size


def _first_number(text: str) -> float:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if not match:
        raise ValueError(text)
    return float(match.group(0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Yingshan data directory or zip")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/test-results/yingshan-full-validation.json")
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "tests/fixtures/yingshan_real_v1/dataset_manifest.json").read_text(encoding="utf-8"))
    expected = {row["filename"]: (line_id, row) for line_id, row in manifest["lines"].items()}
    actual: dict[str, dict[str, object]] = {}
    errors: list[str] = []

    if args.source.is_dir():
        candidates = {path.name: path for path in args.source.rglob("*.csv")}
        for filename, (line_id, row) in expected.items():
            path = candidates.get(filename)
            if path is None:
                errors.append(f"missing {filename}"); continue
            with path.open("rb") as handle:
                digest, size = _hash_stream(handle)
            with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
                headers = [handle.readline().strip() for _ in range(4)]
            actual[filename] = {"line_id": line_id, "sha256": digest, "size_bytes": size, "headers": headers}
    else:
        with zipfile.ZipFile(args.source) as archive:
            members = {Path(name).name: name for name in archive.namelist() if name.lower().endswith(".csv")}
            for filename, (line_id, row) in expected.items():
                name = members.get(filename)
                if name is None:
                    errors.append(f"missing {filename}"); continue
                with archive.open(name) as handle:
                    digest, size = _hash_stream(handle)
                with archive.open(name) as handle:
                    headers = [handle.readline().decode("utf-8-sig", "ignore").strip() for _ in range(4)]
                actual[filename] = {"line_id": line_id, "sha256": digest, "size_bytes": size, "headers": headers}

    for filename, (_line_id, row) in expected.items():
        got = actual.get(filename)
        if not got: continue
        if got["sha256"] != row["sha256"]:
            errors.append(f"sha256 mismatch {filename}")
        if int(got["size_bytes"]) != int(row["size_bytes"]):
            errors.append(f"size mismatch {filename}")
        values = [_first_number(str(item)) for item in got["headers"]]
        expected_values = [row["sample_count"], row["time_window_ns"], row["trace_count"], row["trace_interval_m"]]
        if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(values, expected_values)):
            errors.append(f"header mismatch {filename}: {values} != {expected_values}")
    payload = {"schema": "mygpr.yingshan_full_validation.v1", "status": "failed" if errors else "passed", "files": actual, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if errors:
        print("[yingshan] FAILED"); [print(f" - {x}") for x in errors]; return 1
    print(f"[yingshan] PASS files={len(actual)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
