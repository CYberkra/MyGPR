#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose gprMax GPU environment and minimal compile/smoke behavior."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMOKE_INPUT = ROOT / "experiments" / "gprmax" / "smoke" / "minimal_gpu_smoke.in"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose gprMax GPU environment.")
    parser.add_argument(
        "--clear-pycuda-cache",
        action="store_true",
        help="Clear discovered pycuda cache directories before diagnostics.",
    )
    parser.add_argument(
        "--skip-gprmax-smoke",
        action="store_true",
        help="Skip the minimal gprMax GPU smoke command.",
    )
    parser.add_argument(
        "--smoke-input",
        default=str(DEFAULT_SMOKE_INPUT),
        help="Path to minimal .in file for gprMax smoke.",
    )
    parser.add_argument(
        "--gprmax-python",
        default="",
        help="External runtime python path for gprMax module execution.",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="GPU device id used by minimal gprMax smoke.",
    )
    parser.add_argument(
        "--gprmax-cmd",
        default="",
        help="Deprecated explicit gprMax command override.",
    )
    parser.add_argument(
        "--smoke-timeout-seconds",
        type=float,
        default=180.0,
        help="Timeout for minimal gprMax smoke command.",
    )
    parser.add_argument(
        "--json",
        default="",
        help="Optional output path for JSON report.",
    )
    return parser.parse_args(argv)


def _run_command(cmd: list[str], timeout_s: float = 20.0) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        return {
            "ok": proc.returncode == 0,
            "return_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_seconds": max(0.0, time.perf_counter() - start),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "return_code": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "duration_seconds": max(0.0, time.perf_counter() - start),
            "timed_out": True,
        }
    except Exception as exc:  # pragma: no cover - defensive runtime path
        return {
            "ok": False,
            "return_code": None,
            "stdout": "",
            "stderr": repr(exc),
            "duration_seconds": max(0.0, time.perf_counter() - start),
            "timed_out": False,
        }


def _where(exe_name: str) -> dict[str, Any]:
    result = _run_command(["where", exe_name], timeout_s=10.0)
    paths = []
    if result["ok"]:
        paths = [line.strip() for line in result["stdout"].splitlines() if line.strip()]
    return {
        "available": bool(paths),
        "paths": paths,
        "stderr": result["stderr"],
    }


def _import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    return {
        "ok": True,
        "module_file": str(getattr(module, "__file__", "")),
        "version": str(getattr(module, "__version__", "")),
    }


def _collect_env_summary() -> dict[str, Any]:
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    filtered = []
    for entry in path_entries:
        low = entry.lower()
        if any(k in low for k in ("cuda", "nvidia", "visual studio", "msvc", "vc\\tools", "vctools")):
            filtered.append(entry)
    return {
        "platform": platform.platform(),
        "python_executable": os.sys.executable,
        "python_version": os.sys.version,
        "conda_env_name": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "msvc_env": {
            "VSINSTALLDIR": os.environ.get("VSINSTALLDIR", ""),
            "VCINSTALLDIR": os.environ.get("VCINSTALLDIR", ""),
            "VisualStudioVersion": os.environ.get("VisualStudioVersion", ""),
            "VCToolsInstallDir": os.environ.get("VCToolsInstallDir", ""),
            "WindowsSdkDir": os.environ.get("WindowsSdkDir", ""),
            "WindowsSDKVersion": os.environ.get("WindowsSDKVersion", ""),
            "UCRTVersion": os.environ.get("UCRTVersion", ""),
        },
        "path_cuda_msvc_related": filtered,
    }


def _pycuda_cache_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("PYCUDA_CACHE_DIR", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    tmp = Path(tempfile.gettempdir())
    for child in tmp.glob("pycuda-compiler-cache*"):
        candidates.append(child)
    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item.resolve()) if item.exists() else str(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _clear_pycuda_cache(paths: list[Path]) -> dict[str, Any]:
    removed: list[str] = []
    failed: list[dict[str, str]] = []
    for path in paths:
        try:
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                removed.append(str(path))
        except Exception as exc:
            failed.append({"path": str(path), "error": repr(exc)})
    return {"removed": removed, "failed": failed}


def _minimal_pycuda_compile_and_run() -> dict[str, Any]:
    try:
        import numpy as np
        import pycuda.autoinit  # type: ignore  # noqa: F401
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        from pycuda.driver import CompileError
    except Exception as exc:
        return {
            "minimal_pycuda_compile_ok": False,
            "error": repr(exc),
            "compile_error": None,
            "runtime_error": None,
        }

    try:
        mod = SourceModule(
            r"""
            __global__ void scale2(float *x) {
                int i = threadIdx.x;
                if (i < 4) x[i] = x[i] * 2.0f;
            }
            """
        )
        func = mod.get_function("scale2")
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        host_before = arr.copy()
        dev = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(dev, arr)
        func(dev, block=(4, 1, 1), grid=(1, 1))
        cuda.memcpy_dtoh(arr, dev)
        ok = bool(np.allclose(arr, host_before * 2.0))
        return {
            "minimal_pycuda_compile_ok": ok,
            "compile_error": None,
            "runtime_error": None if ok else "kernel output mismatch",
            "input": host_before.tolist(),
            "output": arr.tolist(),
        }
    except CompileError as exc:
        return {
            "minimal_pycuda_compile_ok": False,
            "compile_error": {
                "repr": repr(exc),
                "stdout": getattr(exc, "stdout", ""),
                "stderr": getattr(exc, "stderr", ""),
                "command_line": getattr(exc, "command_line", ""),
            },
            "runtime_error": None,
        }
    except Exception as exc:
        return {
            "minimal_pycuda_compile_ok": False,
            "compile_error": None,
            "runtime_error": repr(exc),
        }


def _resolve_gprmax_cmd(explicit: str, gprmax_python: str) -> str:
    if explicit.strip():
        return explicit.strip()
    if gprmax_python.strip():
        return f"{Path(gprmax_python).expanduser().resolve()} -m gprMax"
    candidate = Path(r"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe")
    if candidate.exists():
        return f"{candidate} -m gprMax"
    return "python -m gprMax"


def _split_cmd(command: str) -> list[str]:
    import shlex

    try:
        return shlex.split(command, posix=False)
    except Exception:
        return [command]


def _run_gprmax_smoke(
    command: str,
    smoke_input: Path,
    timeout_seconds: float,
    gpu_device: int,
) -> dict[str, Any]:
    if not smoke_input.exists():
        return {
            "ran": False,
            "ok": False,
            "error": f"smoke input missing: {smoke_input}",
        }

    workdir = Path(tempfile.mkdtemp(prefix="gprmax_gpu_smoke_"))
    copied_input = workdir / smoke_input.name
    shutil.copy2(smoke_input, copied_input)
    cmd = [*_split_cmd(command), str(copied_input), "-gpu", str(gpu_device)]
    result = _run_command(cmd, timeout_s=timeout_seconds)
    generated = sorted(str(p) for p in workdir.glob("*"))
    cleanup_errors: list[str] = []
    for p in workdir.glob("*"):
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        except Exception as exc:  # pragma: no cover - cleanup guard
            cleanup_errors.append(f"{p}: {exc!r}")
    try:
        workdir.rmdir()
    except Exception as exc:  # pragma: no cover - cleanup guard
        cleanup_errors.append(f"{workdir}: {exc!r}")
    return {
        "ran": True,
        "ok": bool(result["ok"]),
        "command": cmd,
        "return_code": result["return_code"],
        "timed_out": result["timed_out"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "generated_temp_files": generated,
        "cleanup_errors": cleanup_errors,
        "workdir": str(workdir),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {}
    report["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["environment"] = _collect_env_summary()
    report["imports"] = {
        "gprmax": _import_status("gprMax"),
        "pycuda": _import_status("pycuda"),
    }
    report["nvcc"] = {
        "where": _where("nvcc"),
        "version": _run_command(["nvcc", "--version"], timeout_s=10.0),
    }
    report["cl"] = {
        "where": _where("cl"),
        "version": _run_command(["cl"], timeout_s=10.0),
    }
    report["nvidia_smi"] = {
        "where": _where("nvidia-smi"),
        "default": _run_command(["nvidia-smi"], timeout_s=10.0),
        "query": _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            timeout_s=10.0,
        ),
    }

    cache_candidates = _pycuda_cache_candidates()
    cache_info = {
        "env_pycuda_cache_dir": os.environ.get("PYCUDA_CACHE_DIR", ""),
        "candidates": [str(p) for p in cache_candidates],
    }
    if args.clear_pycuda_cache:
        cache_info["clear_result"] = _clear_pycuda_cache(cache_candidates)
    report["pycuda_cache"] = cache_info

    report["minimal_pycuda"] = _minimal_pycuda_compile_and_run()
    gprmax_python_path = (
        Path(args.gprmax_python).expanduser().resolve()
        if args.gprmax_python.strip()
        else None
    )
    smoke_cmd = _resolve_gprmax_cmd(args.gprmax_cmd, args.gprmax_python)
    smoke_input = Path(args.smoke_input).expanduser().resolve()
    smoke_result = (
        {"ran": False, "ok": False, "skipped": True, "command_hint": smoke_cmd}
        if args.skip_gprmax_smoke
        else _run_gprmax_smoke(
            smoke_cmd,
            smoke_input,
            args.smoke_timeout_seconds,
            args.gpu_device,
        )
    )
    report["gprmax_smoke"] = smoke_result
    host_python_pycuda_available = bool(report["imports"]["pycuda"].get("ok"))
    gprmax_help_ok = False
    if gprmax_python_path is not None and gprmax_python_path.exists():
        help_result = _run_command(
            [str(gprmax_python_path), "-m", "gprMax", "--help"],
            timeout_s=15.0,
        )
        gprmax_help_ok = bool(help_result["ok"])
    elif gprmax_python_path is None:
        # Best effort from resolved command when explicit python is not provided.
        gprmax_help_ok = bool(smoke_result.get("ok")) if not args.skip_gprmax_smoke else False
    cl_available = bool(report["cl"]["where"]["available"])
    nvcc_available = bool(report["nvcc"]["where"]["available"])
    nvidia_smi_available = bool(report["nvidia_smi"]["where"]["available"])
    minimal_smoke_ok = bool(smoke_result.get("ok"))
    gprmax_runtime_gpu_ready = bool(minimal_smoke_ok and gprmax_help_ok)
    if gprmax_runtime_gpu_ready:
        readiness_reason = "external gprMax runtime smoke succeeded"
    elif not gprmax_help_ok:
        readiness_reason = "gprMax runtime help check failed"
    elif not cl_available:
        readiness_reason = "cl.exe unavailable in current shell context"
    elif not nvcc_available:
        readiness_reason = "nvcc unavailable in current shell context"
    elif not nvidia_smi_available:
        readiness_reason = "nvidia-smi unavailable in current shell context"
    else:
        readiness_reason = "minimal gprMax runtime GPU smoke failed"
    report["readiness"] = {
        "host_python_pycuda_available": host_python_pycuda_available,
        "gprmax_python": str(gprmax_python_path) if gprmax_python_path else "",
        "gprmax_python_exists": bool(gprmax_python_path and gprmax_python_path.exists()),
        "gprmax_help_ok": gprmax_help_ok,
        "cl_available": cl_available,
        "nvcc_available": nvcc_available,
        "nvidia_smi_available": nvidia_smi_available,
        "minimal_smoke_ok": minimal_smoke_ok,
        "gprmax_runtime_gpu_ready": gprmax_runtime_gpu_ready,
        "readiness_reason": readiness_reason,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_report(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        out = Path(args.json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"json_report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
