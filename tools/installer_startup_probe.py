"""Measure genuine MyGPR Windows installations from outside the frozen process."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from runpy import run_path

ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
ctypes.windll.user32.GetDpiForWindow.argtypes = [wintypes.HWND]
ctypes.windll.user32.GetDpiForWindow.restype = wintypes.UINT
import h5py
import numpy as np
from PIL import ImageGrab
import win32api
import win32con
import win32gui
import win32process


STEPS = [
    {"method_id": "dewow", "label": "Dewow", "params": {"window": 11}, "enabled": True},
    {"method_id": "trace_savgol_filter", "label": "Savitzky-Golay",
     "params": {"window_traces": 7, "polyorder": 2}, "enabled": True},
]
EXPECTED_SHA = "c91538e9330dbf59d22ec03a1e0341e71808aff823c86356cd6ec774aecac6eb"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def until(find, description, *, timeout=45, interval=.035):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        value = find()
        if value is not None and value is not False:
            return value, time.perf_counter()
        time.sleep(interval)
    raise TimeoutError(description)


def install(artifact_folder, dest, expected_commit):
    metadata = read_json(artifact_folder / "build-provenance.json")
    installer = artifact_folder / "assets" / metadata["installer_name"]
    with installer.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if metadata["application_commit"] != expected_commit or digest != metadata["installer_sha256"]:
        raise ValueError(f"Invalid package provenance for {expected_commit}")
    logfile = dest.parent / f"install-{dest.name}.log"
    subprocess.run([str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                    "/SP-", "/CURRENTUSER", "/NOICONS", f"/DIR={dest}", f"/LOG={logfile}"],
                   check=True, timeout=180)
    exe = dest / "MyGPR.exe"
    embedded = read_json(dest / "_internal" / "windows-build.json")
    if not exe.is_file() or embedded["application_commit"] != expected_commit:
        raise AssertionError(f"Wrong app installed at {dest}")
    return exe, metadata


def settings_for(folder, fixture=None):
    conf = {"theme": "浅色主题", "project_root": str(folder),
            "recent_projects": [], "auto_prefetch_basemap": False, "update_automatic": False}
    if fixture:
        project = folder / "project"
        shutil.copytree(fixture, project)
        project_id = read_json(project / "project.json")["project_id"]
        revision = read_json(project / "raw/L01/L01_gpr_meta.json")["data_sha256"]
        conf["recent_projects"] = [str(project)]
        conf["processing_workspace_sessions"] = {"drafts": [{
            "key": [project_id, "L01", "original", revision],
            "value": {"steps": STEPS, "result_name": "startup-benchmark", "input_artifact_id": "",
                      "save_intermediates": True},
        }], "layouts": []}
    write_json(folder / "settings.json", conf)


def environment(folder, *, source=False):
    env = os.environ.copy()
    for key in ("PYTHONPATH", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        env.pop(key, None)
    for key, name in (("APPDATA", "appdata"), ("LOCALAPPDATA", "localappdata"),
                      ("TEMP", "temp"), ("TMP", "temp"), ("MYGPR_RUNTIME_ROOT", "runtime"),
                      ("MYGPR_EVIDENCE_ROOT", "evidence"), ("MYGPR_LOG_DIR", "logs"),
                      ("MYGPR_GPR_RESULT_RUNS", "runs")):
        directory = folder / name
        directory.mkdir(exist_ok=True)
        env[key] = str(directory)
    env.update(QT_QPA_PLATFORM="windows:darkmode=0", QT_ACCESSIBILITY="0",
               PYTHONIOENCODING="utf-8", MYGPR_SETTINGS_FILE=str(folder / "settings.json"))
    if not source:
        env["PATH"] = os.pathsep.join([str(Path(os.environ["SystemRoot"]) / "System32"),
                                      os.environ["SystemRoot"]])
    return env


def is_process_or_child(pid, parent):
    if pid == parent:
        return True
    try:
        handle = win32api.OpenProcess(0x1000, False, pid)
    except win32api.error:
        return False
    try:
        # PROCESS_BASIC_INFORMATION ends with UniqueProcessId and InheritedFromUniqueProcessId.
        info = (ctypes.c_void_p * 6)()
        status = ctypes.windll.ntdll.NtQueryInformationProcess(
            ctypes.c_void_p(int(handle)), 0, ctypes.byref(info), ctypes.sizeof(info), None)
        return status == 0 and info[5] == parent
    finally:
        handle.Close()


def main_window(pid):
    matches = []

    def visit(hwnd, _):
        if (win32gui.IsWindowVisible(hwnd) and "MyGPR" in win32gui.GetWindowText(hwnd)
                and is_process_or_child(win32process.GetWindowThreadProcessId(hwnd)[1], pid)):
            matches.append(hwnd)
        return True

    win32gui.EnumWindows(visit, None)
    return matches[0] if matches else None


def screenshot(hwnd, region=None):
    """Read only the foreground MyGPR client, discarding frames across a focus change."""
    if win32gui.GetForegroundWindow() != hwnd:
        return None
    _, _, width, height = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
    if not dpi or width < 500 or height < 400:
        return None
    scale = dpi / 96
    if region is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = region(width, height, scale)
    bounds = (left + x0, top + y0, left + x1, top + y1)
    frame = ImageGrab.grab(bounds, all_screens=True).convert("RGB")
    if win32gui.GetForegroundWindow() != hwnd:
        return None
    return frame, bounds, scale


def main_region(_w, _h, scale):
    return tuple(round(v * scale) for v in (12, 44, 154, 80))


def visible_frame(hwnd):
    shot = screenshot(hwnd, main_region)
    if shot is None:
        return None
    frame, _, _ = shot
    gray = np.asarray(frame).mean(axis=2)
    # The project chooser paints light surfaces with dark glyphs.
    return frame if (gray > 218).mean() > .60 and (gray < 100).sum() > 10 else None


def focus_runner_window(hwnd):
    # This runs only on the disposable desktop, whose foreground belongs to the test.
    if win32gui.GetForegroundWindow() != hwnd:
        try:
            win32gui.SetForegroundWindow(hwnd)
        except win32api.error:
            pass


def blue_pixels(rgb):
    rgb = np.asarray(rgb).astype(np.int16)
    return (rgb[:, :, 0] < 75) & (rgb[:, :, 1] > 57) & (rgb[:, :, 1] < 147) & (rgb[:, :, 2] > 200)


def settings_region(_w, _h, scale):
    return tuple(round(v * scale) for v in (5, 115, 43, 280))


def settings_frame(hwnd):
    shot = screenshot(hwnd, settings_region)
    if shot is None:
        return None
    frame, _, scale = shot
    mask = blue_pixels(frame)
    first_columns = mask[:, :round(20 * scale)]
    # A narrow primary-colored left edge identifies the checked B-scan category.
    return frame if first_columns.sum(axis=0).max() > 16 * scale else None


def run_region(width, _h, scale):
    return width - round(370 * scale), round(82 * scale), width - round(6 * scale), round(134 * scale)


def run_button(hwnd):
    shot = screenshot(hwnd, run_region)
    if shot is None:
        return None
    frame, (left, top, _, _), scale = shot
    mask = blue_pixels(frame)
    # Reject thin borders and glyphs. An enabled Run button has a solid blue fill.
    rows = np.flatnonzero(mask.sum(axis=1) > round(58 * scale))
    if rows.size < round(12 * scale):
        return None
    columns = np.flatnonzero(mask[rows].sum(axis=0) >= max(2, round(4 * scale)))
    return left + int(np.median(columns)), top + int(np.median(rows))


def key_chord(hwnd, key):
    if win32gui.GetForegroundWindow() != hwnd:
        raise RuntimeError("MyGPR lost foreground before its shortcut")
    started = time.perf_counter()
    win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
    try:
        win32api.keybd_event(key, 0, 0, 0)
        win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)
    finally:
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
    return started


def click(hwnd, location):
    if win32gui.GetForegroundWindow() != hwnd:
        raise RuntimeError("MyGPR lost foreground before clicking Run")
    win32api.SetCursorPos(location)
    started = time.perf_counter()
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    return started


def read_result(folder):
    line = folder / "project/processed/L01"
    try:
        info = read_json(line / "L01_params.json")
        if info["method"] != STEPS[-1]["method_id"] or info["method_name"] != "startup-benchmark":
            return None
        got_steps = info["params"]["pipeline"]
        if [(s["method_id"], s["params"]) for s in got_steps] != [(s["method_id"], s["params"]) for s in STEPS]:
            raise AssertionError(f"Unexpected processing pipeline {got_steps}")
        name = info["artifact_id"]
        dataset = folder / "project/data/lines/L01.artifacts" / (name + ".h5")
        with h5py.File(dataset, "r") as data:
            matrix = data[f"processing/artifacts/{name}/bscan"][:]
    except (FileNotFoundError, PermissionError, BlockingIOError, json.JSONDecodeError):
        return None
    if matrix.shape != (128, 64) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise AssertionError(f"Bad artifact matrix {matrix.shape} {matrix.dtype}")
    return hashlib.sha256(matrix.tobytes()).hexdigest()


def close(process, hwnd, folder):
    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
    if process.wait(timeout=30) != 0:
        raise RuntimeError("MyGPR did not exit cleanly")
    data = (folder / "logs/mygpr-events.jsonl").read_text(encoding="utf-8")
    if "app exit code=0" not in data:
        raise RuntimeError("Expected MyGPR app exit record is absent")


def operate(hwnd, folder, fixture):
    settings_click = key_chord(hwnd, ord("8"))
    rail, settings_shown = until(lambda: settings_frame(hwnd), "Settings B-scan navigation rendered")
    marks = {"settings_click": settings_click, "settings_visible": settings_shown}
    rail.save(folder / "settings-rail.png")
    ImageGrab.grab(window=hwnd).save(folder / "settings.png")
    if fixture:
        marks["processing_click"] = key_chord(hwnd, ord("3"))
        last_try = marks["processing_click"]

        def ready():
            nonlocal last_try
            button = run_button(hwnd)
            if button is None and time.perf_counter() - last_try > .25:
                last_try = key_chord(hwnd, ord("3"))
            return button

        location, marks["processing_ready"] = until(ready, "Restored pipeline Run button")
        marks["run_click"] = click(hwnd, location)
        digest, marks["artifact_readable"] = until(lambda: read_result(folder), "First result readable")
        marks["result_sha256"] = digest
        # Let the app draw the result before retaining the evidence image.
        until(lambda: run_button(hwnd) is None, "Processing left its runnable state")
        time.sleep(.3)
        ImageGrab.grab(window=hwnd).save(folder / "processing-result.png")
    return marks


def sample(command, cwd, folder, fixture=None, *, source=False):
    folder.mkdir()
    settings_for(folder, fixture)
    env = environment(folder, source=source)
    marks = {}
    hwnd = None
    with (folder / "console.log").open("wb") as log:
        origin = time.perf_counter()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            hwnd, marks["main_window_visible"] = until(lambda: main_window(process.pid), "Main window visible")
            if not source:
                focus_runner_window(hwnd)
            frame, marks["main_frame"] = until(lambda: visible_frame(hwnd), "First main window frame")
            frame.save(folder / "main-frame-region.png")
            marks.update(operate(hwnd, folder, fixture))
            close(process, hwnd, folder)
        except Exception:
            if hwnd and win32gui.IsWindow(hwnd):
                ImageGrab.grab(window=hwnd).save(folder / "failure.png")
            raise
        finally:
            if process.poll() is None:
                target = hwnd or main_window(process.pid)
                if target and win32gui.IsWindow(target):
                    win32gui.PostMessage(target, win32con.WM_CLOSE, 0, 0)
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=15)
    derived = {key: round((stamp - origin) * 1000, 2) if isinstance(stamp, float) else stamp
               for key, stamp in marks.items()}
    derived["settings_click_to_visible"] = round((marks["settings_visible"] - marks["settings_click"]) * 1000, 2)
    if fixture:
        derived["run_click_to_artifact_readable"] = round((marks["artifact_readable"] - marks["run_click"]) * 1000, 2)
    derived["normal_exit"] = True
    write_json(folder / "result.json", derived)
    print(folder.name, derived, flush=True)
    return derived


def compare(dest, executables, fixture, pairs):
    summary = {}
    for scenario, project in (("empty", None), ("restore", fixture)):
        samples = []
        for pair in range(1, pairs + 1):
            order = ("baseline", "candidate") if pair % 2 else ("candidate", "baseline")
            for version in order:
                exe = executables[version]
                row = sample([str(exe)], exe.parent, dest / f"{scenario}-{pair:02d}-{version}", project)
                samples.append({"version": version, **row})
        fields = [name for name, value in samples[0].items() if isinstance(value, float)]
        if project and len({row["result_sha256"] for row in samples}) != 1:
            raise AssertionError("The two installations produced different matrix bytes")
        median = {version: {key: statistics.median(row[key] for row in samples if row["version"] == version)
                            for key in fields} for version in ("baseline", "candidate")}
        paired = []
        for a, b in zip(samples[::2], samples[1::2], strict=True):
            rows = {a["version"]: a, b["version"]: b}
            paired.append({key: round(rows["candidate"][key] - rows["baseline"][key], 2) for key in fields})
        summary[scenario] = {"samples": samples, "median_ms": median, "paired_candidate_minus_baseline_ms": paired}
        write_json(dest / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--candidate-ref")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--pilot-source", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True)
    if not args.pilot_source:
        ImageGrab.grab((0, 0, 64, 64), all_screens=True)
    if args.pilot_source:
        repo = args.pilot_source.resolve()
        result = sample([sys.executable, "-B", str(repo / "app_qt.py")], repo,
                        out / "source-pilot", args.fixture.resolve(), source=True)
        if result["result_sha256"] != EXPECTED_SHA:
            raise AssertionError("Local source pilot differs from the earlier native reference")
        return
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Installed comparison requires a disposable Windows GitHub runner")
    if args.pairs < 1:
        parser.error("use at least one pair")
    run_path("packaging/windows/verify_install.py")["ensure_clean_installation"]()
    inputs = {"baseline": (args.baseline, args.baseline_ref), "candidate": (args.candidate, args.candidate_ref)}
    prepared = {key: install(artifact.resolve(), out / f"installed-{key}", sha) for key, (artifact, sha) in inputs.items()}
    write_json(out / "provenance.json", {key: provenance for key, (_, provenance) in prepared.items()})
    compare(out, {key: exe for key, (exe, _) in prepared.items()}, args.fixture.resolve(), args.pairs)


if __name__ == "__main__":
    main()
