"""Compare two MyGPR source snapshots in isolated native Windows launches."""
import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def prepare_fixture(repo: Path, project_path: Path) -> None:
    sys.path.insert(0, str(repo))
    import numpy as np

    from mygpr.interfaces.backend import MyGPRBackend

    backend = MyGPRBackend.create_default(max_workers=1)
    try:
        project = backend.projects.create_project(project_path, name="Startup benchmark")
        samples = np.arange(128, dtype=np.float32)[:, None]
        traces = np.arange(64, dtype=np.float32)[None, :]
        data = np.sin(samples * 0.18) + 0.1 * np.cos(traces * 0.23) + 0.003 * samples
        backend.projects.save_line_dataset(project.project_id, "L01", data,
                                          name="L01", length_m=6.3, time_window_ns=480.0)
    finally:
        backend.shutdown()


def child(repo, result, origin, restore):
    sys.path.insert(0, str(repo))
    sys.argv = [str(repo / "app_qt.py")]
    marks, spans, beats = {}, {}, []

    def now():
        return (time.perf_counter_ns() - origin) / 1e6

    def mark(key):
        marks.setdefault(key, now())

    mark("before_import")
    import app_qt
    mark("after_import")
    from PyQt6.QtCore import QEvent, QTimer, Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtTest import QTest

    audit = {}

    class Window(app_qt.MyGPRMainWindow):
        def __init__(self, *args, **kwargs):
            mark("window_begin")
            super().__init__(*args, **kwargs)
            mark("window_constructed")
            self._observed_settings = self.pages["settingsInterface"]
            self._observed_settings.installEventFilter(self)
            if restore:
                self.project_controller.dataset_preview_ready.connect(self.raw_available)
                self.processing_controller.run_result_ready.connect(self.result_available)
                self.project_controller.artifact_preview_ready.connect(self.artifact_displayed)
                self.processing_controller.run_finished.connect(self.processing_finished)
            self.probe = QTimer(self)
            self.probe.setInterval(25)
            self.probe.timeout.connect(self.observe)
            self.probe.start()

        def _register_page(self, object_name, *args, **kwargs):
            start = now()
            completed = super()._register_page(object_name, *args, **kwargs)
            spans[object_name] = now() - start
            return completed

        def paintEvent(self, event):
            result = super().paintEvent(event)
            if self.isVisible():
                mark("first_paint")
            return result

        def eventFilter(self, watched, event):
            if (event.type() == QEvent.Type.Paint and "settings_click" in marks
                    and watched is getattr(self, "_observed_settings", None)):
                mark("settings_paint")
            return super().eventFilter(watched, event)

        def _on_backend_ready(self):
            mark("backend_ready")
            result = super()._on_backend_ready()
            mark("backend_ui_done")
            return result

        def _finish_warmup(self):
            result = super()._finish_warmup()
            mark("pages_ready")
            return result

        def observe(self):
            try:
                self.advance()
            except Exception as error:  # Report probe failures before a normal window close.
                audit["error"] = repr(error)
                self.probe.stop()
                self.close()

        def advance(self):
            if self.isVisible() and "first_paint" in marks:
                beats.append(now())
                if not self.splashScreen.isVisible():
                    mark("first_ui_tick_after_splash")
                    if "settings_click" not in marks:
                        nav = self._top_nav.items["settingsInterface"]
                        if nav.isEnabled() and nav.isVisible():
                            mark("settings_click")
                            QTest.mouseClick(nav, Qt.MouseButton.LeftButton, pos=nav.rect().center())
                            mark("settings_click_return")
                    elif self.stackedWidget.currentWidget().objectName() == "settingsInterface":
                        mark("settings_event_loop_back")
                if {"settings_paint", "settings_event_loop_back", "backend_ready", "pages_ready"} <= marks.keys():
                    if not restore or self.advance_processing():
                        mark("all_conditions")
                        if now() - marks["all_conditions"] >= 200:
                            self.probe.stop()
                            mark("close_requested")
                            self.close()
            if now() > 40000:
                raise TimeoutError("startup workflow did not reach all milestones")

        def raw_available(self, bundle):
            if getattr(bundle, "matrix", None) is not None:
                mark("raw_preview_ready")

        def advance_processing(self):
            if "raw_preview_ready" not in marks:
                return False
            nav = self._top_nav.items["processingInterface"]
            if not nav.isEnabled() or not nav.isVisible():
                return False
            page = self.pages["processingInterface"]
            if "processing_click" not in marks:
                mark("processing_click")
                QTest.mouseClick(nav, Qt.MouseButton.LeftButton, pos=nav.rect().center())
                if self.stackedWidget.currentWidget() is not page:
                    raise RuntimeError("processing navigation did not show the page")
                return False
            if "processing_submit" not in marks:
                view = page._workspace.primary_view()
                if view._matrix is None:
                    return False
                mark("processing_view_ready")
                page._pipeline_list.add_step("dewow", "Dewow", {"window": 11})
                page._pipeline_list.add_step("trace_savgol_filter", "Savitzky-Golay",
                                             {"window_traces": 7, "polyorder": 2})
                page._result_name_edit.setText("startup-benchmark")
                if not page._run_btn.isEnabled():
                    raise RuntimeError("the prepared processing chain cannot run")
                mark("processing_submit")
                QTest.mouseClick(page._run_btn, Qt.MouseButton.LeftButton, pos=page._run_btn.rect().center())
                return False
            if audit.get("processing_error"):
                raise RuntimeError(audit["processing_error"])
            if not audit.get("processing_succeeded") or "artifact_shown" not in marks:
                return False
            if "processing_result_readable" not in marks:
                import numpy as np

                artifact = self.backend_controller.backend.projects.read_artifact_dataset(
                    self.project_controller.current_project_id, "L01", audit["artifact_id"])
                matrix = np.asarray(artifact.data)
                if matrix.shape != (128, 64) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
                    raise RuntimeError("unexpected processed dataset")
                audit["result_sha256"] = hashlib.sha256(matrix.tobytes()).hexdigest()
                mark("processing_result_readable")
            return True

        def result_available(self, _job_id, artifact_id):
            audit["artifact_id"] = artifact_id

        def artifact_displayed(self, artifact_id, bundle):
            if artifact_id == audit.get("artifact_id") and getattr(bundle, "matrix", None) is not None:
                mark("artifact_shown")

        def processing_finished(self, succeeded, message):
            audit["processing_succeeded"] = succeeded
            if not succeeded:
                audit["processing_error"] = message

        def closeEvent(self, event):
            result = super().closeEvent(event)
            if event.isAccepted():
                mark("close_accepted")
            return result

    app_qt.MyGPRMainWindow = Window
    code = app_qt.main()
    mark("main_return")
    platform = QGuiApplication.platformName()
    summary = {"platform": platform, "exit": code, "marks_ms": marks, "register_page_ms": spans,
               "metrics": audit, "python": sys.version.split()[0],
               "max_postpaint_timer_gap_ms": max((b-a for a, b in zip(beats, beats[1:])), default=0.)}
    result.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return code if "close_requested" in marks else 1


def isolated_environment(folder: Path) -> dict:
    paths = {key: folder / name for key, name in (
        ("APPDATA", "appdata"), ("LOCALAPPDATA", "localappdata"),
        ("TEMP", "temp"), ("TMP", "temp"), ("MYGPR_RUNTIME_ROOT", "runtime"),
        ("MYGPR_LOG_DIR", "logs"), ("MYGPR_EVIDENCE_ROOT", "evidence"),
        ("MYGPR_GPR_RESULT_RUNS", "gpr-runs"),
    )}
    for path in set(paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    return {**os.environ, **{key: str(path) for key, path in paths.items()},
            "MYGPR_SETTINGS_FILE": str(folder / "settings.json"),
            "QT_QPA_PLATFORM": "windows:darkmode=0", "PYTHONIOENCODING": "utf-8"}


def measure(folder, sources, pairs, restore):
    result = []
    order = [version for pair in range(pairs) for version in
             (("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline"))]
    fixture = folder / "fixture"
    script = str(Path(__file__).resolve())
    if restore:
        fixture_env = isolated_environment(folder / "prepare")
        repo = str(sources["baseline"])
        with (folder / "prepare" / "console.log").open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-B", script, "--make-fixture", repo, str(fixture)],
                           cwd=repo, env=fixture_env, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=45)
    for index, version in enumerate(order, 1):
        run = folder / f"run-{index:02d}-{version}"
        run.mkdir()
        env = isolated_environment(run)
        if restore:
            shutil.copytree(fixture, run / "project")
        config = Path(env["MYGPR_SETTINGS_FILE"])
        config.write_text(json.dumps({
            "project_root": str(run), "recent_projects": [str(run / "project")] if restore else [],
            "auto_prefetch_basemap": False, "update_automatic": False,
        }), encoding="utf-8")
        result_file = run / "result.json"
        command = [sys.executable, "-B", script, "--child",
                   str(sources[version]), str(result_file)]
        with (run / "console.log").open("w", encoding="utf-8") as stream:
            launched = time.perf_counter_ns()
            p = subprocess.run(command + [str(launched), str(int(restore))], cwd=sources[version], env=env,
                               stdout=stream, stderr=subprocess.STDOUT, timeout=55)
        sample = json.loads(result_file.read_text(encoding="utf-8"))
        sample["source"] = version
        marks = sample["marks_ms"]
        logs = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                         for path in (run / "logs").glob("*.jsonl"))
        sample["normal_exit_logged"] = "app exit code=0" in logs
        if (p.returncode != 0 or sample["platform"] != "windows" or "close_accepted" not in marks
                or not sample["normal_exit_logged"]):
            raise RuntimeError(f"Unable to verify native startup/close. See {result_file}")
        sample["derived_ms"] = {
            "app_qt_import": marks["after_import"] - marks["before_import"],
            "register_settings": sample["register_page_ms"]["settingsInterface"],
            "window_construct": marks["window_constructed"] - marks["window_begin"],
            "settings_click_to_paint": marks["settings_paint"] - marks["settings_click"],
            "settings_click_handler": marks["settings_click_return"] - marks["settings_click"],
            "settings_click_to_next_tick": marks["settings_event_loop_back"] - marks["settings_click"],
            "close_to_main_return": marks["main_return"] - marks["close_requested"],
            "max_postpaint_timer_gap": sample["max_postpaint_timer_gap_ms"],
            **{key: marks[key] for key in ("first_paint", "first_ui_tick_after_splash", "backend_ready", "pages_ready")},
        }
        if restore:
            sample["derived_ms"].update(
                raw_preview_ready=marks["raw_preview_ready"],
                processing_click_to_view=marks["processing_view_ready"] - marks["processing_click"],
                processing_submit_to_result=marks["processing_result_readable"] - marks["processing_submit"],
                processing_result_readable=marks["processing_result_readable"],
            )
        result.append(sample)
        print(version, {key: round(val, 1) for key, val in sample["derived_ms"].items()}, flush=True)
    medians = {}
    for version in sorted(set(order)):
        values = [row["derived_ms"] for row in result if row["source"] == version]
        medians[version] = {key: statistics.median(row[key] for row in values) for key in values[0]}
    deltas = []
    for left, right in zip(result[::2], result[1::2], strict=True):
        pair = {sample["source"]: sample["derived_ms"] for sample in (left, right)}
        deltas.append({key: pair["candidate"][key] - pair["baseline"][key] for key in pair["baseline"]})
    summary = {"scenario": "restore" if restore else "empty",
               "sources": {key: str(path) for key, path in sources.items()},
               "samples": result, "median_ms": medians, "paired_candidate_minus_baseline_ms": deltas}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2),
                                         encoding="utf-8")
    if restore and len({row["metrics"]["result_sha256"] for row in result}) != 1:
        raise RuntimeError("Processing results differed across compared source trees")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", default=3, type=int)
    parser.add_argument("--scenario", choices=("empty", "restore"), default="empty")
    args = parser.parse_args()
    sources = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    if sys.platform != "win32" or args.pairs < 1:
        parser.error("Requires Windows and at least one pair")
    if any(not (path / "app_qt.py").is_file() for path in sources.values()):
        parser.error("Both sources must contain app_qt.py")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    measure(output, sources, args.pairs, args.scenario == "restore")


if __name__ == "__main__":
    mode = sys.argv[1:2]
    if mode == ["--child"]:
        repo, result, origin, restore = sys.argv[2:]
        raise SystemExit(child(Path(repo), Path(result), int(origin), bool(int(restore))))
    if mode == ["--make-fixture"]:
        prepare_fixture(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
