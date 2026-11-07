"""Run an allowlisted focused regression suite on GitHub Actions only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "cscan": (
        "tests/test_amplitude_slice_domain.py",
        "tests/test_amplitude_slice_service.py",
        "tests/test_amplitude_slice_projection.py",
        "tests/test_cscan_curtain_geometry.py",
        "tests/test_cscan_curtain_preview.py",
        "tests/test_cscan_curtain_ui.py",
        "tests/test_cscan_spatial_result.py",
        "tests/test_amplitude_slice_ui.py",
    ),
    "update_installation": ("tests/test_windows_update_acceptance.py",),
    "updates": (
        "tests/test_windows_updates.py",
        "tests/test_update_controller.py",
        "tests/test_windows_update_native.py",
        "tests/test_windows_installer_upgrade.py",
    ),
    "bscan": (
        "tests/test_bscan_global_preferences.py",
        "tests/test_bscan_preference_mirror.py",
        "tests/test_bscan_qc_visibility.py",
        "tests/test_bscan_colorbar_toggle.py",
        "tests/test_bscan_viewport_levels.py",
        "tests/test_bscan_rectangle_selection.py",
        "tests/test_bscan_view_time_window.py",
        "tests/test_bscan_gain.py",
        "tests/test_bscan_elevation_warp.py",
        "tests/test_bscan_fullscreen_transition.py",
        "tests/test_bscan_qc_visibility.py",
    ),
    "algorithms": (
        "tests/test_image_denoising.py",
        "tests/test_manual_curve_gain.py",
        "tests/test_manual_gain_preview_core.py",
        "tests/test_native_convergence_baseline.py",
        "tests/test_native_processing_backend.py",
    ),
    "processing": (
        "tests/test_file_tree.py",
        "tests/test_home_page.py",
        "tests/test_project_workspace.py",
        "tests/test_page_controls_wired.py",
        "tests/test_spatial_delivery_ui.py",
        "tests/test_startup_project_restore.py",
        "tests/test_processing_page_v2.py",
        "tests/test_processing_workspace_flow.py",
        "tests/test_processing_async_identity.py",
        "tests/test_method_picker.py",
        "tests/test_component_contracts.py",
        "tests/test_design_tokens.py",
        "tests/test_param_form.py",
        "tests/test_manual_gain_editor.py",
        "tests/test_manual_gain_live_preview.py",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="bscan")
    args = parser.parse_args(argv)
    if os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("此入口仅供 GitHub Actions；日常不在本机启动自动化测试。")
    selected = SUITES.values() if args.suite == "all" else (SUITES[args.suite],)
    paths = tuple(dict.fromkeys(("tests/test_ci_regression_runner.py", *(p for group in selected for p in group))))
    missing = [path for path in paths if not (ROOT / path).is_file()]
    if missing:
        parser.error("当前提交缺少回归用例：" + ", ".join(missing))
    report = ROOT / "output" / "ci-regressions" / "junit.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    timeout = 900 if args.suite == 'update_installation' else 120
    command = [sys.executable, "-m", "pytest", *paths, "-q",
               f"--junitxml={report}", "--durations=15", f"--timeout={timeout}", "--timeout-method=thread"]
    print(f"GitHub focused suite: {args.suite}; {len(paths)} files", flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
