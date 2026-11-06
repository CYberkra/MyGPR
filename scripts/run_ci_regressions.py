"""Compare the existing native-window regression on two source refs."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    if sys.platform != "win32":
        return 0
    root = Path(__file__).resolve().parents[1]
    baseline = Path(os.environ["RUNNER_TEMP"]) / "mygpr-native-baseline"
    base_ref = "4f74dffae2c1ade324b4d2fb40270c8375134f54"
    subprocess.run(["git", "worktree", "add", "--detach", str(baseline), base_ref], cwd=root, check=True)
    results = []
    for run in range(1, 4):
        for name, checkout in (("baseline", baseline), ("workbench", root)):
            print(f"Native OpenGL check {run}: {name}", flush=True)
            code = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_lazy_opengl_window.py", "-q", "--tb=short"],
                cwd=checkout, check=False,
            ).returncode
            results.append((name, code))
    print(f"Comparison: {results}", flush=True)
    return int(any(code for _name, code in results))


if __name__ == "__main__":
    raise SystemExit(main())
