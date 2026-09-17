#!/usr/bin/env bash
# autoresearch.sh — MyGPR 算法适应度基准 (36/36 注册方法全覆盖)
# 主指标: METRIC fitness_score (越高越好)
# 自检在脚本内完成: 编译失败 / 任务表与注册表不一致 → 非零退出
set -uo pipefail

cd "$(dirname "$0")"

# 选择可用解释器: 优先仓库 venv (Windows PE, MSYS bash 可执行);
# WSL bash 无 interop 无法执行 PE → 回退系统 python3 并确保基准依赖。
PY=""
if .venv/Scripts/python.exe --version >/dev/null 2>&1; then
    PY=.venv/Scripts/python.exe
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
    python3 -c "import numpy, scipy, pywt, psutil" >/dev/null 2>&1 || \
        pip3 install --quiet PyWavelets psutil
else
    echo "no usable python interpreter" >&2
    exit 1
fi

"$PY" -m py_compile scripts/algorithm_fitness_benchmark.py || exit 1
"$PY" - <<'EOF' || exit 1
import sys
sys.path.insert(0, ".")
from scripts.algorithm_fitness_benchmark import TASKS
from mygpr.infrastructure.processing.algorithms.methods import NATIVE_ALGORITHMS
covered = {m for _, m, _, _ in TASKS}
assert covered == set(NATIVE_ALGORITHMS), (
    f"task/registry mismatch: {covered ^ set(NATIVE_ALGORITHMS)}"
)
EOF

exec "$PY" scripts/algorithm_fitness_benchmark.py --run-all
