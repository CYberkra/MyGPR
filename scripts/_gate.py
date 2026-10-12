# -*- coding: utf-8 -*-
"""棘轮门禁的共用骨架：baseline 读写 + 「可降不可升」比较 + 产物 JSON。

抽出动机（来自 2026-10-05 代码审查 §5.3）：``check_mypy_budget.py`` /
``check_debt_budget.py`` / ``check_complexity_budget.py`` 各自实现了一份
「读 config/*.json → 与当前值比 → 超了就 FAIL → 打印 → 返回 1」。三份的
措辞、baseline 缺失时的处理、``--write-baseline`` 的语义各不相同。

**保持三个独立入口**（不合并成一个 ``--metric`` 大开关）：CI 日志里
「mypy 棘轮」与「复杂度棘轮」必须各自可读，合并后失败信息会退化成
一行 ``[gate] FAILED``，看不出是哪条预算破了。

本模块只收敛**确定同构**的部分：baseline 文件读写、逐指标比较、
产物 JSON 落盘、退出码。不覆盖各门禁自己的度量逻辑。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RatchetGate:
    """一条「可降不可升」的预算线。

    Args:
        name: 门禁名（打印用，如 ``complexity``）。
        current: 本次实测值。
        baseline: baseline 里的值；两者都按数值比较。
        tolerance: 抖动容差比例。``0.05`` 表示允许比baseline 高 5%
            （吸收平台差异，如 Linux CI 与 Windows 上stub 可用性不同）。
        missing_ok: baseline 里**没有**该指标时的处置。棘轮语义下
            缺项应当是「无法判断」而非「通过」—— 默认 ``False`` 即
            报缺失，避免新增指标忘记写baseline 时静默放行。
    """

    name: str
    current: int
    baseline: int | None = None
    tolerance: float = 0.0
    missing_ok: bool = False

    @property
    def limit(self) -> int:
        """允许的上界 = baseline × (1 + tolerance)，向下取整。"""
        if self.baseline is None:
            return 0
        return int(self.baseline * (1 + self.tolerance))

    def violation(self) -> str | None:
        """返回违规描述；合规返回 ``None``。"""
        if self.baseline is None:
            if self.missing_ok:
                return None
            return f'{self.name}: no baseline recorded (add it or pass missing_ok)'
        if self.current > self.limit:
            return (f'{self.name}: {self.current} > baseline '
                    f'{self.baseline} (limit {self.limit})')
        return None

    def report(self) -> str:
        baseline = 'none' if self.baseline is None else str(self.baseline)
        return f'{self.name}: current={self.current} baseline={baseline} limit={self.limit}'


@dataclass
class GateRun:
    """一次门禁运行的聚合结果：多条 RatchetGate + 产物 JSON。"""

    name: str
    schema: str
    gates: list[RatchetGate] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def add(self, gate: RatchetGate) -> None:
        self.gates.append(gate)

    def violations(self) -> list[str]:
        """按门禁内顺序收集违规（顺序稳定，便于 CI 日志 diff）。"""
        return [text for gate in self.gates
                if (text := gate.violation()) is not None]

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'schema': self.schema,
            'status': 'failed' if self.violations() else 'passed',
            'metrics': {gate.name: gate.current for gate in self.gates},
            'baselines': {gate.name: gate.baseline for gate in self.gates},
        }
        if self.extra:
            payload.update(self.extra)
        return payload

    def emit(self, *, stream: Any = None) -> int:
        """打印逐条状态 + 违规明细，落产物 JSON，返回进程退出码。"""
        for gate in self.gates:
            print(gate.report(), file=stream)
        violations = self.violations()
        if violations:
            print(f'{self.name}: FAILED', file=stream)
            for text in violations:
                print(f' - {text}', file=stream)
            return 1
        print(f'{self.name}: PASS ({len(self.gates)} ratchet(s) enforced)', file=stream)
        return 0


def read_baseline(path: Path, *, field_name: str = 'metrics') -> dict[str, int]:
    """读 baseline JSON 的数字字典；文件不存在或字段缺失返回空字典。"""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    return {str(key): int(value) for key, value in dict(payload.get(field_name) or {}).items()}


def write_baseline(path: Path, payload: dict[str, Any]) -> Path:
    """写 baseline JSON（缩进 2 + 末尾换行，与既有 baseline 文件一致）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')
    return path


def add_write_baseline_flag(parser: Any) -> None:
    """给 argparse 加上三门禁共用的 ``--write-baseline``。"""
    parser.add_argument(
        '--write-baseline',
        action='store_true',
        help='record the current measurement as the new baseline '
             '(run this deliberately: it lowers the ratchet for everyone)')