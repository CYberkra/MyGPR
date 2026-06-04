# MyGPR V0.7.3 测试分组与稳定基线

V0.7.3 的目标是把测试入口从“全量 pytest 一把跑”收敛为可解释、可重复的分组命令。当前仓库同时包含纯算法单元测试、Qt GUI 测试、gprMax contract 测试、slow runner 和集成路径；全量收集会混入 GUI/slow/gprMax 依赖，容易误判。

## 推荐打包前命令

```bash
python scripts/run_test_subset.py baseline
python scripts/run_test_subset.py gui-smoke -- -q
```

`baseline` 当前包含：

1. `scripts/preflight_check.py`
2. fast/headless `unit` 子集

`gui-smoke` 当前包含：

1. `tests/test_app_csv_load_runtime_seam.py`
2. `tests/test_import_export_report.py`

这两个命令分开执行是有意设计。Qt/Matplotlib GUI 批次在部分 headless sandbox 中会出现测试本身已通过但进程收尾较慢的问题，因此不把 GUI smoke 强行塞进 baseline 的同一个长进程链。

## 子集说明

```bash
python scripts/run_test_subset.py list unit
python scripts/run_test_subset.py list gui
python scripts/run_test_subset.py list gui-smoke
python scripts/run_test_subset.py list integration
python scripts/run_test_subset.py list gprmax
python scripts/run_test_subset.py list slow
```

| 子集 | 用途 | 是否建议每次打包跑 |
|---|---|---|
| `baseline` | preflight + fast unit | 是 |
| `gui-smoke` | 最小 Qt/报告烟测 | UI 或报告相关改动时跑 |
| `unit` | 快速、无 gprMax、无 GUI、无 slow 的算法/工具测试 | 是 |
| `gui` | GUI 回归测试 | UI 大改时跑 |
| `integration` | 非 gprMax 的多模块集成测试 | 后端/报告/CLI 改动时跑 |
| `gprmax` | gprMax contract、campaign、pairing、conversion 测试 | gprMax 相关改动时跑 |
| `slow` | candidate sweep、benchmark、runner、demo 类慢测试 | 发布前或专项审计跑 |
| `all` | 全量 pytest | 不作为日常命令，除非环境完整且时间充足 |

## V0.7.3 分类调整

- `tests/test_auto_tune.py` 从 fast unit baseline 中移出，归入 slow。该文件包含较重的 AutoTune 全面候选搜索，不适合作为每次快速基线的一部分。
- `gui-smoke` 被显式化，用于快速确认 Qt import、CSV load seam 和报告 sidecar 生成。
- `integration` runner 不再混入 gprMax 文件；gprMax 有独立子集。
- POSIX 环境下 GUI、preflight、integration 等可能触发 Qt 的命令会优先使用 `xvfb-run -a`。如需禁用，可设置 `MYGPR_TEST_NO_XVFB=1`。

## 本轮实测结果

- `python scripts/run_test_subset.py run --json-out docs/testing/v073_baseline_result.json baseline`
  - 通过。
  - unit 子集：`292 passed, 18 warnings`。
- `python scripts/run_test_subset.py gui-smoke -- -q`
  - 通过。
  - GUI smoke：`6 passed`。

保留 warning：preflight 仍会提示若干本机绝对路径引用。这些是下一轮 `路径配置 / 品牌残留清理` 的目标，不阻断当前稳定基线。
