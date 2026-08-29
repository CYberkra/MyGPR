# MyGPR 测试与质量门禁

适用版本：MyGPR v0.9.37 及后续版本。

> 历史说明：v0.9.28 时代文档描述过基于 `config/test_impact.toml` 的六级风险门禁
> （`run_quality_gate.py affected/merge/nightly/release`）。该体系及其脚本
> 已从当前树中移除，本文描述的是**当前实际存在并强制执行**的门禁。

## 1. 当前门禁清单

CI（`.github/workflows/backend-ci.yml`）对每个 PR / 主分支推送强制执行：

| 门禁 | 执行位置 | 内容 |
|---|---|---|
| 编译检查 | `backend` job | `scripts/check_python_compile.py`（全库可编译） |
| 架构红线 | `backend` job | `scripts/check_architecture.py`（分层方向、环、迁移所有权、增长限额） |
| Schema 目录 | `backend` job | `scripts/check_schema_catalog.py`（所有权 + 迁移政策） |
| 后端冒烟 | `backend` job | `backend_smoke.py`、`backend_project_smoke.py` |
| 全量 pytest + 覆盖率 | `backend` job | `pytest --cov=mygpr --cov=core`，随后 `check_coverage_policy.py` 棘轮校验 |
| 静态检查 | `backend` job | `ruff check .`（零容忍）；mypy 错误数棘轮（`scripts/check_mypy_budget.py`） |
| 治理预算 | `backend` job | `check_debt_budget.py`（宽异常/静默处理/sys.path/超长模块棘轮）、`check_complexity_budget.py`、`check_release_hygiene.py`、`check_project_format_compatibility.py` |
| Python 矩阵 | `backend-matrix` | 3.12 / 3.13 全量 pytest（3.11 已移除：钉版 numpy==2.5.1 要求 ≥3.12） |
| GUI 冒烟 + pytest | `gui-linux-offscreen` / `gui-windows` | 离屏/原生冒烟截图 + **带 Qt 的全量 pytest**（GUI 用例仅在含 Qt 的 job 执行） |
| clean-install | `clean-install` | 干净环境 `pip install .` + 契约测试 + `pip wheel` 可构建性 |

本地等价命令：`make gate`（即 `scripts/run_backend_quality_gate.py`，与 CI backend job 同一清单）。

## 2. 覆盖率棘轮

`config/coverage_policy.json` 定义全局与 6 个关键模块的行/分支覆盖率下限。
阈值按当前实测值设置并只收紧不放松（棘轮）；发现覆盖率下降时，修复测试而不是调低阈值。
全局目标（75%/60%）仍是长期方向，达成前以棘轮下限守门。

## 3. 静态分析棘轮

- **ruff**：零容忍，任何新问题都会让 CI 失败。遗留代码的既定风格差异
  （`scripts/`、`core/` 部分文件）通过 `pyproject.toml` 的 per-file-ignores 显式登记。
- **mypy**：全库 505 个既有错误登记在 `config/mypy_baseline.json`；
  `scripts/check_mypy_budget.py` 只允许减少不允许增加（容差 5%，吸收平台差异）。

## 4. 测试分层与标记

`pytest.ini` 以 `--strict-markers` 注册 marker。当前实际使用的执行层级：
`unit`、`integration`、`slow`、`large_data`、`industrial` 及 `tests/industrial/`
五个子域（acceptance/performance/property/reliability/scientific_validation/static_contract）。

约定：

- 需要 Qt 的测试模块**必须在模块顶部** `pytest.importorskip("PyQt6")`
  （无 Qt 的后端 CI 依赖它静默跳过；session 级 `qapp` fixture 见 `tests/conftest.py`）；
- 需要外部不可变数据的测试必须打 `external_data` marker 并用环境变量门控
  （现存示例：`MYGPR_YINGSHAN_DATA`）；
- 随机数据必须种子化；禁止 `time.sleep` 轮询式等待超过 50ms；
- 测试文件按被测模块命名（`test_<module>.py`）；`*_v09xx` 历史命名仅允许存量。

## 5. 未落地、明确不承诺的项

以下能力**当前不存在**，任何文档/计划引用它们前必须先实现：

- 基于影响图的受影响测试选择（无 `config/test_impact.toml`）；
- nightly / release 独立工作流；
- SBOM 生成与许可证扫描（`requirements-build.txt` 中的工具尚未接入任何自动化）；
- `config/industrial_acceptance_matrix.json` 中列出的现场验收项
  （磁盘满、断电、外接盘断连、10GB 长跑等）尚无自动化测试。

现场与实机验收（DPI、多 GB 数据、真实 RTK/IMU/GPR 同步、错误 CRS 等）属于
发布后人工验收清单，不由本地门禁替代。
