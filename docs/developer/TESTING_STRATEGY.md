# MyGPR 分层测试与风险门禁

适用版本：MyGPR v0.9.28 及后续版本。

## 1. 目标

MyGPR 不再把 900 余项全量测试作为每次修改的默认反馈路径。测试体系由中央策略文件 `config/test_impact.toml` 驱动，根据实际变更自动选择测试，并在高风险变更时升级门禁。

质量要求没有降低：全量自动化仍作为夜间和发布门禁保留；日常开发改为更快的静态检查、受影响测试和固定业务冒烟。

## 2. 门禁层级

| 门禁 | 典型用途 | 内容 |
|---|---|---|
| `l0` | 保存前、文档修改 | 编译、Ruff 严重错误、测试工具 Mypy、版本和策略一致性 |
| `affected` | 日常开发默认入口 | 由变更影响图自动决定 L0、受影响测试、合并门禁或完整门禁 |
| `smoke` | 快速业务健康检查 | 固定核心工程链，不包含所有 GUI 边缘情况 |
| `merge` | PR、合并前 | 受影响测试 + 核心冒烟 + GUI 冒烟 + 跨模块集成冒烟 |
| `nightly` | 每夜构建 | 按测试模块进程隔离运行非硬件、非 Windows、非大数据自动化 |
| `release` | 发布候选 | 全自动化模块隔离回归；随后进行 Windows、真实设备和大数据人工验收 |

`affected` 是推荐的自动入口。它可以对纯文档改动降到 L0，也可以把 Job Manager、工程存储、schema 或传感器同步改动直接提升到 `release`。

## 3. 常用命令

```bash
# 查看某次改动会触发什么，不执行测试
python scripts/run_quality_gate.py affected --changed-file core/gis_layers.py --plan

# 日常：根据 Git diff 自动选择并运行
python scripts/run_quality_gate.py affected

# 指定比较基线
python scripts/run_quality_gate.py merge --base origin/main

# 固定业务冒烟
python scripts/run_quality_gate.py smoke --no-promote

# 夜间和发布门禁
python scripts/run_quality_gate.py nightly --no-promote
python scripts/run_quality_gate.py release --no-promote
```

Windows 可使用：

```bat
scripts\run_quality_gate.bat affected
scripts\run_quality_gate.bat merge --base origin/main
```

## 4. 风险升级规则

中央影响图目前采用以下规则：

- 工程 schema、项目存储、处理会话和源文件注册：`release`；
- Job Manager、取消和事务回滚：`release`；
- 雷达—RTK—IMU 同步和逐道元数据：`release`；
- GIS、标注、正式报告、通用处理链和导入：`merge`；
- 自动调参、运动补偿、gprMax 专项：`affected`；
- UI 公共外壳：`merge`；
- 纯文档：`l0`；
- 未映射的生产代码：保守升级到 `merge`。

修改 `config/test_impact.toml` 本身会触发 `release`，防止通过缩小映射绕过验证。

## 5. 测试分类

所有测试在收集时由 `tests/conftest.py` 根据中央策略自动获得两类标记：

1. 执行层级：`unit`、`gui`、`integration`、`slow`、`gprmax`；
2. 业务域：`jobs`、`sync`、`gis`、`annotation`、`reporting`、`storage`、`project`、`processing` 等。

固定主链额外标记为 `smoke`。平台或资源测试使用 `windows`、`hardware`、`large_data`、`release_only`。

虽然可以直接执行 `pytest -m gis`，日常仍推荐使用 `run_quality_gate.py`，因为它会先缩小测试文件范围，避免 pytest 为了 marker 过滤而导入全部 232 个测试模块。

## 6. 原生 GUI 隔离

PyQt6、Matplotlib 和 VTK 含有进程级原生状态。门禁不会把所有 GUI 测试塞进同一 pytest 进程：

- 纯 headless 受影响测试合并执行以提高速度；
- GUI、gprMax、VTK 和交互式 Matplotlib 测试按文件隔离；
- 夜间和发布回归继续采用“一测试模块一进程”。

这避免把原生库析构冲突误判为业务断言失败。

## 7. CI

- `.github/workflows/quality-gates.yml`：PR/主分支风险门禁和 Windows GUI 冒烟；
- `.github/workflows/nightly.yml`：每日四分片隔离回归；
- `.github/workflows/release-gate.yml`：Linux/Windows 发布候选四分片回归。

每个门禁都会生成 `artifacts/test-results/*.json`，保存实际变更、匹配规则、测试清单、耗时和失败阶段。

## 8. 新增代码或测试时

新增生产模块时，必须在 `config/test_impact.toml` 中添加或确认影响规则。新增测试时，应通过 `test_groups` 使其获得正确业务域和执行层级。然后执行：

```bash
python scripts/check_test_policy.py
python scripts/run_quality_gate.py affected --changed-file <新文件> --plan
```

策略校验会检查缺失测试路径、无匹配测试组、未知 marker、无效门禁等级，以及所有测试是否至少拥有一个层级和业务域标记。

## 9. 现场与实机验收

自动化发布门禁不替代以下项目：Windows 125%/150%/175% DPI、多 GB/数十 GB 数据、外接硬盘断连、磁盘耗尽、断电恢复、真实 RTK/IMU/GPR 时间同步、错误 CRS、GNSS 丢解和正式报告打印。这些属于发布后的现场验收清单，而不是每次提交的本地反馈链。
