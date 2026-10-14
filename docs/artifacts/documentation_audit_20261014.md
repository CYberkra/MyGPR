# 仓库说明与依赖核对记录

日期：2026-10-14。版本仍为 0.9.38；这是工作区文档与依赖清单更新，包含上一轮尚未提交的采集 HDF5 兼容代码，不代表新版本发布。

## 范围

核查根目录使用/交接说明、AGENTS/CLAUDE、开发文档、用户文档、设计规格索引、requirements/pyproject/Conda 声明、CI Python 矩阵与截图收集配置。历史 CHANGELOG、ADR、旧验证快照保留原始证据；7 月交接增加历史状态提示。

## 已修正的问题

| 问题 | 处理 |
| --- | --- |
| AGENTS 仍写七页；GUI 说明写左侧导航、右侧固定日志栏 | 对齐八页顶部页签、底部输出面板 |
| GUI / 用户指南仍接受 Python 3.11 | 对齐 `requires-python >=3.12,<3.14`；本机实测 3.12.0 |
| GUI requirements 使用宽下限，pyproject GUI extra 漏掉 colorlog | 固定已安装的六项直接 GUI 依赖，并对齐两个安装入口 |
| requirements.txt 只有转引，核心版本重复维护 | 精确核心版本写入 requirements.txt；requirements-core.txt 作为兼容别名，默认仍无 Qt |
| Conda 环境的科学包无版本约束 | Python + pip 引用同一 requirements.txt；本轮仅静态核对，未创建 Conda 环境 |
| CLI 示例使用已移除的 recommended_profile，Python 示例缺少 pipeline 定义 | 改为显式 methods 和可独立运行的项目处理示例；实际执行通过 |
| “全部厂商格式均可直接导入”混淆底层读取器与项目向导 | 按入口列明：项目仅直接接受 CSV/TXT/NPY/NPZ/H5/HDF5；厂商二进制向导仍待接通 |
| 格式手册遗漏采集 HDF5 | 补充方向、时窗、real/magnitude、NaN、未知时窗和测距缺失语义 |
| 日志、设置与成果目录描述过时 | 按 core.app_paths、ui.constants、Hybrid Store 对齐 |
| 开发索引引用不存在的 terminology/field_ui/legacy 目录 | 现行索引改指向真实文档；旧协议明确标注为兼容/历史描述 |
| 发布文档声称已有四分片发布、hash-lock 检查与 exe spec | 区分现存源码/轮子检查与尚未建立的 exe 发布流程 |
| CI 截图上传路径仍固定 /tmp 或 D:/tmp | 两个 GUI job 显式设置 MYGPR_SMOKE_SHOTS_DIR，上传同一工作区目录 |
| 测试策略的 mypy 数量、子域数量及“无故障测试”说法过时 | 使用配置为基线；区分已有 ENOSPC/恢复测试和仍需实机完成的故障验收 |
| 文档把哈希、数值回归或离屏测试等同于物理/实机验证 | 明确各类证据的适用范围与缺口 |

## 依赖声明与本机环境

核心和 GUI 的 19 项直接依赖均与本机安装版本一致，完整列表写入 [README](../../README.md)、[AGENTS](../../AGENTS.md)、[requirements.txt](../../requirements.txt) 与 [GUI requirements](../../requirements-gui.txt)。`pyproject.toml`、requirements 递归包含与文档版本已程序核对。

开发环境另有以下差异，未将本机版本直接覆盖仓库的开发基线：

| 工具 | 仓库声明 | 本机实际 |
| --- | --- | --- |
| pytest | 8.4.2 | 9.1.1 |
| coverage | 7.13.3 | 7.16.0 |
| hypothesis | >=6.130,<7 | 未安装 |
| mutmut | >=3.3,<4 | 未安装 |

当前环境不是 `requirements-dev.txt` 的完整复刻。`pip check` 检查已安装包的依赖关系，不保证可选开发工具已齐全；本轮未声称完整质量门禁通过，也未升级或降级本机环境。

## SPEC 的实际位置

- 设计文档在 [docs/superpowers/specs](../superpowers/specs/README.md)，新增索引区分现行合同与历史设计。
- 当前无根目录 `SPEC.md`。
- PyInstaller 的 `gpr_gui.spec` 在 `a9fb92ef`（2026-07-31）删除；`git show a9fb92ef^:gpr_gui.spec` 可查看历史原件。旧 spec 依赖已删除的资源，不直接恢复为可用打包方案。

## 验证

- requirements 递归展开、pyproject 核心/GUI 声明、README/AGENTS 版本和已安装包：一致。
- `pip install --dry-run --no-index -r requirements-core.txt -r requirements-gui.txt`：通过，未修改环境。
- `pip check`、版本一致性、架构检查、release hygiene：通过。
- 最终 35 份入口/用户/开发说明与本记录的 128 个本地 Markdown 链接：通过。
- CLI 文档 JSON 解析、`validate` 和两个示例作业 `run`：通过。
- 文档 Python API 示例：实际执行通过，独立项目保存在 output/api_demo/。
- `test_version_consistency.py`、`test_gpr_format_registry_and_readers.py`、`test_cli_batch_profiles.py`：19 项通过。
- CI YAML 的截图环境变量与上传目录：本地结构校验通过；差异空白检查通过。

尚未重新验证干净安装、Linux/3.13、Conda 创建、Windows 人工点击或远端 CI。厂商格式向导接入、exe 打包和完整哈希锁属于实现/发布缺口，文档已如实标明。未执行 Git 提交或推送。
