# MyGPR-CODE-CLEANUP-001 安全清理与源码审计报告

## 1. 输入与边界

- 输入源码包：`MyGPR_V0.8.16_smoother_slider_compare_ASCII.zip`
- 解压根目录：`MyGPR_V0.8.16`
- 执行时间 UTC：`2026-06-03T04:58:43+00:00`
- 本任务只在上传源码包的副本中执行，不访问也不修改你的本地 `D:\CDUT-UavGPR-Controller\MyGPR` 仓库。
- 本轮只清理缓存 / 编译产物 / 临时文件，不删除业务代码，不做架构重构，不改 AutoTune 算法，不改 UI 工作流。

## 2. 本轮删除内容

- 删除缓存项数量：1
- 删除缓存字节数：651
- 具体列表见：`deleted_cache_files.csv`

本轮删除范围仅限：`__pycache__/`、`.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`、`.ipynb_checkpoints/`、`*.pyc`、`*.pyo`、`.coverage` 和临时备份文件。

## 3. 本轮未删除内容

本轮没有删除以下类型：

- 核心源码
- UI 文件
- AutoTune 相关代码
- tests
- legacy fallback / Workbench 相关代码
- 文档、README、requirements
- experiments / Evidence 轻量状态文件
- scripts/autotune

## 4. 代码规模

| 口径 | 文件数 / 行数 |
|---|---:|
| Python 文件数 | 299 |
| Python 总物理行 | 106866 |
| Python 总有效行近似值 | 89000 |
| 不含 tests 的 Python 物理行 | 84597 |
| 不含 tests 的 Python 有效行近似值 | 71090 |
| 主体运行相关 Python 物理行 | 55589 |
| 主体运行相关 Python 有效行近似值 | 46903 |
| 主体运行且不含 AutoTune 字符路径的物理行 | 51989 |
| 主体运行且不含 AutoTune 字符路径的有效行近似值 | 43829 |

软著申报代码量建议仍按功能范围选择，不建议把 tests、临时脚本和实验审计脚本全部作为“源程序量”口径。

## 5. 大型或不应进入源码仓库的文件

- 发现数量：0
- 详见：`large_file_audit.csv`

判定规则：大于 5 MB 或扩展名属于 `.out/.h5/.vti/.vtk/.vtu/.npy` 的文件会被列入审计表。若为空，说明该源码包没有明显大型仿真原始产物混入。

## 6. 目录分类

主要分类结果见：

- `source_tree_inventory.csv`
- `module_classification.csv`
- `softcopyright_core_file_candidates.csv`
- `archive_candidate_files.csv`
- `do_not_delete_files.csv`

本轮只给出建议，不对 archive candidate 做删除。

## 7. .gitignore 审计

建议 `.gitignore` 覆盖以下规则：

```text
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
*.pyc
*.pyo
.coverage
build/
dist/
*.egg-info/
*.out
*.h5
*.vti
*.vtk
*.vtu
```

当前缺失建议规则：

```text
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
*.pyc
*.pyo
.coverage
build/
dist/
*.egg-info/
*.out
*.h5
*.vti
*.vtk
*.vtu
```

本轮没有直接修改 `.gitignore`，因为这是上传源码包副本。后续在本地仓库执行时可按报告补充。

## 8. 软著申报版建议

建议进入软著申报版源码快照的内容：

- 主程序入口
- UI 主界面与可视化模块
- 数据导入与 B-scan 显示
- 常规处理算法
- UAV-GPR metadata / motion compensation 相关模块
- AutoTune 核心功能
- 报告 / Evidence 导出
- 必要配置、README、requirements

建议暂不作为软著核心展示但保留的内容：

- 测试代码
- 历史实验脚本
- 一次性审计脚本
- 过期 demo
- 临时报告生成器

这些内容不应第一轮删除，应先归档或保留到软著版冻结后再处理。

## 9. 下一步建议

建议下一 gate：`MyGPR-SOFTCOPYRIGHT-002-FEATURE-FREEZE-AND-MANUAL-PLAN`

目标：冻结软著申报版功能范围，确定操作手册章节、截图清单、AutoTune 人性化改进项和源码快照规则。

## 10. Claim boundary

这是面向 MyGPR 软著准备的安全清理与源码审计任务。本任务只清理缓存和临时产物，不删除业务代码，不做应用架构重构，不修改 AutoTune 算法，不修改 UI 工作流，不运行 gprMax 或 AutoTune 实验，不制作最终发布包。
