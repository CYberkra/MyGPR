# 设计规格与现行合同索引

核查日期：2026-10-16。这里仍保存设计 SPEC；文件存在或计划打勾不等于当前源码、CI 或实机验收已经达成全部目标。

## 现行实现与合同

| 范围 | 入口 |
| --- | --- |
| 工作台七页导航、B-scan 标签、成果续算与菜单 | [2026-10-16 现行合同](2026-10-16-workbench-redesign.md) |
| 架构、线程与文件写入约束 | [CLAUDE.md](../../../CLAUDE.md)、[AGENTS.md](../../../AGENTS.md) |
| 当前能力与安装依赖 | [CURRENT_STATE.md](../../../CURRENT_STATE.md)、[README.md](../../../README.md) |
| 后端 API | [API 说明](../../architecture/BACKEND_API_V1.md)、`config/backend_api_v1.json` |
| 项目数据与持久化 | [Hybrid Store](../../developer/HYBRID_PROJECT_STORE_V1.md)、[数据协议](../../developer/project_data_contract.md) |
| 输入格式与采集 HDF5 | [格式参考](../../user/reference/数据格式参考.md) |
| 质量检查与证据范围 | [测试策略](../../developer/TESTING_STRATEGY.md) |

## 保存的设计记录

- [2026-09-03 工程化路线图](2026-09-03-engineering-roadmap.md)：分阶段决策与当时状态；未完成项需按当前代码重新核查。
- [2026-09-06 反卷积 / Inverse-Q](2026-09-06-deconv-inverseq-design.md)。
- [2026-09-06 幅度 C-scan](2026-09-06-phase4-amplitude-cscan-design.md)。
- [2026-09-15 ML 自动处理](2026-09-15-ml-autoprocess-design.md)：设计需求，不是训练或发布验收证明。
- [2026-07-03 三页设计](2026-07-03-three-page-redesign-design.md)、[7 月交接](HANDOFF.md)及同目录 7 月评审文件：历史 UI 方案；当前工作台合同见上表。

## `.spec` 打包文件

当前没有根目录 `SPEC.md`。旧 PyInstaller `gpr_gui.spec` 在 `a9fb92ef`（2026-07-31）清理时删除；查看原件：

```bash
git show a9fb92ef^:gpr_gui.spec
```

旧 spec 引用了已不存在的主题和启动资源，不应直接还原。现行配方为
[packaging/windows/MyGPR.spec](../../../packaging/windows/MyGPR.spec)，配合 Inno Setup；
重建和安装验收见 [Windows 构建说明](../../releases/windows-build.md)。打包 `.spec` 与设计规格是不同文件。
