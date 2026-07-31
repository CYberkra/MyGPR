# MyGPR 0.9.34 — Phase 18B-1 Processing Lab Interactive Migration

发布日期：2026-07-25

## 目标

将旧 Processing Lab 的高频生产能力迁移到 Clean-room MyGPR Studio，并保持前端 SDK、后端服务、项目成果谱系和冻结旧 UI 边界。

## 新增能力

- 交互式处理会话与只读快照。
- 单方法预览、逐步骤应用和现有步骤参数回载编辑。
- 步骤启停、排序、删除以及修改后自动重算后续链。
- 撤销、重做、重置和处理草稿自动恢复。
- 草稿按项目、测线、输入成果和处理分支隔离。
- 处理分支创建、项目级流程模板、跨测线模板复用。
- 项目全部测线独立后台任务批量提交。
- A-scan、频谱、原始/当前网格和差值分析。
- 交互会话保存为版本化处理成果，保留父成果和分支谱系。

## 架构

新能力路径：

```text
Studio ProcessingPage
  → frontend_sdk.ProcessingServiceProtocol
  → Phase12 adapter / Mock backend
  → ProcessingWorkbenchService
  → ProjectService + ProcessingService
```

`studio_app` 未导入 `ui/`、`compatibility/` 或旧 QWidget。Backend API v1 的冻结 facade 字段与方法未发生破坏性变更。

## 未完成项

本版本是 Phase 18B 的第一阶段，不宣称完整 Processing parity。以下留待 Phase 18B-2：

- 高级 AutoTune：目标、ROI、候选空间、评分权重、安全边界、Top-3 与审计。
- 可选择测线子集的批处理、失败重试和批量结果汇总。
- 步骤级质量指标、专用处理历史诊断和 Evidence/质量快照导出。
- 手动结果与推荐结果的专用对比及推荐报告。

## 验证边界

当前容器没有原生 PyQt6，Studio 验证使用离线 Headless Qt 合同层；Windows 真机视觉、DPI 和安装包验收仍是外部发布项。
