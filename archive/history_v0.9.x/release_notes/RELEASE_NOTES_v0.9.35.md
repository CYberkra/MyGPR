# MyGPR v0.9.35 Release Notes

发布日期：2026-07-25

## 主要更新

- 新 Studio 高级 AutoTune 支持处理目标、搜索模式、全图/自动/手动数值 ROI、推荐档位和带方向的评分权重。
- AutoTune 结果保留科学评分，并新增偏好评分、Top-3、置信度、风险提示和偏好审计。
- Processing Lab 增加步骤级质量得分、指标、警告和改进建议。
- 增加当前处理结果与推荐候选的专用比较。
- 批处理支持任意测线子集、失败/取消测线重试和批量结果汇总。
- 增加 `mygpr.processing_evidence.v1` 证据包，包含处理链、步骤诊断、比较指标、谱系和 SHA-256 完整性信息。

## 架构与兼容性

- Studio 继续只通过 `frontend_sdk` 调用后端。
- Backend API v1 facade 保持兼容。
- `ui/`、`compatibility/legacy_app_qt.py` 和其他冻结旧前端文件未修改。
- 新证据导出与高级 AutoTune 配置不改变既有项目成果 Schema。

## 已知限制

- 手动 ROI 当前为数值边界输入，画布拖拽框选将在后续版本完成。
- 跨方法流程组合候选空间、完整安全边界编辑器和专用推荐报告尚未完成。
- 当前容器无法替代 Windows 原生 PyQt6、DPI、EXE 和安装包验收。
