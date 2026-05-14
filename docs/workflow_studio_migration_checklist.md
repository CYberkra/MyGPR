# MyGPR Workflow Studio 功能迁移清单

> 本清单用于跟踪从传统 GUI 页面迁移到 Workflow Studio 的功能进度。

## 功能迁移状态

| 功能模块 | 传统页面 | 工作流集成度 | 状态 | 备注 |
|--------|--------|-----------|------|------|
| **数据导入** | BasicFlowPage | ✅ 已迁移 | Completed | 通过 Raw Input 节点集成，创建/更新按钮工作 |
| **B-scan 预览** | BasicFlowPage + Matplotlib 视图 | ✅ 已迁移 | Completed | 独立的 BscanViewerDialog，支持保存截图、复制到剪贴板 |
| **参数化处理** | BasicFlowPage | ✅ 已迁移 | Completed | 完整的工作流节点卡片和参数编辑 |
| **自动调参** | AutoTunePage | 🔄 部分迁移 | In Progress | Tuning Lab 对话框已实现，参数可写回工作流节点 |
| **高级显示设置** | AdvancedSettingsPage | 🔄 部分迁移 | In Progress | 在 B-scan Viewer 中已实现，暂未完全整合到 Studio |
| **QC/告警** | QualityLogPage | ❌ 未迁移 | Todo | 待整合到 Workflow Studio 的底部面板 |
| **保存证据包** | QualityLogPage | ❌ 未迁移 | Todo | 待定义证据包导出入口 |
| **导出报告** | QualityLogPage | ❌ 未迁移 | Todo | 待实现 |
| **导出图像/数据** | QualityLogPage | 🔄 部分迁移 | In Progress | B-scan Viewer 已有保存功能，完整导出功能待实现 |

---

## 详细状态

### 1. BasicFlowPage 功能
- ✅ **导入数据**：Raw Input 节点已完成
- ✅ **文件夹批量导入**：待验证
- ✅ **形状设置**：通过节点参数完成
- ✅ **单步处理**：Workflow 执行链已实现
- ✅ **参数编辑**：Inspector 面板已实现

### 2. AutoTunePage 功能
- 🔄 **自动调参**：Tuning Lab 对话框已集成到 Studio
- 🔄 **最佳参数应用**：已实现参数写回工作流节点的功能
- ❌ **实验证据保存**：待实现

### 3. AdvancedSettingsPage 功能
- 🔄 **显示设置**：B-scan Viewer 包含完整的显示控制
- 🔄 **色图/对比/裁剪**：在 Viewer 中已实现
- ❌ **持久化设置**：Viewer 的设置暂未与全局设置整合

### 4. QualityLogPage 功能
- ❌ **QC/告警面板**：待整合到 Workflow Studio 底部面板
- ❌ **日志面板**：已在 Workflow Studio 中实现基础版本
- ❌ **证据导出**：待定义在 Workflow Studio 中的入口
- ❌ **报告生成**：待实现

### 5. Matplotlib 主视图
- ✅ **B-scan 显示**：独立的 BscanViewerDialog，独立于主界面
- ✅ **控制功能**：缩放、适配、色图、反相等功能完整

---

## 待完成的任务

### P0 - 核心稳定
- ✅ B-scan Viewer 稳定性
- ✅ Preview 数据链路统一
- ✅ 节点运行状态跟踪
- ✅ Run 前验证

### P1 - 功能闭环
- ✅ Raw Input 节点真实化
- ✅ Sidecar 需求关联
- ✅ AutoTune 写回
- ❌ Save / Evidence / Export 职责固定

### P2 - 体验优化
- ✅ B-scan Viewer polish
- ❌ Splitter/侧栏状态记忆
- ❌ 节点卡片视觉优化
- ❌ 节点库添加位置优化
- ❌ 画布视图行为优化

---

## 后续计划

1. 首先确定 Save / Evidence / Export 的职责定义
2. 实现 Splitter 和侧栏状态的持久化
3. 完善节点卡片的视觉表现
4. 优化画布视图的交互逻辑
5. 整合剩余的 QC/证据导出功能
