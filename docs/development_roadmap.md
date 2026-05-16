# MyGPR Long-Term Development Roadmap

本文档用于长期推进 MyGPR，而不是记录一次性开发想法。

## 近期目标：可用稳定版

目标用户是项目组内部成员和导师演示。核心是能稳定完成日常处理和可解释展示。

- Workflow Studio 作为唯一主工作台。
- `mygpr_standard` 和 `high_quality_uav_gpr` 两套系统模板稳定可用。
- 每一步输出可以通过 B-scan Preview / Compare / QC / Evidence 节点查看。
- 导入真实 UAV-GPR SFCW CSV 后，数据上下文、shape、频带和 trace metadata 不丢。
- 无 RTK/IMU/AGL 时跳过运动补偿或显示风险，不伪造传感器数据。
- 全量测试和 preflight 作为合并门槛。

## 中期目标：科研论证版

目标是支撑组会、论文实验和专利材料。

- gprMax airborne 场景继续扩充，保留真实几何、空气层、地表反射、地下目标和高度变化。
- 自动选参报告保存每步 before/after、参数、指标、真值 ROI 和解释文字。
- 人工 baseline 至少区分经验参数、当前 GUI 参数、专家视觉参数和推荐 profile。
- 增益、背景抑制、去噪、迁移的评分指标分开记录，避免只看最终图像。
- Evidence Package 可复现实验链路。

## 长期目标：论文 / 专利版

目标是形成 MyGPR 的独立技术路线。

- 数据感知参数域：候选参数由样点数、道数、频带、时间窗、噪声水平和真值结构约束。
- 流程级自动选参：从单算法最优扩展到 stage 内比较和 pipeline 级搜索。
- 真值闭环：gprMax 场景输出结构 ROI、背景 ROI、深部噪声 ROI 和自动评分。
- 置信度与风险提示：输出推荐参数、候选分差、多峰最优、过处理风险和人工复核建议。
- 真实数据验证：出差获得 CSV + RTK + IMU + AGL 后补齐运动补偿与外场证据。
- 机器学习/深度学习作为后续增强，不作为当前可用性的前置条件。

## 记忆与交接策略

长期记忆不要依赖聊天记录。优先级如下：

1. 代码和测试：最可靠的项目记忆。
2. `AGENTS.md`: 跨 agent 的操作规则。
3. `docs/`: 架构、流程、契约、路线图。
4. Obsidian：组会、版本快照、阶段性结论。
5. Codex memory：只保存跨会话高价值规则和稳定结论。

每个稳定里程碑应至少包含：

- Git commit。
- 必要时 tag。
- 通过的验证命令。
- docs 或 Obsidian 的简短结论。

不要把大体量 HTML 报告、gprMax 原始输出、截图批量提交到 Git；它们应留在 `output/`，只把结论写入文档。
