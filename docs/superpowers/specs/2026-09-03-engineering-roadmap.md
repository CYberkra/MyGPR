# MyGPR 工程化路线图（2026-09-03 定稿）

三视角辩论（现场工程师/架构师/产品商业化）+ 用户拍板的六项决策与分阶段实施计划。

## 已定稿的六项决策

| # | 决策点 | 结论 |
|---|---|---|
| 1 | A-scan 查看方式 | **B-scan 右键菜单可勾选"A-scan 波形跟随"**，独立浮窗（非模态，记住位置），点击剖面任意道实时显示该道波形；挂 BScanView 自身，处理页/解释页通用 |
| 2 | Wiggle/变面积 | **BScanView 显示模式三态**（灰度/变面积/波形叠加），右键或工具切换；变面积用 QPainterPath 批量构建 + QGraphicsPathItem |
| 3 | 深度切片 | **界面深度切片先行**（复用 basal_interface_annotations + build_georeference_3d + GIS 图层链），抽象沉淀后能量切片只换数据源 |
| 4 | 等值线交付格式 | PNG（报告包，Phase2 末）→ GeoTIFF + SHP + DXF 并列（Phase3 末，依赖等值线数据） |
| 5 | 速度分析 v1 | **仅双曲线拾取拟合**（共偏移距单覆盖数据无速度谱的物理基础）；结果套 AutoTune 证据链模式（error_code + SHA-256），写回测线速度模型 |
| 6 | 排期 | Phase1 显示层 → Phase2 速度分析+网格抽象 → Phase3 切片+三格式交付 |

## Phase 划分与验收

### Phase 1 — 显示层（1 个 PR，预计 1 周）
- [x] 1.1 A-scan 浮窗跟随（右键开关 + 浮窗 + 位置记忆）— feat/phase1-display 已合并 main（24e616e/8d6b43d/72ce730）
- [x] 1.2 Wiggle 三态显示模式 — b92606e（灰度/变面积/波形叠加，右键子菜单切换）
验收：真机点击剖面出波形；三态切换数据一致；全量测试绿；冒烟过 — 已达成（GUI 测试带 importorskip 守卫）

### Phase 2 — 速度分析 + 网格抽象（2 个 PR，预计 2 周）
- [ ] 2.1 双曲线拾取拟合（解释页交互 + 最小二乘 + 速度模型写回 + 证据链）
- [ ] 2.2 测线组/网格抽象（轨迹聚类成组 + application 层通用"网格化属性→GIS 图层"管线）
附带：PNG 深度/成果图进报告包（报告包机制现成）

### Phase 3 — 成图交付（2-3 个 PR，预计 2-3 周）
- [ ] 3.1 界面深度切片视图（深度滑条交互 + 等值线）
- [ ] 3.2 GeoTIFF/SHP 导出（rasterio/fiona，复用 GIS 图层）
- [ ] 3.3 DXF 导出（ezdxf 新依赖；测线轨迹/界面等值线/标注按图层组织）

### Phase 4 — 后备（未排期，记录在案）
能量属性切片（换 3.1 的数据源）；3D 体渲染（pyqtgraph GLVolumeItem）；采集工程导入（调研师兄控制器格式）；行业模块（路面层厚等）

## 架构约束（全 Phase 通用）

- ui→core 只能经 `ui/desktop_backend_facade.py`；`mygpr/application` 禁 import core（架构门禁强制）
- 新持久化结构登记 `config/schema_catalog.json`；新算法需 AlgorithmSpec
- 每个 Phase：feature 分支 → PR → CI 全绿 → 合并；禁止直接推 main（当前 token 限制 workflow 文件改动，CI workflow 修改需注意）
- 渲染数据流沿用 BScanView 契约：(samples, traces) 矩阵、零拷贝视图、鸭子类型 bundle

## 决策记录（辩论摘要）

- 三视角（现场/架构/商业）辩论，共识 4 项、分歧 2 项（切片先做哪种、DXF 排期）由裁决定为上文结论
- 关键反对意见存档：UAV 走测为共偏移距单覆盖数据，速度谱无物理基础（现场视角）；切片 A/B 之争的本质是"交付速度 vs 投标卖点"，A 先行沉淀抽象后 B 增量极小（架构视角）
