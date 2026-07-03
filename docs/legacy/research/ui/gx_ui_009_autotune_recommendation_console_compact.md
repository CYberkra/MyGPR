# GX-UI-009 AutoTune 参数推荐控制台紧凑版

## 目的

上一版 AutoTune 参数推荐页在 MyGPR 当前右侧工作区中出现横向挤压：左侧仍保留 B-scan 主图，右侧控制页宽度有限，三栏布局导致候选卡片、推荐解释和参数控件过窄。

本次将页面改为更适合当前主界面的紧凑结构。

## 主要变化

- 页面仍命名为 `AutoTune 参数推荐`
- 不再使用左/中/右三栏
- 改为：
  - 顶部状态栏
  - `① 配置`
  - `② 对比`
  - `③ 推荐`
  - `④ 审计`
- 选参控件集中在配置页
- 候选图框和排名集中在对比页
- 推荐解释、风险、claim boundary 集中在推荐页
- Trial Table / Metrics / Logs / Warnings / Boundary 集中在审计页

## 安全边界

本次仍只修改 UI 状态联动：

- 不运行 AutoTune
- 不修改 production scoring
- 不运行 gprMax
- 不修改 GX-008/GX-009 模型
- 不修改 MyGPR-Evidence
- 不引入 PyVista/PyVistaQt
- 保留 legacy `AutoTunePage` 兼容层

## 适用原因

当前 MyGPR 主界面是左侧 B-scan + 右侧功能页布局，因此 AutoTune 页不应假设自己拥有全屏宽度。紧凑版以标签页拆分信息密度，更适合嵌入当前主界面。
