#!/usr/bin/env markdown

# 结构审视改造清单（2026-04-10）

这份清单来自对 `GPR_GUI_main_2026-04-12` 的结构性 review。  
目标不是一次性大重构，而是按风险和收益拆成几批小改动逐步落地。

## 使用方式

- 先处理“必须先修”的 correctness 问题
- 再处理“第一批结构收口”
- 最后再收尾清理和噪音项

每一项都尽量控制成可单独提交、可单独验证的改动。

## 一、必须先修

### 1. 修复报告裁剪信息失效

- 位置：`app_qt.py:3601-3607`
- 相关定义：`app_qt.py:4393-4462`
- 问题：`generate_report()` 调 `_get_crop_bounds(...)` 时参数不匹配，异常又被吞掉，导致报告里稳定退化为 `Crop: disabled`
- 为什么先修：这是明确的 correctness 问题，报告会和真实 UI 状态不一致
- 建议验收：
  - 有裁剪时，报告能正确显示裁剪范围
  - 无裁剪时，仍保持当前降级行为
  - 增加一个最小回归测试或最小可验证路径

### 2. 修复基础页 bool 参数解析

- 位置：`ui/gui_basic_flow.py:508-519, 523-554`
- 相关注册表：`core/methods_registry.py:373-383`
- 问题：基础页参数解析只分 `int/float/str`，像 `ccbs.use_custom_ref` 这种 bool 参数会变成字符串
- 为什么先修：这是方法契约错误，可能直接改变算法行为
- 建议验收：
  - bool 参数从基础页传下去时为真正 `bool`
  - 保持现有 int/float/string 行为不回归
  - 增加 GUI 参数解析测试

### 3. 统一单方法执行路径与 processing_engine 元信息契约

- 位置：`app_qt.py:1666-1685`
- 对照：`app_qt.py:304-360`、`core/processing_engine.py:98-170`
- 问题：单方法执行丢弃了 `display_data`、`runtime_warnings` 等 meta，导致两条执行路径行为不一致
- 为什么先修：这是结构性 correctness 问题，会影响后续功能的一致性
- 建议验收：
  - 单方法执行与 worker 执行共享统一 meta 语义
  - GUI 现有预览/日志/导出不回归
  - 至少补一个“方法返回 meta 时两条路径一致”的测试

### 4. 让方法注册与执行分派至少保持单点一致

- 位置：`core/processing_engine.py:29-46, 184-207`
- 对照：`core/methods_registry.py:49-667`
- 问题：注册表不是实际唯一真源，执行层还在手写 legacy/core 分派
- 为什么先修：后面继续加方法时，这会持续制造“UI 可见但运行不对”的风险
- 建议验收：
  - 新增或调整方法时，不需要同时改多个隐蔽分派点，或至少有显式校验
  - 注册信息和执行路径之间的断裂更少

## 二、第一批结构收口

### 5. 给 `app_qt.py` 做第一轮减重

- 当前文件：`app_qt.py:473-5453`
- 问题：主窗口同时承担 UI 组装、页面协调、绘图、导入、线程、报告、日志、ROI、auto-tune 等职责
- 建议拆分方向：
  - auto-tune 协调逻辑
  - 报告与导出逻辑
  - 主图/ROI/显示协调逻辑
- 目标：不是一次拆空，而是先把最重的一块独立成 helper/controller

### 6. 减少主窗口对页面内部私有实现的直接操纵

- 位置：`app_qt.py:1135-1215, 1585-1664, 1904, 2071, 2395, 3486`
- 问题：直接调用 `page_basic._render_params()`、`page_basic._get_params()`、直接写 `_method_param_overrides`
- 建议方向：
  - 页面对外暴露稳定公开方法
  - 主窗口不直接碰页面私有字段
- 目标：把页面从“控件集合”提升为“有边界的组件”

### 7. 真正收口 auto-tune 页的状态边界

- 位置：`app_qt.py:1898-1925, 2051-2251`
- 页面：`ui/gui_auto_tune_page.py`
- 问题：虽然有独立页，但当前方法切换、结果失效、推荐参数应用等逻辑仍主要留在主窗口
- 建议方向：
  - 把更多 auto-tune 状态操作下沉到 `AutoTunePage`
  - 主窗口只负责跨页面协调和线程生命周期
- 目标：后面如果加“后台预分析 / 历史候选 / 多方法缓存”时，不会继续把主窗口压垮

### 8. 拆分 `core/methods_registry.py`

- 当前文件：`core/methods_registry.py:49-1540`
- 问题：方法注册、GUI 预设、推荐流程、质量指标、Stolt 自适应、workflow 定义都塞在一起
- 建议方向：
  - 方法注册主表保留
  - GUI 预设单拆
  - 推荐流程单拆
  - 质量/自适应策略单拆
- 目标：把“一个文件是所有真源”的耦合改成“一个中心模块 + 清晰子模块”

## 三、第二批性能与契约优化

### 9. 收紧大数组复制和结果缓存

- 位置：`app_qt.py:248-259, 273-280, 351-360, 5235-5243`
- 相关：`PythonModule/sec_gain.py:26-32`
- 问题：输入、处理中间结果、输出在多处复制；batch/pipeline 长测线时内存压力高
- 建议方向：
  - 明确哪些地方必须 copy，哪些地方可以只保留引用或轻量 meta
  - 评估 `outputs` 是否必须始终保留完整数组
  - 检查 `gain_curve` 这类 metadata 是否应该全量保留

### 10. 统一高频 kernel 的参数修正与 warning 语义

- 文件：
  - `PythonModule/dewow.py`
  - `PythonModule/set_zero_time.py`
- 问题：有的路径静默 clamp，有的路径直接 validation error；GUI 层没有统一 warning 语义
- 建议方向：
  - 统一“参数被修正”时的 metadata / warning 约定
  - 让 GUI 和 auto-tune 都能读到同一类信号

### 11. 批处理页面（已移除 GUI 入口）

- 2026-04-13 重构后，`ui/gui_batch_report.py` 已不再作为主界面入口存在。
- 这一条保留为结构审查历史记录，不再作为当前 GUI 约束项。

## 四、测试补强

### 12. 补 GUI 新交互链路测试

- 当前文件：`tests/test_gui_presets.py`
- 当前缺口：
  - auto-tune 开始分析
  - 结果失效
  - 自动调参默认来源执行
  - 跨页同步
  - 取消流程

### 13. 补执行入口一致性测试

- 当前缺口：
  - 单方法执行 vs worker 执行的一致性
  - 报告裁剪信息
  - bool 参数从 UI 到 engine 的契约

## 五、可以延后收尾

### 14. 清理迁移残留逻辑

- 文件：`app_qt.py:1077-1079, 1217-1220, 2338-2353, 4108-4123`
- 问题：还有 `legacy_mode`、旧预设兼容等残留，增加阅读噪音

### 15. sidecar UI 入口要么接通、要么降噪

- 文件：`ui/gui_advanced_settings.py:304-341`、`app_qt.py:3805-3830`
- 问题：界面上已经有入口，但处理链里还没真正接通，容易误导用户

## 六、推荐实施顺序

### Phase 1

1. 报告裁剪信息修复
2. bool 参数解析修复
3. 单方法执行元信息统一

### Phase 2

1. auto-tune 页状态边界收口
2. 页面公开接口清理
3. GUI 交互链路测试补强

### Phase 3

1. `methods_registry.py` 职责拆分
2. 批处理 key 语义改造
3. 内存/复制优化

### Phase 4

1. 迁移残留清理
2. sidecar UI 收口

## 七、实施原则

- 每一项尽量独立提交
- 每一批先修 correctness，再动结构
- 不做“大重构一把梭”
- 每次改造后都补对应测试或最小验证路径
