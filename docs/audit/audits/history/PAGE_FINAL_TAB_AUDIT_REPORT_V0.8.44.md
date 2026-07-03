# MyGPR V0.8.44 全标签页最终审计与小修报告

## 审计对象

基线源码包：`MyGPR_V0.8.43_quality_space_layout.zip`

输出版本：`0.8.44`

本轮目标：对 `处理 / 选参 / 显示 / 质量 / 空间` 五个主标签页做最终职责审计，检查内容归属、显示完整性、处理链路边界、display-only 与 processing 分离、空间页空状态表达，以及残留命名/发布卫生问题。

本轮没有修改核心处理算法、AutoTune scoring v2、workflow planner、recipe runner、gprMax 数据链路或输入数据格式。

---

## 已完成的小修

### 1. 版本统一升级

- `VERSION`：`0.8.43` → `0.8.44`
- `README.md / START_MYGPR_README.md / gpr_gui.spec / tests/test_version_consistency.py` 同步到 `0.8.44`
- `CHANGELOG.md` 新增 `0.8.44` 记录

### 2. 清理运行源码内残留的旧助手命名

发现 `ui/gui_advanced_settings.py`、`ui/main_window_display_mixin.py`、`app_qt.py` 中仍有内部变量名 `chatgpt_style_var / chatgpt_style`，虽然用户界面显示为“自动对比度”，但源码包审计时仍不干净。

已改为：

```text
chatgpt_style_var → auto_contrast_var
chatgpt_style     → auto_contrast
```

复扫结果：

```text
ChatGPT / chatgpt / Codex / 何海：0 项运行源码命中
```

备注：少量数值数据文件中出现类似 `0.8.43` 的数字片段属于 CSV 数值，不是版本号。

### 3. 空间页避免无空间元数据时显示误导性三维帘幕

V0.8.43 中普通 B-scan / 仅距离轴数据会在空间页显示“非地理参考剖面预览”，但视觉上仍像一个三维成果，且某些无坐标数据会出现异常大的横向偏移轴，容易误解。

V0.8.44 修正为：

- 空间页只有在 `longitude/latitude + ground_elevation 或 flight_height` 条件满足时，才显示可展开/可导出的三维地理参考图层。
- 普通 B-scan 或只有距离轴的数据，空间页保留空状态提示，不再渲染假三维帘幕。
- 右侧仍显示当前点状态：坐标缺失、高程缺失、仅剖面预览。

这更符合空间页职责：空间页应展示真实空间成果，不应把普通剖面渲染成疑似三维成果。

### 4. 新增最终标签页职责测试

新增：

```text
tests/test_final_tab_responsibility_v0844.py
```

覆盖：

- 显示页可见控件不包含 RTK/IMU/高度计入口，也不包含 Dewow、SVD、AGC、背景抑制等真实处理入口。
- 处理页已经用“处理阶段筛选”替代静态文字流程说明。
- AutoTune 页主明细标签为 `流程 / 参数 / 候选 / 范围 / 说明 / 报告`。
- 质量页与空间页职责分离：质量页切换 QC/记录/报告/高级，空间页保留 RTK/IMU/高度计入口和空间空状态。

---

## 标签页审计结论

### 处理页

结论：基本合格。

当前定位已经清楚：数据导入、手动处理、阶段筛选、当前方法参数、撤销/重置。此前静态 `Raw / 校正 / 抑制 / 增/成` 卡片已经去掉，改成了实际可用的阶段筛选。

继续建议：当前参数面板在小窗口下仍较长，但有滚动区；后续可把高级参数折叠，但不是当前阻断问题。

### 选参页

结论：基本合格。

AutoTune 页当前符合“目标倾向 → 范围 → 生成推荐 → 运行方案 → 明细查看”的主流程。主标签 `流程 / 参数 / 候选 / 范围 / 说明 / 报告` 简洁，未把 trial table、manifest、claim boundary 等科研术语放在主入口。

继续建议：在左侧较窄时，流程表中长参数文本会截断；当前已有 tooltip 和横向滚动，短期可接受。后续可把“参数”标签页做成卡片式明细，减少表格截断。

### 显示页

结论：合格。

显示页当前只保留 display-only 功能：显示模式、色图、交互、显示增强、裁剪、对比、截图。RTK/IMU/高度计入口已迁出，Dewow、SVD、AGC 等真实处理入口没有出现在可见区域。

继续建议：保留 `operation_type=display_only` 测试，后续任何真实处理按钮不能放回显示页。

### 质量页

结论：合格。

质量页当前分为 `数据质量 / 处理记录 / 报告导出 / 高级`，职责清楚。页面用于 QC、处理记录、运行摘要、报告导出，不再承载完整空间成果浏览。

继续建议：质量页内图表仍可能在小窗口下需要滚动，这符合质量页信息密度，不是布局错误。

### 空间页

结论：本轮修正后合格。

空间页集中 RTK、IMU、高度计入口；显示坐标、高程、测线、C-scan、解释线状态。无空间元数据时显示明确空状态，不再把普通 B-scan 渲染成疑似三维成果。

继续建议：`Terrain3DResultsPage` 仍继承 `QualityLogPage`，这是结构耦合；功能上可用，但后续应拆成独立 `SpatialResultsPage + Georef3DRenderer`。

---

## 验证结果

通过：

```bash
python scripts/check_version_consistency.py --expected 0.8.44
python -m compileall -q app_qt.py core ui PythonModule tests
QT_QPA_PLATFORM=offscreen python scripts/preflight_check.py
```

通过 targeted tests：

```text
tests/test_final_tab_responsibility_v0844.py
tests/test_quality_space_layout_v0843.py
tests/test_spatial_sidecar_page_contract.py
tests/test_page_operation_contract.py
tests/test_basic_processing_stage_filter.py
tests/test_version_consistency.py

17 passed
```

GUI 实际启动截图覆盖：

```text
01_processing_stage_filter.png
02_autotune_recipe_tabs.png
03_display_only_enhance.png
04_quality_dashboard.png
05_space_empty_state.png
```

---

## 当前主体代码规模

统计口径：`app_qt.py`、`core/`、`ui/`、`PythonModule/`、`cli_batch.py`、`read_file_data.py`；不含 tests、scripts、docs、缓存和构建产物。

```text
主体 Python 文件：143 个
主体 Python 行数：59,281 行
```

当前最大文件：

| 文件 | 行数 |
|---|---:|
| app_qt.py | 3,795 |
| core/auto_tune.py | 3,400 |
| ui/autotune_tuning_page.py | 2,149 |
| PythonModule/kirchhoff_migration.py | 2,010 |
| core/methods_registry.py | 1,925 |
| ui/gui_quality_log.py | 1,675 |

---

## 最终判断

V0.8.41–V0.8.44 这一轮“页面职责整理”可以收口。

当前五个主标签页的职责已经基本稳定：

```text
处理：导入、手动处理、阶段筛选、真实处理链路
选参：AutoTune 推荐流程、参数、候选、范围、说明、报告
显示：只改变显示方式和对比，不改变处理数组
质量：QC、处理记录、运行摘要、报告导出
空间：RTK/IMU/高度计、航迹、地形、C-scan、三维/空间成果
```

下一阶段不建议继续做页面大改。更合理的后续方向是：

1. AutoTune scoring v2 的 breakdown 写入候选对比和报告。
2. 处理页参数面板做“常用参数 / 高级参数”折叠。
3. 空间页从 `QualityLogPage` 继承中拆出独立 renderer。
4. GSSI DZT、Sensors & Software DT1/HD 做真实样本读取验证。
