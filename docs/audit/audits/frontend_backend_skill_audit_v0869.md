# MyGPR V0.8.69 前端/后端 Skill 式产品审计

## 审计目标

本轮不以新增功能为主，而是检查软件是否仍然让现场用户感觉像“AI 生成的研发半成品”：

- 前端：标签、按钮、空状态、错误提示、报告文字是否让勘探现场用户看得懂。
- 后端：稳定内部键、文件格式和自动化兼容性的同时，避免把英文机器键直接暴露给用户。
- 产品定位：默认正式模式继续保持“实际勘探定位工作台”，研发/仿真/benchmark 功能默认不可见。

## Skill 使用方式

没有引入未知第三方可执行 Skill 包。原因：公开资料显示 Skill 通常是包含说明、脚本和资源的文件夹，第三方 Skill 具备执行和供应链风险。本轮采用本地只读审计清单，等价于两个专用审计 Skill 的工作方式：

1. **Frontend Field UX Audit**
   - 检查主界面导航是否是现场流程语言。
   - 检查按钮是否说明“下一步要做什么”。
   - 检查空状态和错误提示是否给出用户动作。
   - 检查是否残留 AI、科研、仿真、benchmark、raw key、旧版、占位等字样。

2. **Backend Field Reliability Audit**
   - 检查稳定 schema / manifest / JSON 键是否保持兼容。
   - 检查用户界面和报告是否通过 label mapper 显示中文含义。
   - 检查错误、警告、导出报告、成果索引是否不暴露内部机器键。
   - 检查中文成果包名称是否能保留，不再被清洗成英文 fallback。

## 发现与修复

### 1. 交付文件表暴露英文内部键

原问题：成果生成后，底部“交付文件”表可能显示：

- `delivery_manifest`
- `delivery_report`
- `delivery_checksums`
- `processing_result`
- `line_record`

修复：新增 `core/user_labels.py`，将内部稳定键映射为：

- 成果清单
- 项目报告
- 文件校验清单
- 处理结果
- 测线记录

内部 manifest 仍保持原 schema，不破坏自动化和历史项目兼容性。

### 2. 成果报告暴露英文检查代码

原问题：`report.md` 的成果检查可能显示 `no_processing_result`、`no_interpretation` 之类机器键。

修复：报告中改为现场用户可读文字：

- 尚未保存处理结果
- 尚未添加目标标注
- 阻断 / 待复核 / 提示

### 3. “旧版处理窗口”看起来像未完成过渡页

原问题：主界面按钮包含“旧版处理窗口”“保存旧版处理结果”，用户会误以为当前软件新旧混杂、没有收口。

修复：改为：

- 打开完整处理
- 保存处理结果
- MyGPR 完整处理窗口

### 4. “处理版本”改为“处理结果”

原问题：“处理版本”偏研发过程记录，不够工程交付化。

修复：默认 UI 中面向用户的资源树、保存按钮、状态栏、文档页改为“处理结果”。内部 `processing_result` schema 不变。

### 5. 质控检查表暴露 raw code

原问题：质控表和成果检查表的“检查项”列展示内部 code。

修复：列名改为“检查内容”，并通过 `qc_code_label()` 显示中文解释。原 code 保存在 Qt UserRole 中，用于“确认警告”等后续逻辑。

### 6. 中文成果包名称被清洗掉

原问题：`field_delivery` 风格默认名仍偏英文；中文名称会被 `_safe_name()` 清洗成 fallback。

修复：默认成果包名改为“项目成果”，并允许 Unicode 文件夹名。

### 7. “工程 / 工作室 / 实验室 / 证据”等残留词继续收敛

修复：默认用户路径继续统一为：

- 项目
- 测线处理
- 目标定位
- 空间成果
- 成果报告
- 交付文件
- 技术记录

### 8. 研发和仿真相关能力继续保留但默认隐藏

`MYGPR_ENABLE_RESEARCH_UI=1` 或 `MYGPR_PRODUCT_MODE=research` 仍可打开研发入口；正式模式默认不显示。

## 验证

已执行：

```text
python scripts/preflight_check.py
python scripts/check_version_consistency.py --expected 0.8.69
python -m compileall -q app_qt.py cli_batch.py core ui scripts tests PythonModule
```

重点回归：

```text
53 passed, 1 skipped
68 passed
```

跳过项：当前 Linux/offscreen 环境缺少 Windows CJK 字体 fallback，不影响功能逻辑。

## 结论

V0.8.69 默认界面进一步压缩了“AI/科研/研发内测感”。用户看到的是勘探现场流程：项目、测线、目标、空间、报告；内部兼容键和研发工具仍保留在后端或开发模式中，不再直接干扰现场用户。
