# 2026-11-02 色阶跟随查看范围、技能路由与 CI

## 源码定位

`BScanSelectionMixin` 原有三个入口均依赖先建立 `_view_selection` 或
`_view_time_window`：范围变化的定时器排队、定时器回调、色阶源矩阵截取。
普通滚轮缩放/平移并不设置这两个字段，因此即使右键“色阶跟随查看范围”已勾选，
仍按全矩阵统计，且不会随普通范围变化刷新。此前回归用例先执行框选，未覆盖这个入口差异。

修正为：跟随开关独立决定是否按当前 ViewBox 可见范围统计；普通缩放/平移经既有40ms
定时器合并更新。显式框选/数值时窗仍保留自己的状态更新；关闭开关恢复全矩阵统计；
共享固定色阶继续优先。清空数据时不交付在途更新。不修改矩阵或采样分辨率。

新增 `tests/test_bscan_viewport_levels.py`，覆盖普通缩放、平移、直接开关、固定色阶、
包络/显示增益/对比度、全览、清空和视野恢复，校验图像与色标同步。

## 技能路由

移除现行 AGENTS 与研发技能说明中对用户已删除技能的要求，不恢复或重建该技能。
改用现有 code-context-finder / engineering-feedback-loop / python-testing；
科学解释、图件和桌面验收按对应技能范围分别处理。数据形状、真实轴、元数据、
科学数值证据仍由项目合同约束，不由技能示例默认值决定。

调研候选：[OpenAI gh-fix-ci](https://github.com/openai/skills/tree/main/skills/.curated/gh-fix-ci)，
定位于 GitHub Actions 失败日志排查，未安装。没有增加第三方依赖或全局配置。
历史构建快照不属于现行配置，未改写历史存档。

## CI 配置

- AGENTS / CLAUDE / 研发技能工作流 / 测试策略统一：自动化测试默认 GitHub Actions；
  未经用户当前任务明确要求，不运行本地 pytest、冒烟、质量门禁包装或由子代理间接执行测试。
- 原 `MyGPR CI` 保留所有既有门禁与 job，增加JUnit和耗时记录；结果/GUI冒烟截图无论
  成功或失败都按job上传，保留7天；权限限定为读取仓库。
- 新增手动 `MyGPR focused regressions`，选择 bscan / algorithms / processing / all，
  在 Linux / Windows + Python3.13 + 离屏Qt执行固定清单。
- `scripts/run_ci_regressions.py` 是清单事实源；无shell拼接，退出码原样传播，
  缺少所选用例明确报错，普通本机环境拒绝启动。
- 新增入口契约用例，覆盖本机拒绝、白名单、退出码、JUnit和集合去重。
- 手动工作流文件须先进入默认分支才能由workflow_dispatch使用；首次PR仍由原完整CI验证。
  参见 [GitHub官方要求](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow#defining-inputs-for-manually-triggered-workflows)。

## 本轮证据与边界

**没有执行本地测试、应用冒烟或Qt交互复现**。问题定位依据实际代码调用链；
新旧回归均须交由对应提交的 GitHub CI 验证，不能称为已通过。
只执行了目标文件Ruff、Python AST语法、YAML语法/基本结构和清单路径存在性静态检查。
这些静态检查通过不等于GitHub执行通过。

独立只读代码审查无P1/P2；审查者只做源码/配置阅读与AST解析，没有运行测试。
新增视口回归从 `ViewBox.setRange()` 触发真实范围变更信号，尚未加入滚轮/拖动事件本身的端到端用例；
不能据此宣称鼠标交互已在Windows实机通过。

只读查到最近一次主分支run为 [40639926542](https://github.com/CYberkra/MyGPR/actions/runs/40639926542)，
结论success，提交 `ce14092b0578ac9536c4a062fca25706a44133ac`；该记录不包含本轮工作区修改。
本轮未提交、推送、创建PR、合并、打包或安装，当前状态为 **待CI验证**。
