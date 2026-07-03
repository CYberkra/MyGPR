# GX-UI-036 gprMax Simulation Validation Page

本轮把此前 backend-only 的 gprMax campaign dry-run 能力接入新的 project-first 工作台，形成安全的 `仿真验证` 工作空间。

## 完成内容

- 新增 `ui/simulation_validation_page.py`。
- `MyGPRWorkbenchWindow` 顶部生命周期工作区新增 `仿真验证`。
- 页面支持选择/输入 campaign YAML，并调用：
  - `core.gprmax_campaign.load_campaign_yaml`
  - `core.gprmax_campaign.validate_campaign`
- 显示 campaign 总体状态、场景 ready/warning/invalid 状态、问题详情和完整 JSON 摘要。
- 为选中的场景生成可复制的 `scripts/gprmax_campaign_runner.py --run-scene ...` 命令预览。
- 支持命令预览参数：variant、`--num-runs`、`--timeout-seconds`、外部 `--gprmax-python`、`--gpu` / `--gpu-devices`。
- invalid 场景禁用复制命令，避免误把未通过 dry-run 的场景拿去运行。
- 新增 UI 回归：`tests/test_simulation_validation_page_ui.py`。

## 安全边界

- 不从 GUI 直接启动长时间 gprMax。
- 不修改 `.in` / materials / ROI 模型文件。
- 不写 MyGPR-Evidence。
- 不宣称 production AutoTune 或 paper-candidate benchmark。
- 本页是 campaign readiness review + reproducible command planner，不是 batch scheduler。

## 后续建议

- 在可控环境中增加异步单场景运行按钮，并默认要求用户二次确认。
- 增加 paired output / preview-pair UI 表单，把 `target_response` 预览串进同一工作区。
- 后续若接入执行，应复用现有 `WorkbenchTaskWorker`，禁止阻塞主线程。
