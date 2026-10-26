# 传感器完整性与处理执行提示

日期：2026-10-19。工作仓库 `MyGPR_work`，分支 `codex/maintenance-20261017`，基线 `e379a5e`。
本次保留此前 H5 兼容修复和所有既有维护改动，未修改算法数值行为、真实采集数据或仪器设置。

## 用户可见行为

- 项目页选择测线后，在“传感器”详情页显示 RTK 经纬度、高度、IMU 姿态、加速度、角速度及采集时间的可用情况。
  按共同有效的道统计有效数/总数、百分比和缺失或无效数；字段缺省、长度不符、查询失败分别提示，不插值、不补零。
  有效率仅表示当前逐道字段可用，不代表 RTK 固定解、精度或时间同步已验收。
- 查询使用后台只读命令，不读取 B-scan 矩阵；项目代数和同线请求版本拦截过期回包，换项目清空缓存。
- 原生和文件分块处理路径均发送步骤执行结果：“已执行”“已执行（有警告）”或“已跳过（原因）”。
  既有顶层及 `effective_params.skipped` 跳过标记均受支持，算法自身行为保持不变。
- 任务消息和输出日志展示步骤结果/警告，最终完成消息保留跳过步数或警告数量。
  “任务完成”不再被用来暗示所有步骤均实际执行。
- 任务通知独立于进度事件保留，均沿用现有单任务事件保留上限；累计计数不会被裁剪。
  通知自身超出保留窗口时明确告知历史提示截断，forget/prune 随任务释放通知及计数。

## 验证

- 新增 14 项针对性回归，包括缺失统计、读取失败区分、跨项目/同线旧回包、任务消息、两种执行路径、
  嵌套跳过状态、350 条进度后晚订阅，以及有界通知与累计计数。
- 与原有项目页、项目读上下文、任务生命周期/关闭竞态和高度/姿态补偿测试共 107 项通过。
- 单独离屏渲染项目页“传感器”详情和任务消息，核对中文文字显示及滚动布局。
- 独立代码 review 发现两项 P2：嵌套跳过误报和进度淘汰通知；均用失败测试复现后修复，复核无剩余重要问题。
- 写回实际仓库后，加入此前两份 H5 导入测试文件再次回归，共 148 项通过；架构检查和 12 个 Python 文件编译检查通过。

相关回归入口：

```powershell
.venv\Scripts\python.exe -m pytest tests/test_sensor_completeness_notices.py tests/test_project_refresh_context.py tests/test_project_page.py tests/test_coordinator_chains.py tests/test_backend_processing_api.py tests/test_backend_jobs.py tests/test_backend_job_lifecycle_hardening.py tests/test_job_runner_shutdown_races.py tests/test_motion_compensation_height.py tests/test_motion_compensation_attitude.py -q
```

本次没有操作用户正在运行的窗口，没有进行真实硬件或野外测线验收，也未重建已安装的 EXE。
