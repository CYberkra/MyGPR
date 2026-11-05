# 应用内更新实施记录

范围：Windows x64 安装版。保留现有 PyQt6、PyInstaller、Inno Setup 架构。源码开发与其他系统只显示发布页入口。

- [x] 构建专属公钥配置、HTTPS 校验、稳定/预览独立地址。
- [x] WinSparkle 0.9.4 官方归档固定哈希、cdecl 绑定、原生下载与 EdDSA 验签。
- [x] 设置页检查入口、每日检查、渠道记忆；不额外增加介绍文案。
- [x] GUI 线程退出握手：任务、提交态、标注、保存失败、超时、停止和旧回调隔离。
- [x] AppMutex、正常关闭、取消恢复、按文件清单清理旧依赖。
- [x] 密钥生成、签名、本地验签、构建清单匹配与 appcast 生成；证书可选。
- [x] Windows 安装验收增加旧安装器升级模式。
- [x] 编写 GitHub 回归与原生签名/真实安装器小型夹具。
- [x] 功能分支已建立 [PR #36](https://github.com/CYberkra/MyGPR/pull/36)，全量 CI 与 `updates` 专项在 GitHub 执行；当前提交结果以 PR 检查页为准，不能沿用旧提交绿灯。
- [x] 独立静态审查及修复复核：渠道下次启动生效；检查完成前禁止重试；迟到错误不得解除安装冻结；回调保留至进程退出。
- [x] 正式私钥在发布者账户仓库外创建，权限限制为该账户和 SYSTEM；仅公钥进入源码和安装包。
- [ ] 发布者离线备份正式私钥。文件创建不等于离线备份。
- [x] 修复初始化失败后的偏好控件及错误提示；GitHub 先复现 6 个失败再验证修复。
- [x] 增加完整 Windows 包构建、真实 HTTPS 更新器和完整旧版升级的 GitHub 验收入口；执行结果以 PR 和 run 的实际证据为准。
- [x] 完成完整安装包与真实更新器验收：[原生升级验收](https://github.com/CYberkra/MyGPR/actions/runs/42083189676)，6 passed；完整包构建及升级源码为 `cc2c00064187a97a1ce9d4e2c8ab941c7af599f8`。
- [ ] 正式公网订阅发布。没有这一步不宣称已向用户提供自动更新。

本机只运行语法、Ruff、架构、Schema、类型和治理静态检查。测试只在 GitHub 执行。
密钥、第三方二进制、安装包与截图不进入 Git。不自动发布 tag 或 Release。

完整验收补充（2026-11-06）：
- [完整 Windows 构建](https://github.com/CYberkra/MyGPR/actions/runs/42081836896)：正式包及验收旧版包成功。
- [更新专项](https://github.com/CYberkra/MyGPR/actions/runs/42081826398)：Windows 59 passed；Linux 54 passed、5 Windows-only skipped。
- [全量 CI](https://github.com/CYberkra/MyGPR/actions/runs/42081836974)：五个门禁全部通过。
- 完整旧版覆盖升级、真实 HTTPS 更新、稍后、跳过、取消、中断、无效签名拒绝、正常退出和重启通过；合成原始数据、真实项目及处理成果、设置保留。
- 正式 v0.11.0 的遗留版本信息通过可信旧包哈希清单迁移；修改过的版本信息在覆盖前拒绝安装。
- 以上结果属于隔离 Windows runner 和合成验收数据。断电恢复、SmartScreen、正式公网订阅及发布者离线密钥备份仍未完成。
