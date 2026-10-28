# MyGPR 版本迭代规则

MyGPR 后续交付包采用语义化版本号：`MAJOR.MINOR.PATCH`。

当前维护交付版本为 **0.11.4**（2026-10-24），基于 0.11.0 和随后工作区维护。
不重置或改写已有版本/tag；后续修复继续递增 PATCH，新功能递增 MINOR。

## 版本号含义

- `MAJOR`：破坏兼容性的大版本，例如数据格式或主工作流不可兼容变化。
- `MINOR`：新增功能或较大模块升级，例如 AutoTune workflow planner。
- `PATCH`：bugfix、UI 小改、文档、测试、合规清理、轻量结构拆分。

## 每次交付必须同步更新

1. 根目录 `VERSION`。
2. `CHANGELOG.md` 中新增对应版本条目。
3. 交付 zip 文件名，例如 `MyGPR_V<版本号>_<tag>.zip`。
4. 同时更新 `pyproject.toml`；若重新建立 exe 打包，应将 `VERSION` 带入发布包。当前旧 `gpr_gui.spec` 已删除，见[发布流程](RELEASE_PROCESS.md)。
5. 修复报告或开发报告中必须写明 base version 和 output version。
6. README、AGENTS、CLAUDE、GUI_README、CURRENT_STATE、TODO 的当前版本，
   以及 `config/backend_source_manifest.json`。历史变更记录、设计和审计快照保持原版本。
7. Git tag 使用 `v<版本号>`，GitHub Release、wheel 文件名与发布资产清单使用同一版本。

## 默认递增规则

- 崩溃修复、测试修复、文档修复：递增 `PATCH`。
- 新增可见功能但不破坏兼容性：递增 `MINOR`。
- 数据格式、插件契约、Evidence manifest 产生破坏性变化：递增 `MAJOR`，并写迁移说明。

## 自动检查

关于页与报告优先读取源码根 `VERSION`，wheel 安装时读取 `mygpr` 的包元数据；
没有可用版本时显示 `unknown`，不回退到写死的旧版本号。

一致性检查覆盖包元数据、当前入口文档、源码清单和 CHANGELOG 的最新发行条目。
它可以发现发版时的同步遗漏；版本号递增与改动归类仍由每次发布负责。

运行：

```bash
python scripts/check_version_consistency.py
```

或直接运行测试：

```bash
python -m pytest tests/test_version_consistency.py -q
```
