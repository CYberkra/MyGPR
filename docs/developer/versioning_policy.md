# MyGPR 版本迭代规则

MyGPR 后续交付包采用语义化版本号：`MAJOR.MINOR.PATCH`。

## 版本号含义

- `MAJOR`：破坏兼容性的大版本，例如数据格式或主工作流不可兼容变化。
- `MINOR`：新增功能或较大模块升级，例如 AutoTune workflow planner。
- `PATCH`：bugfix、UI 小改、文档、测试、合规清理、轻量结构拆分。

## 每次交付必须同步更新

1. 根目录 `VERSION`。
2. `CHANGELOG.md` 中新增对应版本条目。
3. 交付 zip 文件名，例如 `MyGPR_V0.8.38_<tag>.zip`。
4. 若打包 exe，`gpr_gui.spec` 必须把 `VERSION` 带入发布包。
5. 修复报告或开发报告中必须写明 base version 和 output version。

## 默认递增规则

- 崩溃修复、测试修复、文档修复：递增 `PATCH`。
- 新增可见功能但不破坏兼容性：递增 `MINOR`。
- 数据格式、插件契约、Evidence manifest 产生破坏性变化：递增 `MAJOR`，并写迁移说明。

## 自动检查

运行：

```bash
python scripts/check_version_consistency.py --expected 0.8.38
```

或直接运行测试：

```bash
python -m pytest tests/test_version_consistency.py -q
```
