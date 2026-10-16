# 工作台改版记录 — 2026-10-16

范围为用户确认的首页/项目融合、处理页重置、B-scan 标签吸附、结果续算分叉、
项目树与全应用对象菜单、统一视觉。现行行为见 [工作台合同](../superpowers/specs/2026-10-16-workbench-redesign.md)。
实现位于 `codex/maintenance-20261016` 工作区，基线 `bea6e5f`；原维护改动保留。
没有变更依赖、算法实现或版本，没有提交、推送、合并或重新打包安装。

## 代码落点

| 范围 | 主要实现 |
| --- | --- |
| 项目三状态与导航 | `ui/main_window.py`、`ui/pages/project_page.py`、`home_page.py`、`ui/coordinator_project.py` |
| 标签排序、移动、拆分与视图恢复 | `ui/widgets/bscan_workspace.py` |
| 输入/草稿/冻结任务、历史与重试 | `ui/pages/processing_page.py`、`processing_steps.py`、`processing_menus.py`、`ui/coordinator_processing.py` |
| 直接来源、分支与清理 | `mygpr/application/project/processing_service.py`、`core/processing_artifact_index.py`、存储/目录/删除模块 |
| 项目树与菜单 | `ui/file_tree.py`、`ui/widgets/file_tree_panel.py`、各业务页对象菜单、共享键盘菜单辅助 |

## 验证记录

完整本机日志、脚本与截图位于忽略目录 `output/ui-redesign-20261016/`。
以下证据针对源码；大型 GUI 截图未放入 Git。

- 项目/navigation 回归 95 项；标题栏及设置隔离回归 12 项。
- 首次全量为 1575 passed、6 skipped、6 failed、27 errors；失败均为已明确退役的
  自动布局、原始输入不可关闭、结果画廊或 ChainStrip 页面合同。对应测试迁移到真实
  新工作区行为，35 项独立组件测试及所属类源码、AST 保持一致。
- 独立审查的旧菜单跨项目误操作、损坏 metadata 右键崩溃、legacy 中间来源清理
  缺口均有 RED→GREEN 证据；树深链新增 1050 节点回归。
- Windows 原生独立项目首轮使用 pywinauto/UIA 和真实鼠标：新建→导入 256×64
  float32 NPY→搜索/右键添加方法→执行真实后台任务→选旧成果另建分支→核验树。
  原生标签拖到右缘拆分、拖回合并、关闭/恢复、面板开关、浅深主题均通过，进程正常退出 0。
  方法双击添加未纳入通过结论，真实右键添加已验证。
- 架构、编译、Schema/API、ruff、mypy/复杂度/债务预算和发行卫生检查已通过。
  新增超长模块已按步骤编辑/对象菜单拆分，没有放宽预算。

最终收口：

- `verified-full-suite.log`：**1637 passed、6 skipped，261.02 秒，实际退出码 0**。
  新增项目程序化选择回归另跑 9 项通过。此前带覆盖率全量为 1625 passed、6 skipped，
  覆盖率策略通过；最终 UI 修复另有对应完整回归，未用旧覆盖率声称新增 UI 覆盖率。
- `app_qt.py --smoke` 已将八页旧断言更新为七页，并生成七页及整体共八张截图，退出码 0。
  smoke 仍使用既有硬退出策略，不能代替原生关闭验证。
- 最终 Windows 原生补验的 Qt PID 50076：连续三次历史运行重配后取消均保留草稿，
  拖动拆分/合并后布局文字同步，正常关闭后 `app.exec()` 与父进程均退出 0。
  `desktop-acceptance/acceptance.json` 共 22 项：20 passed、1 passed_after_fix、1 not_verified。
  未验证项为方法双击；真实搜索和右键添加已通过。
- 原生补验曾复现三次确认框取消后的 `0xc0000005`。独立最小实验确认菜单在模态
  回调未返回时提前销毁；共享菜单现等完整派发退栈后释放，鼠标/键盘、主/子菜单共
  12 次真实取消测试通过，并验证最终释放和同步动作语义。原失败证据保留，未以 pytest
  已有退出诊断掩盖这个应用运行期缺陷。
- 最终独立 UI 审查确认已报告的问题均已修复，没有尚未收口的重要发现。
  切测线布局恢复、跨组目标显示状态、迟到原始预览焦点均有 RED→GREEN 回归。
- 合成存储复核：源 NPY 与项目原始 HDF5 逐元素一致；四成果均 `(256,64)`、float32、
  全有限值；直接父链为 A→B→C、A→D。这里只证明流程和存储完整性。
- 最终源码检查：架构、592 文件编译、Schema/API、ruff、mypy/复杂度/债务预算、
  发行卫生、格式兼容、核心与项目后端 smoke、`pip check` 通过。
  项目后端 smoke 的 Windows 临时目录清理警告保留，数据与进程检查通过。
- 原 `TODO.md`、发行卫生脚本、设置隔离和空 B-scan 维护改动与实施前字节一致。
  `git diff --check` 排除原用户 TODO 空白后通过，不擅自清理用户改动。

## 边界与保留事项

- pytest 已存在的 Windows 关停阶段 access violation 在部分测试输出之后仍出现；
  断言通过数、进程退出码和该诊断分别记录，未屏蔽或宣称修复。
- UI 验收使用合成小数据，不是实测 UAV-SFCW 科学验证；没有多 GB、硬件/传感器
  闭环、物理分辨率或断电验收。现有六项外部数据/可选回退测试仍按条件跳过。
- 全树分页/虚拟化尚未实现；读取预览仍按可见视图和窗口采样，深链遍历已改迭代。
- 一处旧标题栏测试 fixture 曾沿用默认真实设置路径，造成测试 MRU 路径写入。
  已隔离 fixture，并给整个测试进程默认设置隔离；只清除确认的测试路径，未触碰项目数据。
  原最近项目列表无可靠快照，不能声称完全恢复了历史 MRU。
