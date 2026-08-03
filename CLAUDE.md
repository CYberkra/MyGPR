# MyGPR v0.9.37 — UAV-GPR 基覆界面勘探工作台

## 项目概述

MyGPR 是面向 GPR / UAV-GPR 野外勘探项目的桌面软件，PyQt6 + qfluentwidgets。
流程：项目建档 → 测线导入与预检 → 雷达/RTK/IMU 同步 → 测线处理 → 界面标注 → 空间成果 → 成果报告。

主界面七个导航页：`主页 / 项目管理 / 处理工作台 / 界面解释 / 成果交付 / 任务中心 / 设置`。

## 运行与测试

```bash
# 环境：Python 3.12 虚拟环境在 .venv（系统默认 Python 3.10 跑不起来）
source .venv/Scripts/activate          # Git Bash
python app_qt.py                       # 启动 GUI
QT_QPA_PLATFORM=offscreen python app_qt.py --smoke   # 离屏截图到 /tmp/mygpr_shots/
python -m pytest tests/ -q             # 测试
python cli_batch.py --help             # 无头批处理入口
```

## 架构分层

```text
app_qt.py                      # GUI 入口（DPI PassThrough、主题、--smoke）
ui/                            # PyQt6 前端
  main_window.py               # FluentWindow 组装器（信号 handler 分发至 window_mixins.py）
  window_mixins.py             # 信号 handler mixin：项目/测线/导入预检/处理/解释/成果/任务
  desktop_backend_facade.py    # ui→core/domain/application 统一导入通道（架构门禁例外）
  pages/                       # 七个页面，纯展示 + 发信号
  widgets/                     # BScanView / CollapsiblePanel / LogPanel 等
  controllers/                 # Qt 控制器：run_worker 后台线程 + 信号回主线程
mygpr/                         # 后端分层（新代码走这里）
  interfaces/backend.py        # MyGPRBackend.create_default()
  application/                 # acquisition / processing / project / ... 服务
  domain/                      # 领域模型
  infrastructure/              # 适配器，调用 core/ 遗留内核
core/                          # 遗留内核（仍活跃）：gpr_data_model、field_*、
                               # sensor_sync、chunked_gpr_io、storage_primitives
PythonModule/                  # 算法包装器，经 method registry + cli_batch 动态加载
cli_batch.py                   # 无头批处理入口（pyproject: mygpr-batch）
tests/                         # pytest；sample_data/ 是测试夹具，勿删
```

## 工程约束

- 长任务走 controller `run_worker` + JobBridge：协作式取消、阶段进度、信号回主线程；工作线程不得直接碰 Qt 控件。
- 大文件用 mmap/分块 I/O（`core/chunked_gpr_io.py`），不得为预检/显示整体读入 RAM。
- GPR 存储 float32；文件先写隐藏临时文件再原子替换（`core/storage_primitives.py`）。
- Windows 注意：只读句柄 `os.fsync` 会抛 Errno 9，统一用 `fsync_file()`（已容忍不支持的平台）。
- `PythonModule/` 方法靠注册表动态 import，静态 grep 不到引用≠死代码；删除前对照 `core/method_registry_metadata.py`。
- `scripts/run_mutation_contract.py`、`generate_schema_catalog.py`、`audit_test_redundancy.py` 被 `config/schema_catalog.json` 治理注册，不可移动。

## 测试与质量

```bash
python scripts/check_python_compile.py
python scripts/check_architecture.py
```

GUI/DPI、多 GB 真实数据和传感器文件必须在 Windows 目标机验收。

## Agent skills

### Issue tracker

Issues 与 PRD 追踪在 GitHub Issues（CYberkra/MyGPR），操作用 `gh` CLI。详见 `docs/agents/issue-tracker.md`。

### Triage labels

使用 mattpocock/skills 默认五角色标签：`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`。详见 `docs/agents/triage-labels.md`。

### Domain docs

单上下文布局：仓库根部 `CONTEXT.md` + `docs/adr/`（由 domain-modeling 类技能按需懒创建）。详见 `docs/agents/domain.md`。
