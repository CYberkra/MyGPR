# MyGPR 0.9.37

(UAV-)GPR 探地雷达数据处理、解释与 GIS 服务的桌面应用 + 无头后端。
PyQt6 + PyQt6-Fluent-Widgets 桌面前端，配分层后端（`mygpr/` 清洁架构 + `core/` 遗留内核）。

> 权威架构文档：`CLAUDE.md`（中文）；当前状态：`CURRENT_STATE.md`；交接规则：`DEV_HANDOFF.md`。

## 代码结构

- `ui/`：Qt 前端——`main_window.py` 注册 8 个导航页，`controllers/` 五个 QObject 控制器，
  `widgets/` B-scan/AScan/参数表单/地图/3D 轨迹控件；UI→core 唯一通道为 `ui/desktop_backend_facade.py`。
- `core/`：遗留内核——GPR 数据模型与 IO（mmap/分块）、项目存储、方法注册表、GIS、报告导出。
- `mygpr/`：分层后端——`interfaces/`（公共 API：`MyGPRBackend.create_default()`）→
  `application/`（按域分组服务）→ `domain/`（纯领域模型）→ `infrastructure/`（legacy 适配与持久化）。
- `PythonModule/`：算法包装模块；算法单一事实来源为
  `mygpr/infrastructure/processing/algorithms/methods` 的 `NATIVE_ALGORITHMS`。
- `scripts/`：质量门禁与治理脚本；`config/`：架构政策、schema 注册表、覆盖率/债务预算。
- `tests/`：656+ 用例，含 `tests/industrial/`（acceptance/performance/property/reliability/
  scientific_validation/static_contract）。
- `cli_batch.py`：无头批处理（`validate`/`run`/`resume`）；`backend_smoke.py`、
  `backend_project_smoke.py`：后端冒烟。

## 环境要求

Python **3.12–3.13**（钉版 `numpy==2.5.1` 要求 ≥3.12；CI 矩阵与 `requires-python` 已对齐）。

## 安装

```bash
python -m venv .venv
. .venv/Scripts/activate       # Linux/macOS: . .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-gui.txt   # 桌面前端
python -m pip install -e .
```

## 启动与验证

```bash
python app_qt.py                                  # 桌面 GUI
QT_QPA_PLATFORM=offscreen python app_qt.py --smoke  # GUI 离屏冒烟
python backend_smoke.py && python backend_project_smoke.py
python -m pytest tests/ -q
```

## 二次开发入口

新前端/集成方使用 `mygpr/interfaces/` 公共边界、`mygpr/application/` 服务层与
`config/backend_api_v1.json` 契约；不要直接 import 持久化内部。
