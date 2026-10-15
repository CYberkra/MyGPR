# MyGPR 0.10.0

(UAV-)GPR 探地雷达数据处理、解释与 GIS 服务的桌面应用 + 无头后端。
PyQt6 + PyQt6-Fluent-Widgets 桌面前端，配分层后端（`mygpr/` 清洁架构 + `core/` 遗留内核）。

> 架构与工程约束：[CLAUDE.md](CLAUDE.md)；当前状态：[CURRENT_STATE.md](CURRENT_STATE.md)；交接规则：[DEV_HANDOFF.md](DEV_HANDOFF.md)；设计规格索引：[specs](docs/superpowers/specs/README.md)。

当前发行版：**v0.10.0**，提供 Windows x64 安装器、源码和 wheel，见[发布说明](docs/releases/v0.10.0.md)。

## 代码结构

- `ui/`：Qt 前端——`main_window.py` 注册 8 个导航页，`controllers/` 五个 QObject 控制器，
  `widgets/` B-scan/AScan/参数表单/地图/3D 轨迹控件；UI→core 唯一通道为 `ui/desktop_backend_facade.py`。
- `core/`：遗留内核——GPR 数据模型与 IO（mmap/分块）、项目存储、方法注册表、GIS、报告导出。
- `mygpr/`：分层后端——`interfaces/`（公共 API：`MyGPRBackend.create_default()`）→
  `application/`（按域分组服务）→ `domain/`（纯领域模型）→ `infrastructure/`（legacy 适配与持久化）。
- `PythonModule/`：算法包装模块；算法单一事实来源为
  `mygpr/infrastructure/processing/algorithms/methods` 的 `NATIVE_ALGORITHMS`。
- `scripts/`：质量门禁与治理脚本；`config/`：架构政策、schema 注册表、覆盖率/债务预算。
- `tests/`：pytest 用例，含 `tests/industrial/`（acceptance/performance/property/reliability/
  scientific_validation/static_contract）。
- `cli_batch.py`：无头批处理（`validate`/`run`/`resume`）；`backend_smoke.py`、
  `backend_project_smoke.py`：后端冒烟。

## 环境要求

Python **3.12–3.13**（钉版 `numpy==2.5.1` 要求 ≥3.12；CI 矩阵与 `requires-python` 已对齐）。

直接依赖基线如下；2026-10-14 本机 Windows `.venv` 为 **Python 3.12.0**，以下核心与 GUI 包均与安装版本一致。发行版的 Python 3.12/3.13 和 GUI 自动检查结果见[发布说明](docs/releases/v0.10.0.md)，依赖清单不作为最新版包推荐。

| 核心包（requirements.txt） | 版本 | GUI 包（requirements-gui.txt） | 版本 |
| --- | --- | --- | --- |
| numpy | 2.5.1 | PyQt6 | 6.11.0 |
| pandas | 2.3.3 | PyQt6-Qt6 | 6.11.1 |
| scipy | 1.18.0 | PyQt6-Fluent-Widgets | 1.11.2 |
| matplotlib | 3.11.0 | pyqtgraph | 0.14.0 |
| Pillow | 12.3.0 | PyOpenGL | 3.1.10 |
| h5py | 3.16.0 | colorlog | 6.12.0 |
| PyYAML | 6.0.3 | — | — |
| PyWavelets | 1.9.0 | — | — |
| pyproj | 3.7.2 | — | — |
| openpyxl | 3.1.5 | — | — |
| rasterio | 1.5.0 | — | — |
| fiona | 1.10.1 | — | — |
| defusedxml | 0.7.1 | — | — |

`requirements.txt` 默认安装无 Qt 后端；`requirements-core.txt` 是同一清单的兼容入口。`pyproject.toml` 的运行依赖与 `gui` extra 保持一致。开发工具版本见 [requirements-test-backend.txt](requirements-test-backend.txt)，构建工具见 [requirements-build.txt](requirements-build.txt)。`constraints/py313-*.lock` 是既有平台快照，不是当前 3.12 环境的完整哈希锁。

## 安装

Windows 用户可从 [v0.10.0 Release](https://github.com/CYberkra/MyGPR/releases/tag/v0.10.0)
下载 `MyGPR-0.10.0-Windows-x64-Setup.exe`，直接安装，无需另装 Python。
适用 Windows 10 1809+ / Windows 11 x64；当前安装器未签名，校验值见 Release 附件。

源码开发与 Python 包安装：

```powershell
py -3.12 -m venv .venv         # 也可选择 Python 3.13
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-gui.txt   # 桌面前端
python -m pip install -e .
```

Linux 使用 `python3.13 -m venv .venv` 和 `source .venv/bin/activate`，随后运行相同的 pip 命令。开发测试额外安装 `python -m pip install -r requirements-dev.txt`。

## 启动与验证

```powershell
python app_qt.py                                  # 桌面 GUI
python backend_smoke.py
python backend_project_smoke.py
python -m pytest tests/ -q
```

PowerShell 离屏冒烟：`$env:QT_QPA_PLATFORM='offscreen'; python app_qt.py --smoke`。截图默认写入系统临时目录的 `mygpr_shots/`，可用 `MYGPR_SMOKE_SHOTS_DIR` 指定。完整质量门禁：`python scripts/run_backend_quality_gate.py`。离屏冒烟不能代替 Windows 实机操作验收。

离屏检查后运行 `Remove-Item Env:QT_QPA_PLATFORM`，恢复正常桌面启动。

## 数据导入

项目测线导入入口直接支持 **CSV / TXT / NPY / NPZ / H5 / HDF5**。CDUT 采集 HDF5 会转换道×采样点方向，保留时窗、分量和逐道传感器缺失值；参见[数据格式参考](docs/user/reference/数据格式参考.md)。

RD3/RD7、DT1、DZT、SEG-Y、ENVI、ImpulseRadar 的底层读取器支持有限子集，但项目导入向导尚未接通这些二进制入口；格式被识别不代表能在向导中直接导入。CLI 批处理输入支持 CSV/TXT 和 H5/HDF5，配置需提供显式 `methods`；`recommended_profile` 已移除。

## SPEC 与打包

设计规格仍保存在 [docs/superpowers/specs/](docs/superpowers/specs/README.md)，旧三页设计为历史记录。当前没有根目录 `SPEC.md`。旧 `gpr_gui.spec` 已在 `a9fb92ef` 删除；现行配方为 [packaging/windows/MyGPR.spec](packaging/windows/MyGPR.spec) 和 Inno Setup 安装器。

Windows 构建工具钉版在 [requirements-windows-build.txt](requirements-windows-build.txt)：PyInstaller 6.22.3、hooks-contrib 2026.8、pip-licenses 5.5.5、cyclonedx-bom 5.5.0；Inno Setup 6.7.3。重建、真实安装验证和发布清单见 [Windows 构建说明](docs/releases/windows-build.md)。

## 二次开发入口

新前端/集成方使用 `mygpr/interfaces/` 公共边界、`mygpr/application/` 服务层与
`config/backend_api_v1.json` 契约；不要直接 import 持久化内部。

本次维护范围、验证和待办见 [2026-10-14 仓库说明核对记录](docs/artifacts/documentation_audit_20261014.md)。
