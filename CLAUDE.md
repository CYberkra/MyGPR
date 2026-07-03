# MyGPR v0.9.24 — 探地雷达勘探定位工作台

## 项目概述

MyGPR 是面向 GPR / UAV-GPR 勘探项目的桌面软件，基于 PyQt6 + qfluentwidgets。
核心流程：项目建档 → 测线导入 → 数据检查 → 测线处理 → 目标标注 → 空间定位 → 成果报告。

默认主界面五个页面：`项目管理 -> 测线处理 -> 目标定位 -> 空间成果 -> 成果报告`

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| GUI | PyQt6 6.6+ + PyQt6-Fluent-Widgets 1.8+ |
| 科学计算 | numpy 1.24+, scipy 1.10+, pandas 2.0+, matplotlib 3.7+ |
| 数据处理 | h5py 3.10+, PyWavelets 1.5+, pyproj 3.6+ |
| 测试 | pytest 8.0+ (标记: unit, gui, integration, slow, wavelet) |
| 打包 | PyInstaller (gpr_gui.spec) |
| 环境 | conda + pip (environment.yml + requirements.txt) |

## 项目结构

```
app_qt.py              # 主入口 (4000行) — PyQt6 GUI 应用
cli_batch.py           # CLI 批处理入口
core/                  # 核心算法和业务逻辑 (176文件)
  auto_tune.py         #   参数自动推荐引擎 (3400行)
  processing_engine.py #   处理管道编排
  gpr_io.py            #   GPR 数据 I/O (DZT, SEGY, HDF5)
  app_errors.py        #   结构化错误处理 (MyGPRError + ErrorInfo)
  benchmark_registry.py#   性能基准注册表
ui/                    # PyQt6 GUI 界面 (114文件)
  autotune_tuning_page.py  # 参数调优页面 (2600行)
  gui_base.py          #   基础工具和函数
  gui_basic_flow.py    #   基础流程页面
  gui_quality_log.py   #   质量与日志页面
tests/                 # 测试套件 (233文件)
  conftest.py          #   共享 fixtures
  fixtures/            #   测试数据
docs/                  # 文档 (266文件)
config/                # 应用配置
configs/               # 预设配置文件
sample_data/           # 示例 GPR 数据
runtime_projects/      # 运行时项目输出
scripts/               # 工具脚本
PythonModule/          # Kirchhoff 偏移模块
```

## 编码约定

### 数值精度
- GPR 信号处理内部计算使用 float64；存储格式使用 float32（GPRDataSet.matrix 默认 dtype=np.float32）。处理引擎输出已调整为 float64。
- 警惕整数除法 (`//`) 在信号处理中的精度丢失

### 错误处理
- 使用 `core/app_errors.py` 中的 `MyGPRError` 基类
- 所有异常包装为 `ErrorInfo` dataclass
- 遵循 `error_info_from_exception()` 模式

### 处理管道顺序
```
dewow → gain → filter → migration → time-depth conversion
```
严格保持此顺序，不可跳过或重排。

### 测试
- `-m unit` — 核心算法和纯函数（快速，无 GUI）
- `-m gui` — PyQt6 UI 类测试（需要 QApplication）
- `-m integration` — 多模块工作流测试
- `-m slow` — 耗时测试
- `-m wavelet` — PyWavelets 相关测试
- 新功能必须配测试，改算法必须跑 `pytest -m unit`

### GUI 线程安全
- 重计算必须在 QThread/QWorker 中执行
- 禁止从工作线程直接更新 GUI
- matplotlib FigureCanvas 注意生命周期管理

## 关键文件参考

| 文件 | 行数 | 用途 |
|------|------|------|
| app_qt.py | 4000 | 主窗口、菜单、全局日志 |
| core/auto_tune.py | 3400 | 自动参数推荐核心 |
| ui/autotune_tuning_page.py | 2600 | 参数调优 GUI |
| core/processing_engine.py | - | 处理管道引擎 |
| core/gpr_io.py | - | 数据文件读写 |

## 环境激活

```bash
conda activate mygpr
# 或
pip install -r requirements.txt
```
