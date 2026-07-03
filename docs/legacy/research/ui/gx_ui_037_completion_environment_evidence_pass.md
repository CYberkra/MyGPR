# GX-UI-037 Completion, Environment, and Evidence Drawer Pass

本轮针对上传包中仍显得“没收口”的可见问题做小范围补全，不改变算法、AutoTune 评分、gprMax 运行或 Evidence 文件格式。

## 发现的问题

1. Windows 启动器已经检查 `pywt`，但 `check_mygpr_environment.bat` 未检查该依赖，环境诊断结果可能和启动结果不一致。
2. 用户需要“下载/安装所需环境”时，根目录只有检查与启动脚本，没有面向普通用户的本地 `.venv` 安装入口。
3. 新工作台的多个 Matplotlib 页面含中文标题；当页面被测试或工具直接导入时，未必经过 `app_qt.py` 的字体初始化，Linux/offscreen 环境会出现中文缺字警告。
4. 成果交付页可以生成报告与证据包，但主工作台底部“证据”抽屉没有在生成后列出产物，视觉上像入口未完成。

## 完成内容

- 新增 `install_mygpr_environment.bat` / `安装MyGPR环境.bat`：
  - 自动选择 Python 3；
  - 创建或复用当前包内 `.venv`；
  - 升级 pip/setuptools/wheel；
  - 安装 `requirements-dev.txt`。
- `check_mygpr_environment.bat` 补充 `pywt` 依赖检查，与启动器保持一致。
- 新增 `ui/matplotlib_fonts.py`，并在 `ui/__init__.py` 中统一初始化 Matplotlib CJK 字体兜底；同时让 `ui/gui_base.py` 复用该 helper。
- `MyGPRWorkbenchWindow` 现在监听 `DeliveryPage.package_built`：
  - 自动刷新工程树；
  - 将 manifest、report、checksums、spatial synthesis 和 evidence index 写入底部“证据”表；
  - 切换到底部证据抽屉，给用户明确完成反馈。

## 验证

- `python scripts/preflight_check.py`
- `python -m pytest -q tests/test_workbench_ui.py`
- `python -m pytest -q tests/test_processing_lab_ui.py`
- `python -m pytest -q tests/test_delivery_page_ui.py`
- `python -m pytest -q tests/test_autotune_tuning_page_target_response.py`

## 边界

- 不修改处理算法。
- 不修改 AutoTune scoring。
- 不运行 gprMax。
- 不修改 Evidence schema。
- 不引入 PyVista / PyVistaQt。
