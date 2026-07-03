# MyGPR v0.9.9 Windows 发布前稳定性与启动器整理

## 范围

本轮不改变算法数学含义、CSV 数据协议、坐标投影规则或报告导出 schema。目标是发布前稳定性收口：启动入口、环境检查、发布包清理、中文路径验证和关键链路回归。

## 变更

1. 新增 `scripts/check_env.py`：跨平台只读环境检查，覆盖 Python 版本、核心依赖、项目目录结构和日志目录可写性。
2. 新增 `scripts/run_app.py`：跨平台统一启动入口，设置 Qt、OpenGL、Matplotlib cache 和 faulthandler 默认值。
3. 更新 Windows 批处理脚本版本号到 `v0.9.9`，并补充 `pyproj` 环境检查。
4. 新增 `scripts/check_release_hygiene.py`：检查发布树中是否残留 `__pycache__`、`.pytest_cache`、`runtime_projects`、`logs` 等不应进入发布包的目录。
5. 发布 zip 打包时排除 cache、runtime、历史测试项目和 pyc 文件。

## 发布前人工验证清单

- 双击 `start_mygpr.bat` 启动；
- 双击 `check_mygpr_environment.bat` 检查依赖；
- 中文路径下解压并启动；
- 新建项目、单条导入、批量导入、取消后续导入；
- 运行数据质检；
- 保存处理结果；
- 生成报告包；
- 项目备份并确认 ZIP 可打开。

## 已知边界

- Windows 双击实际测试需要在用户本机完成；容器环境只能验证脚本语法、Python 启动入口和 offscreen GUI 截图。
- PDF 报告已纳入 v0.9.9 beta 报告包闭环；正式行业报告模板仍需继续细化。

## 发布包卫生补充

- 发布包根目录不得携带 `runtime_projects/`、`logs/`、缓存目录或字节码文件。
- `scripts/check_env.py` 和 `scripts/run_app.py` 在非 Windows 环境下使用用户状态目录保存日志和 Matplotlib 缓存，不再回退到发布包根目录创建 `logs/`。
