# MyGPR 用户文档

> 适用版本 v0.12.0。软件内入口为 **设置 → 关于 → 使用手册**。

| 我想…… | 去哪里 |
|---|---|
| **从零学起**，第一次完整跑通处理流程 | [教程：第一次处理营山数据](tutorials/第一次处理营山数据.md) |
| **完成某个具体任务**（导入某格式、调参、备份……） | [任务指南](how-to/index.md) |
| **记录停车、干扰或目标范围**，按区间保留或剔除 | [标记与事件区间](how-to/标记与事件区间.md) |
| **查某个细节**（格式矩阵、依赖、文件位置） | [参考手册](reference/index.md) |
| **理解原理**（证据链、数据安全模型） | [原理解释](explanation/处理证据链与数据安全.md) |

项目页集中提供新建、打开、搜索和固定最近项目。处理页使用可拖动的 B-scan 标签，续算从所选已保存的输入开始。见 [工作台速览](../../GUI_README.md)和 [处理与成果续算](how-to/运行处理与AutoTune.md)。

## 快速安装

Windows x64 可从 [最新正式版](https://github.com/CYberkra/MyGPR/releases/latest) 下载安装器，安装包内置 Python 和 Qt。源码启动使用 Python **3.12–3.13**，依赖见 [README](../../README.md)。

```bash
python -m pip install -r requirements.txt        # 后端依赖（精确钉版）
python -m pip install -r requirements-gui.txt    # 图形界面依赖
python app_qt.py                                  # 启动桌面应用
```

无图形环境走 [无界面批处理](how-to/无界面批处理.md)。

## 快速链接

- 图形界面导览速览：[GUI_README](../../GUI_README.md)
- 开发者文档：[docs/developer/](../developer/) 与 [CLAUDE.md](../../CLAUDE.md)
- 更新历史：[CHANGELOG](../../CHANGELOG.md)
