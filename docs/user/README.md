# MyGPR 用户文档

> 适用版本 v0.9.38+。文档按 [Diátaxis](https://diataxis.fr/) 四象限组织——按你的目的选择入口：

| 我想…… | 去哪里 |
|---|---|
| **从零学起**，第一次完整跑通处理流程 | [教程：第一次处理营山数据](tutorials/第一次处理营山数据.md) |
| **完成某个具体任务**（导入某格式、调参、备份……） | [任务指南](how-to/index.md) |
| **查某个细节**（格式矩阵、预设档、文件位置） | [参考手册](reference/index.md) |
| **理解原理**（证据链、数据安全模型） | [原理解释](explanation/处理证据链与数据安全.md) |

## 快速安装

```bash
python -m pip install -r requirements-core.txt   # 后端依赖（精确钉版）
python -m pip install -r requirements-gui.txt    # 图形界面依赖
python app_qt.py                                  # 启动桌面应用
```

无图形环境走 [无界面批处理](how-to/无界面批处理.md)。

## 快速链接

- 图形界面导览速览：[GUI_README](../../GUI_README.md)
- 开发者文档：[docs/developer/](../developer/) 与 [CLAUDE.md](../../CLAUDE.md)
- 更新历史：[CHANGELOG](../../CHANGELOG.md)
