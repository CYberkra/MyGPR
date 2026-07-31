# MyGPR Backend TODO

当前后端基线：**v0.9.36**

## P0

1. 使用营山完整六测线数据执行后端 Golden 回归。
2. 完成多 GB 导入、取消、磁盘不足、异常断电与恢复测试。
3. 扩展厂商格式真实样例：RD3/RD7、DZT、DT1/HD、OKO、SEG-Y、ENVI。
4. 固化 `mygpr/interfaces/` 与 `config/backend_api_v1.json`，供新前端调用。

## P1

1. 继续拆分 `core/` 历史算法与基础设施适配器。
2. 完成 GIS、三维、制图和报告服务的无界面 API。
3. 增加项目自动保存、崩溃恢复和增量备份服务。
4. 完善 AutoTune 安全边界、候选流程空间和证据导出。

前端设计、Qt 页面和桌面打包不在本交付包范围内。
