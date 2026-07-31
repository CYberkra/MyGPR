# MyGPR v0.9.27 分层质量门禁重构

本版本不改变 v0.9.26 的现场生产业务合同，重点重构研发、合并、夜间和发布验证体系。

## 核心变化

- 中央影响图：`config/test_impact.toml`；
- 自动测试选择：`scripts/select_tests.py`；
- 统一质量门禁：`scripts/run_quality_gate.py`；
- 策略一致性检查：`scripts/check_test_policy.py`；
- pytest 自动层级/业务域标记；
- headless 测试合并执行，Qt/Matplotlib/VTK/gprMax 测试文件级隔离；
- PR、Windows GUI 冒烟、夜间四分片和 Linux/Windows 发布门禁；
- 机器可读的门禁计划与执行证据；
- Tiny/Standard/Stress 测试数据分层。
- 完成 `cylinder_single_v1` 场景/真值/预览固定资产与 GX-008 只读模型配对样例，消除测试顺序依赖；
- 测试运行时项目、日志与缓存迁移到临时目录，质量门禁不再污染发布源码树。

## 默认使用

```bash
python scripts/run_quality_gate.py affected --plan
python scripts/run_quality_gate.py affected
```

高风险的工程存储、schema、Job Manager 和传感器同步改动会自动提升到发布级自动回归。纯文档改动降为 L0；未知生产代码至少提升到合并门禁。
