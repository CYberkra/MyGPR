# MyGPR Round5：已有算法接入新测线处理页

本轮目标是把 1080P 工作台的“测线处理”页面接入已有算法注册表与统一处理引擎。

## 范围

- 新增 `core/field_processing_bridge.py`
- 测线处理页算法分类来自 `core.methods_registry.PROCESSING_METHODS`
- 右侧参数面板根据算法参数 schema 动态生成
- “预览 / 应用处理 / 保存处理结果”调用已有 `core.processing_engine.run_processing_method`
- 处理结果继续保存到 Round3/Round4 固化的项目目录结构
- 不接入 `QUICK_PRESETS`
- 不接入 `workflow_executor`
- 不提供默认多算法流水线
- 不在本轮处理页暴露会改变道间距/道数的采集间距整理功能

## UI 文案原则

右侧卡片标题为“处理设置”，不使用“单算法处理”作为用户可见描述。用户看到的是：

1. 算法分类
2. 选择算法
3. 参数设置
4. 当前执行信息

## 数据流

```text
FieldWorkbenchWindow
  -> GPRDataSet
  -> field_processing_bridge.run_registered_method
  -> processing_engine.prepare_runtime_params
  -> processing_engine.run_processing_method
  -> methods_registry.PROCESSING_METHODS
  -> PythonModule / legacy adapter
  -> GPRDataSet(processed)
  -> FieldProjectStore.save_processed_line
```

## 验证

```bash
python -m py_compile core/field_processing_bridge.py ui/field_workbench_window.py app_qt.py
python -m pytest tests/test_field_project_store.py tests/test_round4_data_interfaces.py tests/test_field_processing_bridge.py -q
```

当前验证结果：9 passed。
