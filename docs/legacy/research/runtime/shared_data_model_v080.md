# SharedDataModel / SharedDataQtAdapter 架构说明（V0.8.0）

V0.8.0 将 MyGPR 共享数据状态拆为两层：

```text
core/shared_data_model.py        # 纯 Python 数据模型
ui/shared_data_qt_adapter.py     # Qt 信号适配层
core/shared_data_state.py        # 兼容 shim
```

## 使用原则

### Headless / core / CLI

优先使用：

```python
from core.shared_data_model import SharedDataModel
```

该入口不依赖 PyQt6，可在普通 unit test、CLI batch、Evidence replay 等环境下使用。

### GUI

继续使用：

```python
from core.shared_data_state import SharedDataState
```

或直接使用：

```python
from ui.shared_data_qt_adapter import SharedDataQtAdapter
```

GUI 层会得到 `changed` Qt signal。

## 事件通知

`SharedDataModel` 通过 Python listener 提供轻量通知：

```python
state = SharedDataModel()
events = []
state.add_change_listener(events.append)
state.load_data(data)
```

`SharedDataQtAdapter` 在此基础上额外发射 Qt signal：

```python
state.changed.connect(on_changed)
```

## 兼容边界

`core/shared_data_state.py` 仍保留历史名称 `SharedDataState`，以避免一次性改动大量 GUI、脚本和测试 import。

长期目标是逐步将非 GUI 代码改为直接依赖 `SharedDataModel`。
