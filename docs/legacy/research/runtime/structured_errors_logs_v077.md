# MyGPR V0.7.7 结构化错误与日志规范

## 目标

MyGPR 的运行日志需要同时满足两类需求：

1. 用户界面中简洁可读的事件流。
2. Evidence 报告包中可审计、可检索、可归档的结构化 sidecar。

V0.7.7 引入 `runtime_events.json`，用于保存结构化日志事件和最近的结构化错误。

## 错误类型

新增基础错误类型：

- `InputDataError`
- `ProcessingMethodError`
- `EvidenceExportError`
- `GprMaxConversionError`
- `AutoTuneScoringError`

所有错误都可转换为 `mygpr.error_info.v1`：

```json
{
  "schema": "mygpr.error_info.v1",
  "error_type": "InputDataError",
  "error_code": "MYGPR_INPUT_DATA_ERROR",
  "category": "input_data",
  "user_message": "CSV 加载失败",
  "technical_detail": "...",
  "hint": "...",
  "recoverable": true,
  "context": {}
}
```

## 运行事件

`runtime_events.json` 使用 schema `mygpr.runtime_events.v1`。

每条事件包含：

- `timestamp`
- `event_type`: `SYS / INFO / DATA / METHOD / WARN / ERR / EXPORT`
- `level`: `info / warning / error`
- `source`
- `message`
- `context`

## 当前边界

V0.7.7 先建立结构化日志和错误 sidecar 基线。GUI 层仍保留部分宽捕获，后续再逐步替换成更精确的异常类型。
