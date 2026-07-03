# MyGPR History Memory Policy v0.7.6

MyGPR 的处理链路 stepper 需要保存若干历史 B-scan 快照以支持撤销、对比和临时回看。大规模 UAV-GPR 数据在多步骤处理后可能造成明显内存压力，因此 V0.7.6 加入历史快照内存预算。

## 环境变量

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `MYGPR_HISTORY_MAX_STEPS` | `10` | 最多保存多少个完整历史步骤 |
| `MYGPR_HISTORY_MAX_BYTES` | `268435456` | 完整历史步骤总内存预算，默认 256 MiB |
| `MYGPR_HISTORY_MAX_SNAPSHOT_BYTES` | `134217728` | 单个历史步骤最大保存尺寸，默认 128 MiB |
| `MYGPR_HISTORY_MAX_PRUNED_SUMMARIES` | `50` | 最多保留多少条被裁剪历史摘要 |

## 行为

- 仍保存完整数组的历史步骤可撤销、可在 stepper 中点击回看。
- 超过单快照上限的步骤不保存完整数组，只记录摘要。
- 超过总内存预算时，自动从最旧完整历史步骤开始裁剪。
- 被裁剪步骤会记录 label、shape、dtype、统计值、metadata summary 和裁剪原因。
- 报告包的 `manifest.json` 和 `processing_chain.json` 会记录 `history_memory`。

## Evidence 边界

历史内存裁剪不改变当前正式结果数组，也不改变处理算法输出。裁剪只影响历史回看 / 撤销能力。报告导出总是使用正式当前结果，不使用 stepper 临时预览。
