# GX-UI-011/012 Product Polish and AutoTune Closure MVP

## Scope

This update applies a focused UI productization pass without changing processing algorithms or production AutoTune scoring.

Implemented areas:

1. Main workspace product polish
2. Unified Chinese terminology and status chips
3. AutoTune functional closure MVP

## Main workspace product polish

The main B-scan workspace now uses a productized plot card:

- title: current dataset name or empty-state title
- subtitle: current display stage and user guidance
- status chips: loaded state, processing stage, shape
- rounded white card container
- cleaner onboarding empty state

The Matplotlib plotting logic is unchanged.

## Unified terminology

The UI now prefers Chinese-facing terms:

- Raw/Input -> 原始数据
- Candidate -> 候选结果
- Recommendation -> 推荐结果
- Trial Table -> 候选记录表
- Metrics -> 指标
- Warnings -> 风险提示
- Claim Boundary -> 结论边界

## AutoTune closure MVP

The AutoTune page now forms a safe UI-local loop:

1. data metadata syncs from the main workspace
2. ROI/candidate/scoring controls update page state
3. Start Recommendation generates a UI-local recommendation preview
4. candidate ranking, recommendation panel, trial table and warnings update together

Important boundary:

- no production AutoTune execution
- no production scoring logic change
- no gprMax execution
- no Evidence mutation
- no PyVista/PyVistaQt dependency

## Known limitations

- B-scan thumbnail preview is still metadata-only.
- The recommendation score is a deterministic UI placeholder.
- Evidence export remains disabled.
- Legacy AutoTune page remains as compatibility layer.
