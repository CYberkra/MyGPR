# NP-005 No-prior Mode Status and Integration Audit

## 1. Executive Summary

MyGPR no-prior mode now has an Evidence-backed policy chain, UI/export state integration, and action guardrails in main UI paths.

Current positioning:

- It is a **risk-aware safety and claim-boundary mechanism** for users without ROI/target prior.
- It is **not** a target detector.
- It is **not** an AutoTune performance claim.

## 2. Evidence Basis

This implementation is grounded in the accepted Evidence chain below (MyGPR-Evidence repo):

1. `field/DS-001R_scoped_yingshan_yaan_csv_inventory/`
   - Role: scoped field CSV inventory and raw QC.
   - Proves: local YingShan/YaAn CSV lines can be enumerated and reconstructed for raw preview/decay QC.
   - Does not prove: underground truth, target type, processing superiority.

2. `field/DS-002_field_preview_selection_no_prior_qc_set/`
   - Role: representative line selection for no-prior/visual QC.
   - Proves: selected small candidate set is available for no-prior baseline use.
   - Does not prove: target correctness or processing effectiveness.

3. `field/NP-001_no_prior_qc_metric_baseline/`
   - Role: no-prior raw QC metrics and warning baseline.
   - Proves: warning-oriented metrics can classify selected lines as high-risk.
   - Does not prove: thresholds are universally validated.

4. `field/NP-002_no_prior_processing_safety_pilot/`
   - Role: safety policy pilot after high-risk no-prior warnings.
   - Proves: conservative behavior contract (warn first, block aggressive auto recommendations, manual review).
   - Does not prove: processing improvement, AutoTune superiority, or target detection.

5. NP-003 (MyGPR source integration)
   - Role: policy model + UI/export/diagnostic exposure.
   - Proves: no-prior policy is encoded and visible in app outputs.
   - Does not prove: full action-path enforcement coverage.

6. NP-004 (MyGPR source integration)
   - Role: main UI action guardrails + guard event logging.
   - Proves: core `app_qt.py` action paths enforce block/confirm policy in high-risk no-prior conditions.
   - Does not prove: all legacy/parallel UI execution entries are guarded.

7. NP-006 (MyGPR source integration)
   - Role: Workbench legacy template execution guard callback wrapper.
   - Proves: Workbench template execution now reuses main-window no-prior guard source of truth.
   - Does not prove: every Workbench legacy method path is fully guarded.

## 3. Current MyGPR Implementation

### Policy model

- `core/no_prior_qc_policy.py`
  - no-prior level derivation (`ok/caution/high_risk`)
  - action policy matrix
  - claim boundary and Chinese warning templates

### Guardrail helper

- `core/no_prior_ui_guardrails.py`
  - action-level decision (`allowed/caution/blocked/requires_confirmation`)
  - guard event payload builder

### Main UI integration points

- `app_qt.py`
  - UI quality labels/drawer summary
  - quality report runtime section
  - quality snapshot export payload
  - replay evidence app_context payload
  - diagnostics copy payload
  - action guards on main AutoTune/recommendation/method/pipeline entry points
  - guard event list tracking (`_no_prior_guard_events`)

### Workbench legacy integration points

- `ui/gui_workbench.py`
  - template execution (`_on_template_execute`) now calls `_guard_workbench_action("workflow_run", ...)`
  - `set_no_prior_guard_callback(...)` allows Workbench to reuse main guard decision
  - callback missing path now shows explicit warning/confirmation instead of silent execution
- `app_qt.py`
  - injects Workbench callback via `_enforce_workbench_no_prior_action_guard(...)`
  - callback reuses `_enforce_no_prior_action_guard(...)` and main event logging path

### Tests

- `tests/test_no_prior_qc_policy.py`
- `tests/test_no_prior_ui_guardrails.py`

## 4. Action Policy Table (Current)

| Action | Policy | Notes |
| --- | --- | --- |
| `raw_preview` | allowed | baseline non-invasive preview |
| `contrast_clip_display` | caution (display-only) | not amplitude-preserving |
| `conservative_energy_decay_gain_display` | caution (display-only) | requires warning/confirmation path in high-risk |
| `AGC_display_only` | caution (display-only) | not amplitude-preserving, no target claim |
| `background_suppression_conservative` | requires confirmation / caution | high-risk no-prior should not run silently |
| `background_suppression_aggressive` | blocked | high-risk no-prior guardrail |
| `dewow` | blocked in no-prior safety pilot | pending later validation |
| `migration` | blocked in no-prior safety pilot | out of no-prior safety scope |
| `AutoTune` | blocked | high-risk no-prior |
| `preset_recommendation` | blocked | high-risk no-prior |
| `workflow_run` | requires confirmation | not globally blocked, but manually acknowledged |

## 5. Claim Boundary

### Allowed

- raw QC warnings
- conservative display-first recommendations
- manual-review-required enforcement
- display-only enhancement caveat
- exportable no-prior warning/guard metadata

### Prohibited

- target detection claims
- underground correctness claims
- AutoTune superiority claims
- preset promotion claims
- amplitude-preserving claims after display gain/AGC
- universal commercial QC claims

## 6. Known Gaps

1. Workbench template path is guarded, but legacy Workbench single-method execution/preview paths still need a complete per-action guard audit.
2. Guard thresholds are still heuristic (not yet externally calibrated/validated).
3. No dedicated full manual UI smoke evidence package yet.
4. No user study or expert-evaluation protocol yet.
5. No field ground-truth validation.
6. No-prior warning does not replace expert interpretation.
7. Evidence repo retains old broad `DS-001_local_field_dataset_inventory` historical noise risk (not part of scoped DS-001R baseline).

## 7. Recommended Next Tasks (Ranked)

1. **NP-006B**: Workbench single-method execution guard audit and minimal routing so non-template legacy paths share no-prior guard semantics.
2. **UI-SMOKE-001**: Manual no-prior UI smoke checklist and result capture.
3. **EV-001**: Evidence repository cleanup plan for broad historical noise artifacts.
4. **AT-017**: scoring/risk-penalty what-if diagnostics (separate from no-prior guardrail semantics).
5. **Paper-outline update**: add no-prior safety mode as a risk-control contribution.

## 8. Suggested Paper Wording

Suggested conservative statement:

> The no-prior mode does not attempt target detection. It provides risk-aware processing guardrails and claim-boundary enforcement when no target prior or ROI is available.

## 9. Suggested UI Wording (Chinese)

- 当前数据触发高风险质控告警，建议先查看原始剖面并进行人工复核。
- 系统不会自动推荐激进背景抑制或默认参数。
- 以下显示增强仅用于可视化，不代表幅值保真或目标识别。
- 未提供目标区域或先验信息，因此本结果不构成地下目标判断。
- 建议由有经验人员复核后再尝试任何参数化处理。
