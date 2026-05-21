# NP-003 No-prior Warning UI / Spec

## Scope

NP-003 implements a UI/spec-facing no-prior QC safety policy in MyGPR.  
It is based on accepted evidence:

- NP-001: no-prior raw QC metric baseline
- NP-002: no-prior processing safety pilot

This work does **not** modify processing algorithms, AutoTune scoring, or motion compensation semantics.

## No-prior policy model

`core/no_prior_qc_policy.py` defines:

- warning level: `ok | caution | high_risk`
- context flags: `target_prior_available`, `roi_available`
- safety decisions:
  - `safe_auto_recommendation_allowed`
  - `aggressive_background_suppression_allowed`
  - `amplitude_claim_allowed`
  - `auto_gain_allowed`
  - `manual_review_required`
  - `recommended_initial_policy`
- action-policy matrix (allowed / caution / blocked)
- claim boundary and Chinese user-facing warning templates

For `high_risk` with no target prior and no ROI:

- auto recommendation is blocked
- aggressive background suppression is blocked
- amplitude claim is blocked
- auto gain recommendation is blocked
- manual review is required
- initial policy is `conservative_display_only`

## UI integration points

In `app_qt.py`, no-prior policy is now surfaced in:

- runtime quality drawer labels (`no_prior_level`, `no_prior_policy`, blocked actions)
- quality summary text (quality/export page)
- generated Markdown report runtime section
- copied diagnostics payload
- quality snapshot JSON payload
- replay evidence app context payload

## Action policy (UI/spec)

- allowed: `raw_preview`
- caution (display-only): `contrast_clip_display`, `conservative_energy_decay_gain_display`, `AGC_display_only`, `background_suppression_conservative`
- blocked: `background_suppression_aggressive`, `dewow` (pilot scope), `migration`, `AutoTune`, `preset_recommendation`

## Claim boundary

- No target detection claim
- No underground correctness claim
- No AutoTune performance claim
- No preset promotion
- Display transforms are not amplitude-preserving claims
- Thresholds are heuristic unless later validated

## Follow-up

Suggested next task: NP-004 integrate this policy into explicit workflow-stage guardrails and warning UI affordances (without changing core algorithms).

## NP-004 Guardrail Behavior

NP-004 adds action-level UI guardrails driven by the same no-prior policy model.

### Guarded actions

- blocked (high-risk + no target prior + no ROI):
  - `AutoTune`
  - `preset_recommendation`
  - `background_suppression_aggressive`
- requires confirmation (high-risk):
  - `AGC_display_only`
  - `conservative_energy_decay_gain_display`
  - `background_suppression_conservative`
  - `workflow_run`
- allowed:
  - `raw_preview`

### UI behavior

- blocked actions show warning dialog and do not execute.
- confirmation actions show warning dialog and require explicit user continue.
- warning text uses NP-002 Chinese templates and keeps display-only caveat.

### Event logging

Each blocked/confirmed guard decision records a lightweight event with:

- timestamp
- action_id
- decision
- no_prior_level
- reason
- manual_review_required
- override_used

Events are included in diagnostics, report runtime state, quality snapshot JSON, and replay evidence app context.

### Current limitation

Guardrails are wired to confirmed main-window action handlers (`app_qt.py`).  
If future code introduces new action entry points outside these handlers, they still need explicit guard wiring.

## NP-006 Workbench Legacy Guard Wrapper

NP-006 adds a minimal Workbench callback wrapper to prevent legacy template execution from silently bypassing no-prior guardrails.

- `app_qt.py` now injects a Workbench callback (`_enforce_workbench_no_prior_action_guard`) that reuses `_enforce_no_prior_action_guard`.
- `ui/gui_workbench.py` calls this callback before `_on_template_execute(...)` with action id `workflow_run`.
- if callback denies action, template execution is blocked.
- if callback is missing, Workbench shows a warning/confirmation dialog before execution to avoid silent bypass.

This keeps main-window no-prior policy as the source of truth and avoids duplicating policy logic in Workbench.
