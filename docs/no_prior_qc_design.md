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
