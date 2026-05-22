# AutoTune Current Status and Next Steps

## 1) Executive Summary

AutoTune research in this branch has moved from single-scene parameter exploration into a multi-scene, gate-controlled validation workflow.  
AT-013 expanded replay coverage across GX-003/004/005/006, and AT-014 strengthened metric fidelity with scene-specific metrics.  
The preset-finalization gate remains **blocked**.  
Current outputs are suitable for diagnosis, method discussion, and paper-safe process claims, but **not** for preset promotion.

## 2) Evidence Timeline

- **AT-006**: zero-time policy hardening for native gprMax-converted validation context.
- **AT-008A**: reduced primary lane with no zero-time + no dewow.
- **AT-009**: background + gain policy refinement in reduced lane.
- **AT-011**: relative trace-count-aware background-window candidate policy.
- **AT-012**: design assumption audit and preset-finalization gates.
- **GX-004/005/006**: native benchmark expansion (no-target, multi-target, layered/background).
- **AT-013**: multi-scene replay of AT-011 relative policy.
- **AT-014**: metric-fidelity upgrade (GX-004 no-target control, GX-005 strict per-target, GX-006 layer/interface-aware).

## 3) Current Validated Points

- Fixed absolute `ntraces` is not safe as a universal default.
- Relative AT-011 candidate generation is more defensible than fixed absolute windows.
- In the current native validation primary lane, zero-time and dewow should remain excluded.
- GX-004 false-positive-control metrics do not currently indicate artifact risk in AT-014.
- GX-005 shows target imbalance risk under strict per-target processed metrics.
- GX-006 shows layer/interface suppression risk under layer-aware metrics.
- `energy_decay_gain` remains a conservative/interpretable gain lane; it is not proof of amplitude-preserving recovery.

## 4) Current Non-Validated / Forbidden Claims

- No overall AutoTune superiority claim.
- No field-performance validation claim.
- No universal preset claim.
- No “local n=9 is globally optimal” claim.
- No amplitude-preserving claim for gain outputs.
- No full 3D / UAV field-scale claim from current thin-2D synthetic gprMax scenes.

## 5) Interpretation of Local n=9

`local n=9` is the current best pattern under AT-014 metric-fidelity validation across GX-003/004/005/006.  
This is evidence against prematurely promoting `near_full_line` as default.  
It is **not** sufficient evidence to promote `local n=9` as a preset.  
It should be treated as a candidate for further risk-controlled testing.

## 6) Gate Status and Blockers

Current gate status: **blocked**

### `gx005_target_imbalance`
- Meaning: candidate behavior is not consistently balanced between shallow/deep (or target A/B) preservation/contrast.
- Why this blocks preset finalization: a preset that benefits one target while degrading another is not robust.
- Needed to clear: evidence of stable target-balance behavior across scenes/candidates, with bounded imbalance metrics.

### `gx006_interface_suppression`
- Meaning: layer/interface structure can be excessively attenuated in layered/complex background scenes.
- Why this blocks preset finalization: removing meaningful interfaces can produce misleading “cleaner” outputs.
- Needed to clear: interface-aware protection criteria and evidence that selected candidates preserve key interface structure.

### `synthetic_thin2d_scene_limit`
- Meaning: current evidence is still constrained by synthetic thin-2D scene assumptions.
- Why this blocks preset finalization: external validity to broader field conditions is not established.
- Needed to clear: additional scene diversity and stronger transfer checks, including non-ROI/no-prior QC design and broader context validation.

## 7) Recommended Next Tasks (Ranked)

1. **AT-016 Risk-flag / scoring diagnostics (docs-first or bounded runner)**  
   Purpose: explain why `local n=9` ranks best while GX-005/GX-006 risks remain; separate ranking behavior from risk gate behavior without changing scoring semantics.

2. **DS-002 External GPR/radar decay-profile comparison**  
   Purpose: compare Ying Shan field-line shallow/deep decay against public references and assess typicality.

3. **Field no-prior QC design**  
   Purpose: define non-expert AutoTune workflow without ROI, with global QC metrics, warning flags, and anomaly proposal.

4. **Paper-candidate outline**  
   Purpose: frame the paper around evidence-driven gated validation workflow, not AutoTune superiority.

## 8) Recommended Immediate Next Task

**Recommended now: AT-016 Risk-flag and scoring diagnostics design.**  
Reason: it directly addresses the current contradiction between candidate ranking and blocker flags, while preserving current scoring and algorithm boundaries.

## 9) Suggested Claim Language for Paper

### Safe wording examples
- “ROI-aware constrained parameter validation”
- “multi-scene gated validation”
- “evidence-driven workflow”
- “risk flags prevent premature preset promotion”

### Forbidden wording examples
- “AutoTune finds globally optimal parameters”
- “AutoTune outperforms manual processing”
- “local n=9 is universally optimal”
- “field target correctness is validated”

## 10) Handoff Summary

- Current source commit: `90e45fa5bb4cf857080d544149fe846c5af34ded`
- Current evidence commit: `e8487142ecd71f90970667bf18704b556b5cc957`
- Accepted status: AT-013 and AT-014 accepted; metric-fidelity synthesis complete.
- Gate status: **blocked**
- Next recommended task: **AT-016 Risk-flag and scoring diagnostics design**

## 11) AT-021 Update (No-prior Background Labeling Only)

AT-021 introduces non-blocking recommendation labeling for no-prior high-risk background AutoTune outcomes:

- no scoring change
- no algorithm change
- no candidate filtering
- no modal popup / no hard block in this path

Risky background outcomes are now explicitly marked as diagnostic/manual-review recommendations in logs and export metadata (instead of being phrased as validated default recommendations).

This is a wording/metadata safety layer. It does not upgrade scientific claims and does not validate field correctness.
