#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MR-001 merge readiness audit for promoting research baseline into main."""

# MR-001 Merge Readiness Audit (Research -> main)

## 1. Executive Summary

- ready_for_fast_forward_merge: **yes**
- final decision: **proceed**
- merge type: **fast-forward only**
- main HEAD before merge: `289c843d4c7a7d208070a4882d9c558bb2dfeaf1`
- research HEAD (pre-report): `75c2f8ea641f31ea7b9349a3ddd925ba9e0f3e28`
- branch relation: `origin/main...origin/codex/research-gprmax-autotune = 0 258`
- validation summary: local gates passed; latest research CI run succeeded.

## 2. Scope of Merge

This merge promotes the current research baseline into main, including:

- CI and preflight workflow hardening
- processing engine / methods registry evolution
- motion_compensation_v2 and trace metadata integration
- AutoTune framework, diagnostics utilities, and comparison/export paths
- gprMax benchmark and package tooling
- no-prior QC model, guardrails, and recommendation labels (AT-021 non-blocking)
- UI-STAB stream improvements for Workflow Studio direction
- Workbench legacy fallback guard integration
- expanded docs/tests for research validation chain

## 3. Not in Scope

- MyGPR-Evidence is **not** merged.
- No Evidence artifacts are imported into MyGPR by this merge.
- This is **not** a final public release.
- This is **not** an AutoTune superiority claim.
- This is **not** field-performance validation.
- Workbench is **not** removed.

## 4. Algorithm Compatibility Summary

Reference: MR-ALG-001 (`docs/merge_audit_algorithm_output_compatibility.md`)

- no P0 blockers found
- deterministic fixture comparison mostly identical
- metadata-aware baseline differences are expected
- shape-changing methods are by design (`time_cut` / resample paths)

Required merge note wording:

“This merge promotes the current research processing baseline to main. Some processing outputs may differ from the previous main baseline due to metadata-aware runtime parameter injection, updated background suppression behavior, AGC compatibility changes, and UAV motion compensation V2 resampling semantics. These differences are documented and expected.”

## 5. UI Status

- UI-STAB-001 completed.
- UI-SMOKE-001 status: partial_accept (documented offscreen/local smoke evidence).
- Real desktop interactive smoke remains recommended.
- Workbench remains legacy fallback and is not deleted.

## 6. AutoTune Status

- No claim of universal AutoTune superiority.
- AT-019/AT-020 identified YingShan background suppression risk semantics.
- AT-021 added non-blocking recommendation labels and warning metadata only.
- no-prior high-risk background recommendations are advisory/manual-review labeled, not ground-truth decisions.

## 7. Risk Classification

### P0
- none

### P1
- merging main to research baseline changes historical default behavior surface
- metadata-aware processing can produce differences vs old main
- UI is research-stable but not final polish release
- no-prior risk labels are heuristic by design

### P2
- Evidence repository cleanup remains a separate track
- real desktop UI smoke should be repeated post-merge
- Workbench retirement is not complete

## 8. Final Recommendation

- ready_for_fast_forward_merge: **yes**
- proceed with `--ff-only` merge from `origin/codex/research-gprmax-autotune` into `main`.

