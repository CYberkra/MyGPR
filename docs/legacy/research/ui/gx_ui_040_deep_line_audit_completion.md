# GX-UI-040 Deep Line-Level Audit Completion

Base version: 0.8.65  
Output version: 0.8.66

This pass responds to the requested deeper audit after V0.8.65. It used a line-level scanner over auditable package source/text files, then fixed the remaining user-visible coverage gaps found during triage.

## Completed

- Added standard runtime `requirements.txt` and made `requirements-dev.txt` depend on it.
- Updated installer and docs to use runtime requirements for normal environment setup.
- Hardened gprMax simulation validation GPU-device input: malformed device text now blocks command generation/copy.
- Implemented `cli_batch.py resume --summary <summary.json>` for failed-job reruns.
- Reworded unsupported method/family errors so they are clear guardrails, not apparent unfinished work.
- Archived audit summary and raw findings under `docs/audits/`.

## Boundaries

- No processing algorithm changes.
- No AutoTune scoring changes.
- No Evidence schema changes.
- No direct long-running gprMax run from the GUI.
