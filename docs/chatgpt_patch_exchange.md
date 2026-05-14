# ChatGPT Patch Exchange Protocol

This note defines the reliable handoff format for patches generated outside
Codex and applied by `tools/watch_chatgpt_patch_inbox.py`.

## Current Rule

Every patch must declare the exact repository base it was generated from:

```text
Base-Commit: <current codex/uavgpr-workflow-refactor commit hash>
```

The patch tool compares that value with current `HEAD` after `git pull
--ff-only`. If the value is missing or stale, the tool stops before applying the
patch and writes a failure report under:

```text
.chatgpt_inbox/_reports/
```

## Why This Is Necessary

Many web-generated patches are synthetic unified diffs. They may contain fake
`index` hashes such as `aaaaaaa`, `bbbbbbb`, or `c0ffee0`, so `git apply
--3way` cannot reconstruct the real base version. When the web-side context is
older than the branch, no local watcher can safely infer the intended edit.

The correct fix is to regenerate the patch from the current commit instead of
hand-merging stale hunks repeatedly.

## Web GPT Prompt Template

Use this before asking for another patch:

```text
Repository: CYberkra/MyGPR
Branch: codex/uavgpr-workflow-refactor
Base-Commit: <paste latest commit hash here>

Before generating a patch:
1. Fetch the current files from this exact commit.
2. Generate a unified diff against this exact base.
3. Put `Base-Commit: <hash>` as the first line.
4. If you cannot fetch current files, stop and ask me to provide them.
5. Do not reuse old snippets or invent fake git index hashes.
```

## Legacy Escape Hatch

For an old patch that intentionally has no base header, run manually with:

```powershell
python tools/apply_chatgpt_patch.py --branch codex/uavgpr-workflow-refactor --patch-file <patch> --mygpr-smoke --allow-missing-base
```

Do not use that mode for routine workflow. It is only for emergency local
experiments.
