# MyGPR V0.7.4 local path configuration

MyGPR no longer uses developer workstation paths as active defaults. Configure local-only paths with environment variables instead.

## Environment variables

- `MYGPR_EVIDENCE_ROOT`: path to the sibling or external MyGPR-Evidence repository. Default fallback is `../MyGPR-Evidence` when it exists, then the user-writable MyGPR app data Evidence directory.
- `MYGPR_GPR_RESULT_RUNS`: optional local gprMax run inventory directory. Default fallback is the user-writable MyGPR output directory.
- `MYGPR_YINGSHAN_LINE9_CSV`: optional local YingShan Line9 field CSV used only by local diagnostic scripts.
- `MYGPR_GPRMAX_PYTHON` or explicit `--gprmax-python`: optional local gprMax Python runtime.
- `MYGPR_GPRMAX_ROOT`: optional local gprMax checkout root for benchmark scripts.
- `MYGPR_OBSIDIAN_VAULT`: optional local Obsidian vault for meeting/checkpoint helpers.

## Supported path placeholders

The gprMax campaign loader and research dashboard support these placeholders:

- `${MYGPR_REPO_ROOT}`
- `${MYGPR_EVIDENCE_ROOT}`
- `${MYGPR_GPR_RESULT_RUNS}`

Historical audit documents may still contain absolute paths to preserve reproducibility context. Active code, config, and launch scripts should not require those paths.
