# Release process

Reviewed against the source tree on 2026-10-14. Source/wheel installation is supported; an automated executable release pipeline is not currently present.

## Existing checks and package build

1. Use Python 3.12–3.13 and install the runtime, development and build requirement profiles.
2. Run `python scripts/run_backend_quality_gate.py` (the backend CI checklist), plus the GUI jobs described in [Testing strategy](TESTING_STRATEGY.md).
3. Check `python scripts/check_version_consistency.py`; VERSION and pyproject.toml must agree.
4. In a clean build environment run `python -m pip wheel . --wheel-dir dist`. CI checks package installation and wheel construction.
5. Preserve checks, dependency versions and source revision with the release. Smoke checks do not replace Windows user acceptance.

## GitHub source/wheel release

1. Choose an unused semantic version following [versioning policy](versioning_policy.md).
   Update VERSION, pyproject, current entry documents, source manifest and CHANGELOG;
   add a versioned note under `docs/releases/`.
2. Submit the version change on a `codex/` branch, review it, and merge its PR after all five CI jobs pass.
   Verify the merged main commit's CI before publishing.
3. Export the exact merged commit with `git archive`; build the wheel from that clean export in an
   isolated build environment. Do not package local TODO edits, raw input data, output or wheel caches.
4. Verify the exported version and the wheel's METADATA; install the wheel in an isolated environment
   and check GUI/report version lookup and package entry points outside the source checkout.
5. Create `MyGPR_V<version>_source.zip`, the wheel, `SHA256SUMS.txt` and `release-manifest.json`.
   Record the exact source commit, dependency versions and successful CI URL. Verify hashes and ZIP integrity.
6. Create an annotated `v<version>` tag at that merged commit, push the tag, and create the GitHub Release
   with the reviewed notes and verified assets. Verify the published tag target and downloaded asset hashes.

The first GitHub Release prepared using this process is [v0.10.0](../releases/v0.10.0.md).
Existing baseline/rollback tags remain historical references, not release version numbers.

## Work still required for an exe release

The old `gpr_gui.spec` was deleted in `a9fb92ef` (2026-07-31). There is no current PyInstaller spec, isolated four-shard release runner, signing job or hash-lock checker. See the [specification index](../superpowers/specs/README.md).

Before distributing an exe, establish and validate resource collection, native dependencies, VERSION inclusion, a supported smoke argument (`--smoke` exists in app_qt.py), dependency-license/SBOM evidence, signatures/checksums, and Windows install/upgrade/rollback/uninstall. These are release requirements, not claims that tools already exist. Build-tool dependencies alone do not implement this workflow.
