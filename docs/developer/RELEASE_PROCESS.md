# Release process

Reviewed against the source tree on 2026-10-17. Source/wheel installation and a local Windows installer build/verification workflow are present. Release publishing remains a reviewed GitHub operation; there is no independent automated release workflow in CI.

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

## Windows installer build and verification

The current recipes are `packaging/windows/MyGPR.spec` (PyInstaller onedir) and `MyGPR.iss` (Inno Setup), driven by `build.py` / `build.ps1`. Follow [Windows build instructions](../releases/windows-build.md) for the supported build interpreter and pinned tools. The historical `gpr_gui.spec` deleted in `a9fb92ef` is not the current recipe.

Build from an immutable source ref via `git archive` in an isolated environment. The recipe includes VERSION, configuration, dynamic algorithms and package metadata, checks native dependency origins, generates license/SBOM inventories, and runs the frozen backend check before producing the installer. `verify_install.py` checks installation, package/HDF5 evidence, seven-page native smoke, reinstall, displayed version, normal GUI close, uninstall and preservation of user files in an owned QA directory.

Preserve the application commit, recipe hashes, environment inventory, checksums and verification evidence alongside the installer, without overwriting the original source/wheel release manifest. A successful build alone is not installation or industrial acceptance. Current installers are unsigned; signing, independent release CI, and broader upgrade/rollback/system coverage remain separate work. The frozen `--package-check` command uses an explicit process exit and does not establish natural interpreter cleanup; the GUI close and smoke checks are separate evidence.

For a commercial/industrial release, complete the required Windows, CUDA and hardware evidence in `config/industrial_acceptance_matrix.json` before claiming acceptance. Daily source gates and local smoke do not replace that matrix; see [Testing strategy](TESTING_STRATEGY.md).
