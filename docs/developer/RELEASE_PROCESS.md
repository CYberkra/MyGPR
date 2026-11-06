# Release process

GitHub Actions builds and verifies the source package, wheel, and Windows installer. The publisher signs the installer locally and publishes the reviewed GitHub Release with the signed update feed. See [Application updates](../releases/application-updates.md).

## Existing checks and package build

1. Run the backend, GUI, and clean-install jobs in GitHub Actions as described in [Testing strategy](TESTING_STRATEGY.md).
2. Use `python scripts/check_version_consistency.py` for a local static check of VERSION, package metadata, and documentation.
3. Run **MyGPR Windows packages**. The workflow uses a clean Python 3.13 environment and preserves the wheel used to build the Windows release package.
4. Record the tested commit, workflow URLs, dependencies, and package hashes with the release.

## GitHub source/wheel release

1. Choose an unused semantic version following [versioning policy](versioning_policy.md).
   Update VERSION, pyproject, current entry documents, source manifest and CHANGELOG;
   add a versioned note under `docs/releases/`.
2. Submit the version change on a `codex/` branch, review it, and merge its PR after all five CI jobs pass.
   Verify the merged main commit's CI before publishing.
3. Export the exact merged commit with `git archive` and download the wheel built from that commit by GitHub Actions.
4. Verify the exported version and the wheel's METADATA; install the wheel in an isolated environment
   and check GUI/report version lookup and package entry points outside the source checkout.
5. Create `MyGPR_V<version>_source.zip`, the wheel, `SHA256SUMS.txt` and `release-manifest.json`.
   Record the exact source commit, dependency versions and successful CI URL. Verify hashes and ZIP integrity.
6. Prepare a GitHub Release draft targeting that merged commit. Attach the reviewed notes, source archive, wheel, verified installer, manifests, checksums, and `appcast.xml`. Publish after review, then verify the tag target, downloaded hashes, and stable feed URL.

The first GitHub Release prepared using this process is [v0.10.0](../releases/v0.10.0.md).
Existing baseline/rollback tags remain historical references, not release version numbers.

## Windows installer build and verification

The current recipes are `packaging/windows/MyGPR.spec` (PyInstaller onedir) and `MyGPR.iss` (Inno Setup), driven by `build.py` / `build.ps1`. Follow [Windows build instructions](../releases/windows-build.md) for the supported build interpreter and pinned tools. The historical `gpr_gui.spec` deleted in `a9fb92ef` is not the current recipe.

Build from an immutable source ref via `git archive` in an isolated environment. The recipe includes VERSION, configuration, dynamic algorithms and package metadata, checks native dependency origins, generates license/SBOM inventories, and runs the frozen backend check before producing the installer. `verify_install.py` checks installation, package/HDF5 evidence, seven-page native smoke, reinstall, displayed version, normal GUI close, uninstall and preservation of user files in an owned QA directory.

The publisher uses the repository-pinned public key to sign and verify the installer payload with Ed25519. The GitHub **update_installation** regression verifies the complete legacy installer upgrade and the signed HTTPS update through a real Windows installer, with evidence of normal GUI exit and relaunch. Store the application commit, recipe hashes, verification reports, and checksums with the installer. Record Windows Authenticode status separately in the build manifest.

For a commercial/industrial release, complete the required Windows, CUDA and hardware evidence in `config/industrial_acceptance_matrix.json` before claiming acceptance. Daily source gates and local smoke do not replace that matrix; see [Testing strategy](TESTING_STRATEGY.md).
