# Release process

Reviewed against the source tree on 2026-10-14. Source/wheel installation is supported; an automated executable release pipeline is not currently present.

## Existing checks and package build

1. Use Python 3.12–3.13 and install the runtime, development and build requirement profiles.
2. Run `python scripts/run_backend_quality_gate.py` (the backend CI checklist), plus the GUI jobs described in [Testing strategy](TESTING_STRATEGY.md).
3. Check `python scripts/check_version_consistency.py`; VERSION and pyproject.toml must agree.
4. In a clean build environment run `python -m pip wheel . --wheel-dir dist`. CI checks package installation and wheel construction.
5. Preserve checks, dependency versions and source revision with the release. Smoke checks do not replace Windows user acceptance.

## Work still required for an exe release

The old `gpr_gui.spec` was deleted in `a9fb92ef` (2026-07-31). There is no current PyInstaller spec, isolated four-shard release runner, signing job or hash-lock checker. See the [specification index](../superpowers/specs/README.md).

Before distributing an exe, establish and validate resource collection, native dependencies, VERSION inclusion, a supported smoke argument (`--smoke` exists in app_qt.py), dependency-license/SBOM evidence, signatures/checksums, and Windows install/upgrade/rollback/uninstall. These are release requirements, not claims that tools already exist. Build-tool dependencies alone do not implement this workflow.
