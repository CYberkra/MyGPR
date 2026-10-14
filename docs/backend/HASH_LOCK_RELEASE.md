# Hash-locked release dependencies — planned

Status checked 2026-10-14: `constraints/py313-linux.lock` and `constraints/py313-windows.lock` are existing version snapshots without archive hashes. The formerly documented `*-hashed.lock` files, `scripts/check_hash_locked_release.py`, and `MYGPR_REQUIRE_HASH_LOCKS` build gate are absent from the current tree.

For current source installation use [requirements.txt](../../requirements.txt), [GUI requirements](../../requirements-gui.txt), or the matching pyproject extras. Do not claim a hash-verified commercial release based on these version pins.

A future executable release must generate native-platform locks from the actual package archives, verify them in a clean environment, and add a real enforcement step to the build workflow. Hashes must never be invented from installed files or package names. See [Release process](../developer/RELEASE_PROCESS.md).
