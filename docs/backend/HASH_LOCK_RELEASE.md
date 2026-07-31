# Hash-locked release dependencies

Commercial release installation must use a platform-native, fully resolved pip lock with archive hashes:

- `constraints/py313-linux-hashed.lock`
- `constraints/py313-windows-hashed.lock`

Generate each file on the matching clean operating system with a resolver such as `pip-compile --generate-hashes`, review it, then run:

```bash
python scripts/check_hash_locked_release.py --platform linux --required
python scripts/check_hash_locked_release.py --platform windows --required
```

Set `MYGPR_REQUIRE_HASH_LOCKS=1` for commercial builds. The build then refuses to fall back to unhashed constraints. Hashes must come from the actual package archives/wheelhouse; they must never be fabricated from installed files or package names.
